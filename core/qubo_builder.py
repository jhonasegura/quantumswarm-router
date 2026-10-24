r"""
core/qubo_builder.py
====================
Traduce la asignación misión→dron a un QUBO / Hamiltoniano de Ising.

Variables binarias
------------------
    x_{i,k} = 1  si la misión i la realiza el dron k        (N·M variables)

Función objetivo (todo cuadrático en x, por tanto QUBO-nativo)
--------------------------------------------------------------
    H = H_coste + H_compacidad + H_conflicto + H_batería + P·H_one-hot

  H_coste      = Σ_{i,k} [ t(base_k→i) + urgencia_{ik} ] · x_{ik}
                 Tiempo de aproximación y retraso estimado si la ventana
                 de la misión cierra antes de que el dron pueda llegar.

  H_compacidad = λ_c Σ_k Σ_{i<j} t_k(i→j) · x_{ik} x_{jk}
                 Misiones lejanas en el mismo dron cuestan (aproxima la
                 longitud de ruta sin codificar el orden de visita).

  H_conflicto  = γ Σ_{k≠l} Σ_{i<j} w_{ij} · x_{ik} x_{jl}
                 w_ij > 0 si las misiones i, j están cerca con ventanas de
                 tiempo solapadas, o si los tramos base_k→i y base_l→j se
                 cruzan. Es una INTERACCIÓN ENTRE DRONES: justo lo que un
                 VRP clásico no modela (coste de arco = un solo vehículo).

  H_batería    = λ_b Σ_k ( (L_k + Σ_i e_{ik} x_{ik}) / C_k )²
                 Restricción de capacidad como penalización cuadrática
                 SUAVE. Ventaja: no requiere variables de holgura. Una
                 desigualdad Σ e·x ≤ C convertida con slack binario añadiría
                 ~log2(C) qubits por dron. L_k es la carga ya fijada por
                 lotes anteriores (descomposición).

  H_one-hot    = Σ_i ( Σ_k x_{ik} − 1 )²
                 Restricción DURA "cada misión exactamente a un dron". La
                 declaramos como restricción de igualdad en un
                 QuadraticProgram y QuadraticProgramToQubo la convierte en
                 penalización con peso P.

Cómo se elige P (la parte delicada)
-----------------------------------
  Si P es pequeño, el óptimo del QUBO puede violar one-hot; si es enorme, el
  paisaje energético se vuelve abrupto y QAOA (con pocas capas) concentra
  poca probabilidad en el subespacio factible. Usamos la cota
      P = 1.5 · max_u ( |a_u| + Σ_v |b_uv| ) + 1
  que garantiza que activar/desactivar una variable para romper one-hot
  nunca compensa, sin inflar P más de lo necesario.

Descomposición en lotes
-----------------------
  N·M qubits (p.ej. 10×3 = 30) es demasiado para simular QAOA con un bucle
  variacional en un portátil (2^30 amplitudes). Partimos las misiones en
  lotes espacialmente coherentes (barrido angular) de ≤ max_qubits/M
  misiones. Las misiones ya asignadas entran en los lotes siguientes como
  términos LINEALES (campo externo), igual que en las técnicas de
  descomposición usadas en hardware real (p.ej. qbsolv / D-Wave hybrid).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from .scenario import Scenario

log = logging.getLogger(__name__)

# Pesos base de la formulación (los sliders de la UI los escalan)
W_COMPACT = 0.35
W_CONFLICT_BASE = 2.0
W_BATTERY_BASE = 30.0
W_URGENCY = 2.0


@dataclass
class QuboBatch:
    mission_ids: list[int]
    n_drones: int
    labels: list[str]
    linear: np.ndarray         # QUBO completo (con penalización): término lineal
    quad: np.ndarray           # triangular superior estricta
    const: float
    penalty: float
    backend: str               # "qiskit-optimization" | "numpy (fallback)"
    qubo_qp: Any = None        # QuadraticProgram convertido (si qiskit disponible)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_qubits(self) -> int:
        return len(self.linear)

    def var(self, b: int, k: int) -> int:
        return b * self.n_drones + k

    def symmetric_matrix(self) -> np.ndarray:
        """Matriz Q simétrica (diag = término lineal) para visualizarla."""
        Q = (self.quad + self.quad.T) / 2
        Q[np.diag_indices_from(Q)] = self.linear
        return Q


# ----------------------------------------------------------------------------
# Grafo de conflictos (networkx): misiones que comparten espacio aéreo/tiempo
# ----------------------------------------------------------------------------
def conflict_graph(scn: Scenario) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(scn.N))
    D = scn.dist_matrix()
    radius = 4 * scn.safety_radius
    for i in range(scn.N):
        for j in range(i + 1, scn.N):
            mi, mj = scn.missions[i], scn.missions[j]
            d = D[scn.mission_node(i), scn.mission_node(j)]
            prox = max(0.0, 1 - d / radius)
            if prox <= 0:
                continue
            ov = max(0.0, min(mi.tw_end, mj.tw_end) - max(mi.tw_start, mj.tw_start))
            width = max(1e-6, min(mi.tw_end - mi.tw_start, mj.tw_end - mj.tw_start))
            w = prox * (0.5 + 0.5 * min(1.0, ov / width))
            G.add_edge(i, j, weight=w)
    return G


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, q1) != orient(p1, p2, q2)) and (orient(q1, q2, p1) != orient(q1, q2, p2))


def make_batches(scn: Scenario, max_qubits: int) -> list[list[int]]:
    """Barrido angular alrededor del centro -> lotes espacialmente coherentes."""
    if max_qubits < scn.M:
        raise ValueError(f"max_qubits ({max_qubits}) debe ser ≥ número de drones ({scn.M})")
    size = max(1, max_qubits // scn.M)
    c = scn.area / 2
    order = sorted(range(scn.N), key=lambda i: np.arctan2(scn.missions[i].y - c, scn.missions[i].x - c))
    return [order[s:s + size] for s in range(0, len(order), size)]


def energy_estimate(scn: Scenario, mission_id: int, drone_id: int) -> float:
    d = scn.drones[drone_id]
    m = scn.missions[mission_id]
    dist = scn.dist_matrix()[d.id, scn.mission_node(mission_id)]
    return dist * scn.energy_per_km(d) + m.service_time * scn.hover_rate(d)


# ----------------------------------------------------------------------------
# Construcción del QUBO de un lote
# ----------------------------------------------------------------------------
def build_batch_qubo(
    scn: Scenario,
    batch: list[int],
    fixed: dict[int, int],
    loads: np.ndarray,
    graph: nx.Graph | None = None,
) -> QuboBatch:
    M, B = scn.M, len(batch)
    n = B * M
    D = scn.dist_matrix()
    G = graph if graph is not None else conflict_graph(scn)
    gamma = W_CONFLICT_BASE * scn.conflict_penalty
    lam_b = W_BATTERY_BASE * scn.wear_factor

    lin = np.zeros(n)
    Q = np.zeros((n, n))     # triangular superior estricta
    const = 0.0

    def v(b, k):
        return b * M + k

    def add_q(u, w, val):
        if u == w:
            lin[u] += val
        else:
            Q[min(u, w), max(u, w)] += val

    def conflict_w(i, j, k, l):
        """Peso de conflicto si i va con dron k y j con dron l (k≠l)."""
        w = G[i][j]["weight"] if G.has_edge(i, j) else 0.0
        a, b = scn.drones[k].depot, scn.drones[l].depot
        if _segments_cross(a, (scn.missions[i].x, scn.missions[i].y),
                           b, (scn.missions[j].x, scn.missions[j].y)):
            mi, mj = scn.missions[i], scn.missions[j]
            ov = max(0.0, min(mi.tw_end, mj.tw_end) - max(mi.tw_start, mj.tw_start))
            w += 0.5 * (1.0 if ov > 0 else 0.3)
        return w

    # --- H_coste: término lineal -------------------------------------------
    for b, i in enumerate(batch):
        m = scn.missions[i]
        for k, d in enumerate(scn.drones):
            t_arr = D[k, scn.mission_node(i)] / d.speed
            lin[v(b, k)] += t_arr + W_URGENCY * m.priority * max(0.0, t_arr - m.tw_end)

    # --- H_compacidad y H_conflicto dentro del lote (cuadráticos) -----------
    for b1 in range(B):
        for b2 in range(b1 + 1, B):
            i, j = batch[b1], batch[b2]
            dij = D[scn.mission_node(i), scn.mission_node(j)]
            for k in range(M):
                add_q(v(b1, k), v(b2, k), W_COMPACT * dij / scn.drones[k].speed)
                for l in range(M):
                    if l != k:
                        w = conflict_w(i, j, k, l)
                        if w > 0:
                            add_q(v(b1, k), v(b2, l), gamma * w)

    # --- Interacción con misiones ya fijadas: se vuelve lineal --------------
    for f, l in fixed.items():
        for b, i in enumerate(batch):
            dif = D[scn.mission_node(i), scn.mission_node(f)]
            lin[v(b, l)] += W_COMPACT * dif / scn.drones[l].speed
            for k in range(M):
                if k != l:
                    w = conflict_w(i, f, k, l)
                    if w > 0:
                        lin[v(b, k)] += gamma * w

    # --- H_batería: (L_k + Σ e x)^2 / C_k^2, expandido con x² = x ----------
    for k, d in enumerate(scn.drones):
        s = lam_b / d.battery ** 2
        e = [energy_estimate(scn, i, k) for i in batch]
        const += s * loads[k] ** 2
        for b in range(B):
            lin[v(b, k)] += s * (2 * loads[k] * e[b] + e[b] ** 2)
            for b2 in range(b + 1, B):
                add_q(v(b, k), v(b2, k), 2 * s * e[b] * e[b2])

    labels = [f"m{i}→D{k + 1}" for i in batch for k in range(M)]

    # --- Peso de penalización one-hot ---------------------------------------
    Qs = Q + Q.T
    penalty = float(1.5 * np.max(np.abs(lin) + np.abs(Qs).sum(1)) + 1.0)
    if not np.isfinite(penalty):
        raise ValueError("Coeficientes QUBO no finitos: revisa los parámetros del escenario")

    # --- Conversión: qiskit-optimization con verificación + fallback numpy ---
    warnings: list[str] = []
    try:
        ql, qq, qc, qubo_qp = _convert_with_qiskit(lin, Q, const, labels, B, M, penalty)
        # Verificación cruzada contra la expansión manual en estados aleatorios
        ml, mq, mc = _onehot_penalty_numpy(lin, Q, const, B, M, penalty)
        rng = np.random.default_rng(0)
        for _ in range(8):
            x = rng.integers(0, 2, n)
            e1 = qc + ql @ x + x @ qq @ x
            e2 = mc + ml @ x + x @ mq @ x
            if not np.isclose(e1, e2, rtol=1e-6, atol=1e-6):
                raise RuntimeError("La conversión de qiskit no coincide con el modelo")
        backend = "qiskit-optimization"
    except ImportError:
        ql, qq, qc = _onehot_penalty_numpy(lin, Q, const, B, M, penalty)
        qubo_qp, backend = None, "numpy (fallback)"
        warnings.append("qiskit-optimization no disponible: QUBO construido con numpy.")
    except Exception as exc:  # noqa: BLE001 - nunca romper la UI por la conversión
        log.warning("Fallo en QuadraticProgramToQubo: %s", exc)
        ql, qq, qc = _onehot_penalty_numpy(lin, Q, const, B, M, penalty)
        qubo_qp, backend = None, "numpy (fallback)"
        warnings.append(f"Conversión qiskit falló ({exc}); se usó la expansión manual.")

    return QuboBatch(
        mission_ids=list(batch), n_drones=M, labels=labels,
        linear=ql, quad=qq, const=qc, penalty=penalty,
        backend=backend, qubo_qp=qubo_qp, warnings=warnings,
    )


def _convert_with_qiskit(lin, Q, const, labels, B, M, penalty):
    """Modelo con restricciones -> QUBO usando qiskit-optimization."""
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo

    names = [f"x_{j}" for j in range(len(lin))]  # nombres seguros (sin '→')
    qp = QuadraticProgram("quantumswarm_batch")
    for nm in names:
        qp.binary_var(name=nm)
    quad = {(names[u], names[w]): float(Q[u, w])
            for u, w in zip(*np.nonzero(Q))}
    qp.minimize(constant=float(const), linear=[float(a) for a in lin], quadratic=quad)
    for b in range(B):
        qp.linear_constraint(
            linear={names[b * M + k]: 1 for k in range(M)},
            sense="==", rhs=1, name=f"onehot_{b}",
        )
    qubo = QuadraticProgramToQubo(penalty=penalty).convert(qp)
    if qubo.get_num_vars() != len(lin):
        raise RuntimeError("El conversor añadió variables inesperadas")
    ql = np.asarray(qubo.objective.linear.to_array(), dtype=float)
    raw = np.asarray(qubo.objective.quadratic.to_array(), dtype=float)
    return _fold(ql, raw) + (float(qubo.objective.constant), qubo)


def _fold(lin: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza a (lineal, triangular superior estricta). x_i² = x_i."""
    lin = lin + np.diag(Q)
    upper = np.triu(Q, 1) + np.tril(Q, -1).T
    return lin, upper


def _onehot_penalty_numpy(lin, Q, const, B, M, P):
    """Expansión manual de P·Σ_i(Σ_k x_ik − 1)² = P·Σ_i(−Σ_k x_ik + 2Σ_{k<l} x_ik x_il + 1)."""
    lin, Q = lin.copy(), Q.copy()
    for b in range(B):
        for k in range(M):
            lin[b * M + k] -= P
            for l in range(k + 1, M):
                Q[b * M + k, b * M + l] += 2 * P
    return lin, Q, const + P * B


# ----------------------------------------------------------------------------
# QUBO -> Ising
# ----------------------------------------------------------------------------
def ising_terms(qb: QuboBatch) -> tuple[list[tuple[tuple[int, ...], float]], str]:
    """Devuelve [(qubits, coeficiente)] de Z_i y Z_iZ_j.
    Primero intenta QuadraticProgram.to_ising() de qiskit; si no, x = (1 − z)/2 a mano."""
    if qb.qubo_qp is not None:
        try:
            op, _offset = qb.qubo_qp.to_ising()
            terms = []
            for pauli, coeff in zip(op.paulis, op.coeffs):
                if pauli.x.any():
                    raise RuntimeError("Término no diagonal en el Hamiltoniano de coste")
                qs = tuple(int(q) for q in np.flatnonzero(pauli.z))
                if qs and abs(coeff) > 1e-12:
                    terms.append((qs, float(np.real(coeff))))
            return terms, "QuadraticProgram.to_ising()"
        except Exception as exc:  # noqa: BLE001
            log.warning("to_ising falló (%s); usando conversión manual", exc)
    n = qb.n_qubits
    h = -qb.linear / 2
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            b = qb.quad[i, j]
            if b != 0:
                h[i] -= b / 4
                h[j] -= b / 4
                terms.append(((i, j), b / 4))
    terms = [((i,), float(h[i])) for i in range(n) if abs(h[i]) > 1e-12] + terms
    return terms, "manual x=(1−z)/2"


def qubo_energies(qb: QuboBatch) -> tuple[np.ndarray, np.ndarray]:
    """Energía QUBO de los 2^n estados y bits (convención little-endian de Qiskit:
    el bit i del índice = qubit i = variable i)."""
    n = qb.n_qubits
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.float64)
    E = qb.const + bits @ qb.linear + np.einsum("si,ij,sj->s", bits, qb.quad, bits, optimize=True)
    return E, bits
