r"""
core/quantum_solver.py
======================
Motor cuántico híbrido: QAOA sobre qiskit-aer + COBYLA, con descomposición en lotes.

Pipeline
--------
  1. make_batches         -> lotes de misiones (≤ max_qubits qubits por QUBO)
  2. build_batch_qubo     -> QuadraticProgram con restricciones -> QUBO (qiskit-optimization)
  3. ising_terms          -> H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j   (to_ising)
  4. QAOA de p capas:
         |ψ(γ,β)⟩ = Π_p  e^{-iβ_p Σ X} · e^{-iγ_p H_C}  |+⟩^{⊗n}
     e^{-iγ h Z} = RZ(2γh),  e^{-iγ J ZZ} = RZZ(2γJ),  e^{-iβX} = RX(2β)
  5. COBYLA minimiza ⟨ψ|H_C|ψ⟩ (bucle clásico-cuántico)
  6. Muestreo (shots) -> mejor bitstring FACTIBLE (one-hot); si ninguno lo es,
     decodificación por marginales (post-procesado clásico)
  7. Secuenciación clásica por dron + reparación de batería

Cómo maneja QAOA la matriz de penalizaciones
--------------------------------------------
  Las restricciones no existen como tales para el circuito: están "horneadas"
  en las energías. Cada estado de la base computacional es una asignación
  completa; la penalización P eleva la energía de los estados no factibles,
  y el operador de coste e^{-iγH_C} les aplica una fase distinta. La
  interferencia que produce el mezclador X amplifica estados de baja energía,
  así que la probabilidad migra hacia asignaciones factibles y baratas. Por
  eso medimos P(factible) y P(óptimo): indican lo bien que QAOA "aprendió" la
  matriz de penalizaciones. Con el enfoque clásico, en cambio, las
  restricciones podan el espacio de búsqueda; nunca se visitan estados
  inválidos.

  Ratio de aproximación (sobre el subespacio factible del lote):
      r = (E_max − E_QAOA) / (E_max − E_min)  ∈ [0, 1],  r = 1 ⇔ óptimo exacto
  E_min/E_max se obtienen por enumeración exacta (posible porque n ≤ ~16).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .evaluator import Solution, evaluate
from .qubo_builder import (QuboBatch, build_batch_qubo, conflict_graph, energy_estimate,
                           ising_terms, make_batches, qubo_energies)
from .scenario import Scenario
from .sequencer import best_sequence, repair_battery

log = logging.getLogger(__name__)


@dataclass
class BatchReport:
    mission_ids: list[int]
    n_qubits: int
    assignment: dict[int, int]
    energy_found: float
    energy_opt: float
    energy_max: float
    approx_ratio: float
    approx_ratio_expected: float
    p_feasible: float
    p_optimal: float
    found_optimal: bool
    repaired: bool
    history: list[float]
    top_states: list[dict[str, Any]]
    qubo_matrix: np.ndarray
    labels: list[str]
    penalty: float
    circuit_depth: int | None
    n_rzz: int
    backend: str
    ising_source: str
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Simuladores de estado: Aer (principal) y numpy (respaldo)
# ----------------------------------------------------------------------------
class _AerQAOA:
    """Construye el circuito QAOA a partir de los términos de Ising y lo simula en Aer."""

    def __init__(self, n: int, terms, reps: int, seed: int):
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterVector
        from qiskit_aer import AerSimulator  # importarlo registra save_statevector

        self.reps = reps
        self.gamma = ParameterVector("γ", reps)
        self.beta = ParameterVector("β", reps)
        qc = QuantumCircuit(n, name="QAOA")
        qc.h(range(n))
        for p in range(reps):
            for qs, c in terms:                     # e^{-iγ_p H_C}
                if len(qs) == 1:
                    qc.rz(2 * c * self.gamma[p], qs[0])
                elif len(qs) == 2:
                    qc.rzz(2 * c * self.gamma[p], qs[0], qs[1])
            for q in range(n):                      # e^{-iβ_p Σ X}
                qc.rx(2 * self.beta[p], q)
        qc.save_statevector()
        self.sim = AerSimulator(method="statevector", seed_simulator=seed)
        self.circuit = transpile(qc, self.sim, optimization_level=1)
        self.depth = self.circuit.depth()

    def probs(self, theta: np.ndarray) -> np.ndarray:
        binds = {**dict(zip(self.gamma, theta[:self.reps])), **dict(zip(self.beta, theta[self.reps:]))}
        bound = self.circuit.assign_parameters(binds)
        sv = np.asarray(self.sim.run(bound).result().get_statevector())
        return np.abs(sv) ** 2


class _NumpyQAOA:
    """Simulación exacta del mismo circuito con numpy (para entornos sin qiskit-aer).
    Como H_C es diagonal, e^{-iγH_C} es un vector de fases."""

    def __init__(self, n: int, diag: np.ndarray, reps: int):
        self.n, self.diag, self.reps = n, diag, reps
        self.depth = None

    def probs(self, theta: np.ndarray) -> np.ndarray:
        n = self.n
        psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
        for p in range(self.reps):
            psi *= np.exp(-1j * theta[p] * self.diag)
            c, s = np.cos(theta[self.reps + p]), -1j * np.sin(theta[self.reps + p])
            for q in range(n):
                v = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
                a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
                v[:, 0, :] = c * a0 + s * a1
                v[:, 1, :] = s * a0 + c * a1
        return np.abs(psi) ** 2


def _ising_diag(terms, bits: np.ndarray) -> np.ndarray:
    z = 1 - 2 * bits           # bit 0 -> z=+1, bit 1 -> z=-1
    d = np.zeros(len(bits))
    for qs, c in terms:
        d += c * np.prod(z[:, list(qs)], axis=1)
    return d


# ----------------------------------------------------------------------------
# QAOA sobre un lote
# ----------------------------------------------------------------------------
def run_qaoa_batch(qb: QuboBatch, reps: int, maxiter: int, shots: int,
                   restarts: int, rng: np.random.Generator, use_aer: bool = True) -> BatchReport:
    n, M, B = qb.n_qubits, qb.n_drones, len(qb.mission_ids)
    warnings = list(qb.warnings)

    energies, bits = qubo_energies(qb)
    feasible = (bits.reshape(-1, B, M).sum(-1) == 1).all(-1)
    E_feas = energies[feasible]
    E_min, E_max = float(E_feas.min()), float(E_feas.max())

    terms, ising_src = ising_terms(qb)
    scale = max(abs(c) for _, c in terms) if terms else 1.0
    terms_n = [(qs, c / scale) for qs, c in terms]   # normalizar ayuda a COBYLA
    diag_n = _ising_diag(terms_n, bits)              # = (E − offset)/scale

    # Sanity check: el Hamiltoniano de Ising debe reproducir el QUBO (salvo constante)
    delta = diag_n * scale - energies
    if np.ptp(delta) > 1e-6 * max(1.0, np.ptp(energies)):
        warnings.append("Ising y QUBO difieren: se usa la diagonal QUBO directamente.")
        diag_n = (energies - energies.mean()) / max(np.ptp(energies), 1e-9)
        terms_n = None

    engine, backend = None, "numpy statevector"
    if use_aer and terms_n is not None:
        try:
            engine = _AerQAOA(n, terms_n, reps, int(rng.integers(1 << 30)))
            # Verificación cruzada con el simulador numpy en un punto aleatorio
            th = rng.uniform(0, 1, 2 * reps)
            if not np.allclose(engine.probs(th), _NumpyQAOA(n, diag_n, reps).probs(th), atol=1e-6):
                raise RuntimeError("Aer y la referencia numpy no coinciden")
            backend = "qiskit-aer statevector"
        except ImportError:
            warnings.append("qiskit/qiskit-aer no disponible: simulación con numpy.")
            engine = None
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Aer falló ({exc}); simulación con numpy.")
            engine = None
    if engine is None:
        engine = _NumpyQAOA(n, diag_n, reps)

    history: list[float] = []

    def objective(theta):
        p = engine.probs(theta)
        history.append(float(p @ energies))
        return float(p @ diag_n)

    best = None
    for r in range(max(1, restarts)):
        if r == 0:   # schedule tipo "annealing lineal": γ crece, β decrece
            frac = (np.arange(reps) + 0.5) / reps
            x0 = np.concatenate([0.6 * frac, 0.6 * (1 - frac)])
        else:
            x0 = np.concatenate([rng.uniform(0, 0.8, reps), rng.uniform(0, np.pi / 2, reps)])
        res = minimize(objective, x0, method="COBYLA",
                       options={"maxiter": int(maxiter), "rhobeg": 0.3})
        if best is None or res.fun < best.fun:
            best = res

    probs = engine.probs(best.x)
    probs = probs / probs.sum()

    # --- Muestreo con shots finitos (como en hardware real) -----------------
    counts = rng.multinomial(int(shots), probs)
    sampled = np.flatnonzero(counts)
    sampled_feas = sampled[feasible[sampled]]
    repaired = False
    if len(sampled_feas):
        chosen = int(sampled_feas[np.argmin(energies[sampled_feas])])
        x = bits[chosen].astype(int)
    else:
        # Decodificación por marginales: P(x_ik = 1) y argmax por misión
        repaired = True
        marg = probs @ bits
        x = np.zeros(n, dtype=int)
        for b in range(B):
            x[b * M + int(np.argmax(marg[b * M:(b + 1) * M]))] = 1
        chosen = int((x << np.arange(n)).sum())
        warnings.append("Ninguna muestra factible: asignación reparada por marginales.")

    E_found = float(energies[chosen])
    span = E_max - E_min
    ratio = 1.0 if span < 1e-12 else float(np.clip((E_max - E_found) / span, 0, 1))
    p_feas = float(probs[feasible].sum())
    E_cond = float(probs[feasible] @ E_feas / p_feas) if p_feas > 1e-12 else E_max
    ratio_exp = 1.0 if span < 1e-12 else float(np.clip((E_max - E_cond) / span, 0, 1))
    opt_mask = feasible & np.isclose(energies, E_min, atol=1e-9 * max(1, abs(E_min)))
    p_opt = float(probs[opt_mask].sum())

    assignment = {qb.mission_ids[b]: int(np.argmax(x[b * M:(b + 1) * M])) for b in range(B)}

    top = []
    for s in np.argsort(probs)[::-1][:10]:
        xs = bits[s].astype(int)
        desc = " ".join(qb.labels[j] for j in np.flatnonzero(xs)) or "∅"
        top.append({"estado": desc, "bitstring": "".join(map(str, xs)),
                    "probabilidad": float(probs[s]), "energía": float(energies[s]),
                    "factible": bool(feasible[s])})

    return BatchReport(
        mission_ids=qb.mission_ids, n_qubits=n, assignment=assignment,
        energy_found=E_found, energy_opt=E_min, energy_max=E_max,
        approx_ratio=ratio, approx_ratio_expected=ratio_exp,
        p_feasible=p_feas, p_optimal=p_opt,
        found_optimal=bool(opt_mask[chosen]), repaired=repaired,
        history=history, top_states=top, qubo_matrix=qb.symmetric_matrix(),
        labels=qb.labels, penalty=qb.penalty, circuit_depth=engine.depth,
        n_rzz=sum(1 for qs, _ in terms if len(qs) == 2) * reps,
        backend=backend, ising_source=ising_src, warnings=warnings,
    )


# ----------------------------------------------------------------------------
# Solver completo
# ----------------------------------------------------------------------------
def solve_quantum(scn: Scenario, reps: int = 2, max_qubits: int = 12, maxiter: int = 120,
                  shots: int = 2048, restarts: int = 2, seed: int = 7) -> Solution:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    max_qubits = int(np.clip(max_qubits, scn.M, 16))   # 2^16 amplitudes: límite seguro
    graph = conflict_graph(scn)
    batches = make_batches(scn, max_qubits)

    fixed: dict[int, int] = {}
    loads = np.zeros(scn.M)
    reports: list[BatchReport] = []
    q_time = 0.0
    for batch in batches:
        qb = build_batch_qubo(scn, batch, fixed, loads, graph)
        tq = time.perf_counter()
        rep = run_qaoa_batch(qb, reps, maxiter, shots, restarts, rng)
        q_time += time.perf_counter() - tq
        reports.append(rep)
        for i, k in rep.assignment.items():
            fixed[i] = k
            loads[k] += energy_estimate(scn, i, k)

    routes = {d.id: best_sequence(scn, d, [i for i, k in fixed.items() if k == d.id])
              for d in scn.drones}
    routes, moves = repair_battery(scn, routes)

    elapsed = time.perf_counter() - t0
    weights = np.array([r.n_qubits for r in reports], dtype=float)
    extra = {
        "method": f"QAOA p={reps} · {len(batches)} sub-QUBO(s) · COBYLA",
        "batches": reports,
        "approx_ratio": float(np.average([r.approx_ratio for r in reports], weights=weights)),
        "approx_ratio_expected": float(np.average([r.approx_ratio_expected for r in reports], weights=weights)),
        "p_feasible": float(np.average([r.p_feasible for r in reports], weights=weights)),
        "optimal_batches": sum(r.found_optimal for r in reports),
        "max_qubits_used": int(max(r.n_qubits for r in reports)),
        "total_qubit_vars": int(scn.N * scn.M),
        "battery_repairs": moves,
        "qaoa_time_s": q_time,
        "backend": reports[0].backend if reports else "-",
        "conflict_edges": graph.number_of_edges(),
    }
    return evaluate(scn, routes, "Cuántico híbrido", elapsed, extra)
