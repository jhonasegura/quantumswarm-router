"""
core/evaluator.py
=================
Evaluador ÚNICO para ambos enfoques. Cualquier solución (clásica o cuántica)
se expresa como `routes: dict[drone_id -> lista ordenada de mission_id]` y se
simula aquí con la misma física, así la comparación es justa.

Qué se mide:
  * Tiempos de llegada/servicio con ventanas de tiempo (espera si llega pronto,
    retraso si llega tarde).
  * Energía consumida (crucero + estacionario) vs. batería.
  * Conflictos espaciotemporales: se muestrean las trayectorias en el tiempo y
    se detecta cuándo dos drones están a menos del radio de seguridad.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scenario import Drone, Scenario

LATE_WEIGHT = 2.0          # coste por minuto de retraso (x prioridad)
CONFLICT_WEIGHT = 15.0     # coste base de un conflicto (x conflict_penalty/5)
BATTERY_WEIGHT = 200.0     # coste por dron que agota la batería
UNSERVED_WEIGHT = 150.0    # coste por misión no atendida (x prioridad)


@dataclass
class RouteTrace:
    drone_id: int
    sequence: list[int]
    arrivals: list[float]
    starts: list[float]
    lateness: list[float]
    distance: float
    energy: float
    end_time: float
    battery_ok: bool
    # Trayectoria piecewise-lineal: (t, x, y)
    trajectory: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))


@dataclass
class Solution:
    solver: str
    routes: dict[int, list[int]]
    traces: dict[int, RouteTrace]
    conflicts: list[dict[str, Any]]
    kpis: dict[str, float]
    compute_time: float
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def simulate_route(scn: Scenario, drone: Drone, seq: list[int]) -> RouteTrace:
    """Simula una ruta base -> misiones -> base."""
    D = scn.dist_matrix()
    node = drone.id
    t = 0.0
    dist = 0.0
    energy = 0.0
    arrivals, starts, lateness = [], [], []
    traj = [(0.0, *drone.depot)]
    e_km = scn.energy_per_km(drone)
    e_hover = scn.hover_rate(drone)

    for mid in seq:
        m = scn.missions[mid]
        nxt = scn.mission_node(mid)
        leg = D[node, nxt]
        t += leg / drone.speed
        dist += leg
        energy += leg * e_km
        arrivals.append(t)
        traj.append((t, m.x, m.y))
        start = max(t, m.tw_start)          # espera posada si llega antes
        starts.append(start)
        lateness.append(max(0.0, start - m.tw_end))
        if start > t:
            traj.append((start, m.x, m.y))
        t = start + m.service_time
        energy += m.service_time * e_hover
        traj.append((t, m.x, m.y))
        node = nxt

    leg = D[node, drone.id]
    t += leg / drone.speed
    dist += leg
    energy += leg * e_km
    traj.append((t, *drone.depot))

    return RouteTrace(
        drone_id=drone.id,
        sequence=list(seq),
        arrivals=arrivals,
        starts=starts,
        lateness=lateness,
        distance=dist,
        energy=energy,
        end_time=t,
        battery_ok=energy <= drone.battery + 1e-9,
        trajectory=np.array(traj, dtype=float),
    )


def route_cost(scn: Scenario, drone: Drone, seq: list[int]) -> float:
    """Coste escalar rápido de UNA ruta (lo usa el secuenciador)."""
    tr = simulate_route(scn, drone, seq)
    late = sum(scn.missions[m].priority * l for m, l in zip(seq, tr.lateness))
    over = max(0.0, tr.energy - drone.battery)
    return tr.end_time + 10.0 * late + 100.0 * over


def _position(traj: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpola posición en los instantes t. Fuera de la ruta -> inactivo."""
    ts = traj[:, 0].copy()
    # np.interp necesita tiempos no decrecientes; añadimos un epsilon a duplicados
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1e-6
    x = np.interp(t, ts, traj[:, 1])
    y = np.interp(t, ts, traj[:, 2])
    active = (t > 0) & (t < ts[-1])
    return x, y, active


def detect_conflicts(scn: Scenario, traces: dict[int, RouteTrace], dt: float = 0.25) -> list[dict]:
    """Detecta episodios de pérdida de separación entre pares de drones."""
    flying = [tr for tr in traces.values() if tr.sequence]
    if len(flying) < 2:
        return []
    t_end = max(tr.end_time for tr in flying)
    t = np.arange(0.0, t_end + dt, dt)
    pos = {tr.drone_id: _position(tr.trajectory, t) for tr in flying}
    conflicts = []
    ids = sorted(pos)
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            xa, ya, act_a = pos[a]
            xb, yb, act_b = pos[b]
            close = (np.hypot(xa - xb, ya - yb) < scn.safety_radius) & act_a & act_b
            if not close.any():
                continue
            # Cada flanco de subida = un episodio de conflicto
            edges = np.flatnonzero(np.diff(np.concatenate([[0], close.astype(int)])) == 1)
            for e in edges:
                conflicts.append({
                    "t": float(t[e]),
                    "x": float((xa[e] + xb[e]) / 2),
                    "y": float((ya[e] + yb[e]) / 2),
                    "drones": (a, b),
                })
    return conflicts


def evaluate(
    scn: Scenario,
    routes: dict[int, list[int]],
    solver: str,
    compute_time: float,
    extra: dict | None = None,
) -> Solution:
    traces = {d.id: simulate_route(scn, d, routes.get(d.id, [])) for d in scn.drones}
    conflicts = detect_conflicts(scn, traces)

    served = {m for seq in routes.values() for m in seq}
    unserved = [m.id for m in scn.missions if m.id not in served]
    late_count = 0
    weighted_late = 0.0
    for tr in traces.values():
        for mid, l in zip(tr.sequence, tr.lateness):
            if l > 1e-6:
                late_count += 1
                weighted_late += scn.missions[mid].priority * l

    total_energy = sum(tr.energy for tr in traces.values())
    total_capacity = sum(d.battery for d in scn.drones)
    battery_viol = sum(not tr.battery_ok for tr in traces.values())
    served_priority = sum(scn.missions[m].priority for m in served)
    max_batt_use = max(tr.energy / scn.drones[tr.drone_id].battery for tr in traces.values())

    cost = (
        sum(tr.end_time for tr in traces.values() if tr.sequence)
        + LATE_WEIGHT * weighted_late
        + CONFLICT_WEIGHT * (scn.conflict_penalty / 5.0) * len(conflicts)
        + BATTERY_WEIGHT * battery_viol
        + UNSERVED_WEIGHT * sum(scn.missions[m].priority for m in unserved)
    )

    kpis_raw = {
        "costo_total": cost,
        "makespan_min": max((tr.end_time for tr in traces.values() if tr.sequence), default=0.0),
        "tiempo_vuelo_total_min": sum(tr.end_time for tr in traces.values() if tr.sequence),
        "distancia_total_km": sum(tr.distance for tr in traces.values()),
        "energia_total_wh": total_energy,
        "uso_bateria_flota_pct": 100 * total_energy / total_capacity,
        "uso_bateria_max_pct": 100 * max_batt_use,
        # Eficiencia energética: puntos de prioridad atendidos por cada 10 Wh
        "eficiencia_pts_por_10wh": 10 * served_priority / total_energy if total_energy > 0 else 0.0,
        "misiones_atendidas": len(served),
        "misiones_sin_atender": len(unserved),
        "entregas_tarde": late_count,
        "retraso_ponderado": weighted_late,
        "violaciones_bateria": battery_viol,
        "conflictos_espacio_aereo": len(conflicts),
        "tiempo_computo_s": compute_time,
    }
    kpis = {k: float(v) for k, v in kpis_raw.items()}
    return Solution(
        solver=solver,
        routes={k: list(v) for k, v in routes.items()},
        traces=traces,
        conflicts=conflicts,
        kpis=kpis,
        compute_time=compute_time,
        extra=extra or {},
    )
