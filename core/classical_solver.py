"""
core/classical_solver.py
========================
Baseline clásico: VRPTW multi-depósito con Google OR-Tools.

Modelo:
  * Un vehículo por dron, cada uno sale y vuelve a su propia base.
  * Dimensión "Time": tránsito por vehículo (velocidad propia) + servicio.
    Ventanas de tiempo: inicio duro, fin BLANDO (retraso penalizado x prioridad),
    para que el solver siempre encuentre algo factible.
  * Dimensión "Energy": consumo por vehículo, capacidad = batería (DURA).
  * Disjunciones: una misión puede descartarse pagando una penalización alta
    proporcional a su prioridad (si no cabe en ninguna batería).

Cómo trata las penalizaciones frente al QUBO:
  OR-Tools maneja restricciones de CAPACIDAD y TIEMPO de forma nativa y exacta
  (propagación de restricciones + búsqueda local). Lo que NO puede expresar de
  forma natural son los costes de INTERACCIÓN ENTRE RUTAS de drones distintos
  (conflictos de espacio aéreo): el coste de un arco depende sólo del vehículo
  que lo recorre, no de lo que hacen los demás. Por eso aquí los conflictos se
  miden a posteriori en el evaluador. En el QUBO, en cambio, un término
  cuadrático x_ik·x_jl captura directamente esa interacción.
"""
from __future__ import annotations

import time

from .evaluator import Solution, evaluate, simulate_route
from .scenario import Scenario
from .sequencer import best_sequence

SCALE = 100  # OR-Tools trabaja con enteros: centésimas de minuto / Wh


def solve_classical(scn: Scenario, time_limit_s: float = 2.0) -> Solution:
    t0 = time.perf_counter()
    status = "ok"
    try:
        routes = _solve_ortools(scn, time_limit_s)
        method = "OR-Tools VRPTW (Guided Local Search)"
        if routes is None:
            status = "OR-Tools no encontró solución; se usó heurística greedy"
            routes, method = _greedy(scn), "Greedy (fallback)"
    except ImportError:
        status = "ortools no está instalado; se usó heurística greedy"
        routes, method = _greedy(scn), "Greedy (fallback)"
    elapsed = time.perf_counter() - t0
    return evaluate(scn, routes, "Clásico", elapsed, {"method": method, "status": status})


def _solve_ortools(scn: Scenario, time_limit_s: float) -> dict[int, list[int]] | None:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    M, N = scn.M, scn.N
    D = scn.dist_matrix()
    depots = list(range(M))
    manager = pywrapcp.RoutingIndexManager(M + N, M, depots, depots)
    routing = pywrapcp.RoutingModel(manager)

    def service(node: int) -> float:
        return scn.missions[node - M].service_time if node >= M else 0.0

    # IMPORTANTE: guardamos los callbacks en una lista para que el recolector
    # de basura de Python no los libere mientras el solver C++ los usa.
    keep_alive = []
    time_cb_idx, energy_cb_idx = [], []
    for d in scn.drones:
        def t_cb(fi, ti, d=d):
            i, j = manager.IndexToNode(fi), manager.IndexToNode(ti)
            return int(round((D[i, j] / d.speed + service(i)) * SCALE))

        def e_cb(fi, ti, d=d):
            i, j = manager.IndexToNode(fi), manager.IndexToNode(ti)
            return int(round((D[i, j] * scn.energy_per_km(d) + service(i) * scn.hover_rate(d)) * SCALE))

        keep_alive += [t_cb, e_cb]
        time_cb_idx.append(routing.RegisterTransitCallback(t_cb))
        energy_cb_idx.append(routing.RegisterTransitCallback(e_cb))

    for k in range(M):
        routing.SetArcCostEvaluatorOfVehicle(time_cb_idx[k], k)

    horizon = int(scn.horizon * 2 * SCALE)
    routing.AddDimensionWithVehicleTransits(time_cb_idx, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetGlobalSpanCostCoefficient(20)  # equilibra el makespan

    routing.AddDimensionWithVehicleTransitAndCapacity(
        energy_cb_idx, 0, [int(d.battery * SCALE) for d in scn.drones], True, "Energy"
    )

    for m in scn.missions:
        idx = manager.NodeToIndex(M + m.id)
        time_dim.CumulVar(idx).SetRange(int(m.tw_start * SCALE), horizon)
        time_dim.SetCumulVarSoftUpperBound(idx, int(m.tw_end * SCALE), 10 * m.priority)
        routing.AddDisjunction([idx], 1_000_000 * m.priority)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(max(0.5, time_limit_s) * 1000))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    routes: dict[int, list[int]] = {}
    for k in range(M):
        idx = routing.Start(k)
        seq = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node >= M:
                seq.append(node - M)
            idx = sol.Value(routing.NextVar(idx))
        routes[k] = seq
    return routes


def _greedy(scn: Scenario) -> dict[int, list[int]]:
    """Asignación greedy por prioridad -> inserción en el dron más barato con batería."""
    routes = {d.id: [] for d in scn.drones}
    order = sorted(scn.missions, key=lambda m: (-m.priority, m.tw_end))
    for m in order:
        best = None
        for d in scn.drones:
            cand = best_sequence(scn, d, routes[d.id] + [m.id])
            tr = simulate_route(scn, d, cand)
            if tr.energy <= d.battery:
                score = tr.end_time + 5 * sum(tr.lateness)
                if best is None or score < best[0]:
                    best = (score, d.id, cand)
        if best is not None:
            routes[best[1]] = best[2]
    return routes
