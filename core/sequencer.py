"""
core/sequencer.py
=================
Ordena las misiones que el motor cuántico asignó a cada dron.

Diseño híbrido (descomposición tipo "cluster-first, route-second"):
  * La parte combinatoriamente difícil y con interacciones de pares
    (¿qué dron hace qué misión, sin saturar baterías ni cruzarse?) va al QUBO.
  * El orden dentro de cada dron es un problema pequeño (≤ 7 misiones casi
    siempre) que se resuelve exacto por fuerza bruta, o con 2-opt si es mayor.
    Codificar también el orden en el QUBO exigiría N²·M qubits (≈300 para
    10 misiones y 3 drones), inviable hoy en simuladores o hardware NISQ.
"""
from __future__ import annotations

from itertools import permutations

from .evaluator import route_cost, simulate_route
from .scenario import Drone, Scenario

EXACT_LIMIT = 7  # 7! = 5040 permutaciones: rápido en Python


def best_sequence(scn: Scenario, drone: Drone, missions: list[int]) -> list[int]:
    missions = list(missions)
    if len(missions) <= 1:
        return missions
    if len(missions) <= EXACT_LIMIT:
        return list(min(permutations(missions), key=lambda p: route_cost(scn, drone, list(p))))
    return _two_opt(scn, drone, _nearest_neighbor(scn, drone, missions))


def _nearest_neighbor(scn: Scenario, drone: Drone, missions: list[int]) -> list[int]:
    D = scn.dist_matrix()
    left = set(missions)
    node, seq = drone.id, []
    while left:
        # Vecino más cercano sesgado por urgencia (fin de ventana temprano)
        nxt = min(left, key=lambda m: D[node, scn.mission_node(m)] / drone.speed
                  + 0.3 * scn.missions[m].tw_end)
        seq.append(nxt)
        left.remove(nxt)
        node = scn.mission_node(nxt)
    return seq


def _two_opt(scn: Scenario, drone: Drone, seq: list[int]) -> list[int]:
    best, best_c = seq, route_cost(scn, drone, seq)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = route_cost(scn, drone, cand)
                if c < best_c - 1e-9:
                    best, best_c, improved = cand, c, True
    return best


def repair_battery(scn: Scenario, routes: dict[int, list[int]], max_moves: int = 20) -> tuple[dict[int, list[int]], int]:
    """Post-procesado clásico: si un dron excede su batería, mueve su misión de
    menor prioridad al dron con más margen donde quepa. Devuelve (rutas, nº movimientos)."""
    routes = {k: list(v) for k, v in routes.items()}
    moves = 0
    for _ in range(max_moves):
        energy = {d.id: simulate_route(scn, d, routes[d.id]).energy for d in scn.drones}
        over = [d for d in scn.drones if energy[d.id] > d.battery and routes[d.id]]
        if not over:
            break
        src = max(over, key=lambda d: energy[d.id] - d.battery)
        moved = False
        for mid in sorted(routes[src.id], key=lambda m: scn.missions[m].priority):
            for dst in sorted(scn.drones, key=lambda d: energy[d.id] - d.battery):
                if dst.id == src.id:
                    continue
                cand = best_sequence(scn, dst, routes[dst.id] + [mid])
                if simulate_route(scn, dst, cand).energy <= dst.battery:
                    routes[src.id] = best_sequence(scn, src, [m for m in routes[src.id] if m != mid])
                    routes[dst.id] = cand
                    moves += 1
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return routes, moves
