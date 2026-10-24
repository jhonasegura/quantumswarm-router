"""
core/scenario.py
================
Generador de escenarios multi-agente: flota de drones + misiones georreferenciadas.

Unidades (consistentes en todo el proyecto):
    distancia  -> km
    tiempo     -> minutos
    velocidad  -> km/min
    energía    -> Wh

Convención de nodos (compartida por los dos solvers y el evaluador):
    nodos 0 .. M-1       -> bases (depots) de cada dron
    nodos M .. M+N-1     -> misiones
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Paleta para distinguir drones en los mapas (accesible en tema claro y oscuro)
DRONE_COLORS = ["#1f77b4", "#e4572e", "#29a36a", "#9467bd", "#f2a541"]

MIN_DRONES, MAX_DRONES = 2, 5
MIN_MISSIONS, MAX_MISSIONS = 4, 12


@dataclass
class Drone:
    id: int
    name: str
    depot: tuple[float, float]
    speed: float          # km/min
    battery: float        # capacidad útil en Wh
    consumption: float    # Wh/km en crucero (antes del factor de desgaste)
    hover: float          # Wh/min en vuelo estacionario durante el servicio
    color: str


@dataclass
class Mission:
    id: int
    x: float
    y: float
    priority: int         # 1 (baja) .. 3 (crítica)
    service_time: float   # minutos de inspección/entrega sobre el punto
    tw_start: float       # ventana de tiempo [tw_start, tw_end] en minutos
    tw_end: float


@dataclass
class Scenario:
    drones: list[Drone]
    missions: list[Mission]
    wear_factor: float = 1.0        # multiplica todo el consumo energético
    conflict_penalty: float = 5.0   # peso de los conflictos de espacio aéreo
    safety_radius: float = 0.5      # km: separación mínima entre drones
    horizon: float = 90.0           # minutos
    area: float = 10.0              # lado del área de operación en km
    seed: int = 0
    _dist: np.ndarray | None = field(default=None, repr=False)

    @property
    def M(self) -> int:
        return len(self.drones)

    @property
    def N(self) -> int:
        return len(self.missions)

    def node_xy(self) -> np.ndarray:
        """Coordenadas de todos los nodos: primero bases, luego misiones."""
        depots = [d.depot for d in self.drones]
        missions = [(m.x, m.y) for m in self.missions]
        return np.array(depots + missions, dtype=float)

    def dist_matrix(self) -> np.ndarray:
        """Matriz euclídea (M+N)x(M+N), cacheada."""
        if self._dist is None:
            xy = self.node_xy()
            diff = xy[:, None, :] - xy[None, :, :]
            self._dist = np.sqrt((diff ** 2).sum(-1))
        return self._dist

    def mission_node(self, mission_id: int) -> int:
        return self.M + mission_id

    def energy_per_km(self, drone: Drone) -> float:
        return drone.consumption * self.wear_factor

    def hover_rate(self, drone: Drone) -> float:
        return drone.hover * self.wear_factor


def generate_scenario(
    n_drones: int = 3,
    n_missions: int = 8,
    seed: int = 42,
    wear_factor: float = 1.0,
    conflict_penalty: float = 5.0,
    safety_radius: float = 0.5,
    area: float = 10.0,
    horizon: float = 90.0,
) -> Scenario:
    """Crea un escenario reproducible.

    Las bases se distribuyen en el perímetro (como hangares en los bordes de la
    zona) y las misiones en el interior, de modo que las rutas de distintos
    drones tienden a converger en el centro: ahí aparecen los conflictos
    espaciotemporales que el modelo QUBO intenta evitar.
    """
    if not MIN_DRONES <= n_drones <= MAX_DRONES:
        raise ValueError(f"n_drones debe estar entre {MIN_DRONES} y {MAX_DRONES}")
    if not MIN_MISSIONS <= n_missions <= MAX_MISSIONS:
        raise ValueError(f"n_missions debe estar entre {MIN_MISSIONS} y {MAX_MISSIONS}")
    if wear_factor <= 0 or safety_radius <= 0 or area <= 0 or horizon <= 0:
        raise ValueError("Los parámetros físicos deben ser positivos")

    rng = np.random.default_rng(seed)
    center = area / 2

    drones: list[Drone] = []
    phase = rng.uniform(0, 2 * np.pi)
    for k in range(n_drones):
        ang = phase + 2 * np.pi * k / n_drones
        depot = (center + 0.46 * area * np.cos(ang), center + 0.46 * area * np.sin(ang))
        drones.append(
            Drone(
                id=k,
                name=f"Dron-{k + 1}",
                depot=(float(depot[0]), float(depot[1])),
                speed=float(rng.uniform(0.6, 0.9)),
                battery=float(rng.uniform(55, 75)),
                consumption=float(rng.uniform(2.0, 2.6)),
                hover=float(rng.uniform(0.4, 0.7)),
                color=DRONE_COLORS[k % len(DRONE_COLORS)],
            )
        )

    missions: list[Mission] = []
    for i in range(n_missions):
        x, y = rng.uniform(0.12 * area, 0.88 * area, size=2)
        tw_start = float(rng.uniform(0, horizon * 0.35))
        width = float(rng.uniform(15, 35))
        missions.append(
            Mission(
                id=i,
                x=float(x),
                y=float(y),
                priority=int(rng.choice([1, 2, 3], p=[0.4, 0.35, 0.25])),
                service_time=float(rng.uniform(1.5, 4.0)),
                tw_start=tw_start,
                tw_end=min(horizon, tw_start + width),
            )
        )

    return Scenario(
        drones=drones,
        missions=missions,
        wear_factor=float(wear_factor),
        conflict_penalty=float(conflict_penalty),
        safety_radius=float(safety_radius),
        horizon=float(horizon),
        area=float(area),
        seed=int(seed),
    )
