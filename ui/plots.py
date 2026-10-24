"""
ui/plots.py
===========
Figuras Plotly del dashboard. Sin lógica de negocio: sólo leen Scenario/Solution.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from core.evaluator import Solution
from core.scenario import Scenario

PRIORITY_SIZE = {1: 11, 2: 15, 3: 20}


def route_map(scn: Scenario, sol: Solution | None = None, title: str = "") -> go.Figure:
    fig = go.Figure()

    # Rutas por dron
    if sol is not None:
        for d in scn.drones:
            tr = sol.traces[d.id]
            if not tr.sequence:
                continue
            xy = tr.trajectory
            fig.add_trace(go.Scatter(
                x=xy[:, 1], y=xy[:, 2], mode="lines",
                line=dict(color=d.color, width=3), opacity=0.85,
                name=f"{d.name} ({len(tr.sequence)} misiones)",
                hovertemplate=f"{d.name}<br>Energía {tr.energy:.1f}/{d.battery:.0f} Wh<extra></extra>",
            ))

    # Misiones (color = dron asignado, tamaño = prioridad)
    owner = {}
    if sol is not None:
        for k, seq in sol.routes.items():
            for pos, m in enumerate(seq):
                owner[m] = (k, pos + 1)
    for m in scn.missions:
        k, pos = owner.get(m.id, (None, None))
        color = scn.drones[k].color if k is not None else "#8a8a8a"
        fig.add_trace(go.Scatter(
            x=[m.x], y=[m.y], mode="markers+text",
            marker=dict(size=PRIORITY_SIZE[m.priority], color=color,
                        line=dict(width=1.5, color="white"),
                        symbol="circle" if k is not None else "circle-open"),
            text=[f"m{m.id}" + (f"·{pos}" if pos else "")], textposition="top center",
            showlegend=False,
            hovertemplate=(f"Misión {m.id} · prioridad {m.priority}<br>"
                           f"Ventana {m.tw_start:.0f}–{m.tw_end:.0f} min<br>"
                           f"Servicio {m.service_time:.1f} min<extra></extra>"),
        ))

    # Bases
    fig.add_trace(go.Scatter(
        x=[d.depot[0] for d in scn.drones], y=[d.depot[1] for d in scn.drones],
        mode="markers+text", text=[d.name for d in scn.drones], textposition="bottom center",
        marker=dict(symbol="square", size=16, color=[d.color for d in scn.drones],
                    line=dict(width=2, color="black")),
        name="Bases", hoverinfo="text",
    ))

    # Conflictos espaciotemporales
    if sol is not None and sol.conflicts:
        fig.add_trace(go.Scatter(
            x=[c["x"] for c in sol.conflicts], y=[c["y"] for c in sol.conflicts],
            mode="markers", marker=dict(symbol="x", size=18, color="#d62728", line=dict(width=2)),
            name=f"Conflictos ({len(sol.conflicts)})",
            text=[f"t={c['t']:.1f} min · D{c['drones'][0] + 1}↔D{c['drones'][1] + 1}" for c in sol.conflicts],
            hovertemplate="%{text}<extra></extra>",
        ))

    pad = 0.05 * scn.area
    fig.update_layout(
        title=title, height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.08),
        xaxis=dict(range=[-pad, scn.area + pad], title="km", constrain="domain"),
        yaxis=dict(range=[-pad, scn.area + pad], title="km", scaleanchor="x", scaleratio=1),
    )
    return fig


def timeline(scn: Scenario, sol: Solution) -> go.Figure:
    """Gantt de servicio por dron: muestra el solapamiento temporal."""
    fig = go.Figure()
    for d in scn.drones:
        tr = sol.traces[d.id]
        for mid, start in zip(tr.sequence, tr.starts):
            m = scn.missions[mid]
            fig.add_trace(go.Bar(
                y=[d.name], x=[m.service_time], base=[start], orientation="h",
                marker_color=d.color, showlegend=False, text=f"m{mid}", textposition="inside",
                hovertemplate=f"m{mid}: {start:.1f}–{start + m.service_time:.1f} min<extra></extra>",
            ))
        if tr.sequence:
            fig.add_trace(go.Scatter(x=[tr.end_time], y=[d.name], mode="markers",
                                     marker=dict(symbol="triangle-left", size=10, color=d.color),
                                     showlegend=False, hovertemplate=f"Regreso {tr.end_time:.1f} min<extra></extra>"))
    for c in sol.conflicts:
        fig.add_vline(x=c["t"], line=dict(color="#d62728", dash="dot"))
    fig.update_layout(barmode="overlay", height=60 + 45 * scn.M,
                      margin=dict(l=10, r=10, t=10, b=30), xaxis_title="minutos")
    return fig


def convergence(batches) -> go.Figure:
    fig = go.Figure()
    for b, rep in enumerate(batches):
        h = np.asarray(rep.history)
        fig.add_trace(go.Scatter(y=h, mode="lines", name=f"Lote {b + 1} ({rep.n_qubits} qubits)"))
        fig.add_hline(y=rep.energy_opt, line=dict(dash="dot", width=1),
                      annotation_text=f"óptimo lote {b + 1}", annotation_position="bottom right")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="evaluación de COBYLA", yaxis_title="⟨H_C⟩ (energía QUBO)")
    return fig


def qubo_heatmap(rep) -> go.Figure:
    Q = rep.qubo_matrix
    # Escala simétrica tipo log para que P no "apague" el resto de términos
    Z = np.sign(Q) * np.log1p(np.abs(Q))
    fig = go.Figure(go.Heatmap(
        z=Z, x=rep.labels, y=rep.labels, colorscale="RdBu_r", zmid=0,
        customdata=Q, hovertemplate="%{y} · %{x}<br>Q = %{customdata:.2f}<extra></extra>",
        colorbar=dict(title="sign·log(1+|Q|)"),
    ))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), yaxis_autorange="reversed")
    return fig


def state_distribution(rep) -> go.Figure:
    top = rep.top_states
    fig = go.Figure(go.Bar(
        x=[t["probabilidad"] for t in top][::-1],
        y=[t["estado"] for t in top][::-1],
        orientation="h",
        marker_color=["#29a36a" if t["factible"] else "#d62728" for t in top][::-1],
        customdata=[t["energía"] for t in top][::-1],
        hovertemplate="%{y}<br>P = %{x:.3f}<br>E = %{customdata:.2f}<extra></extra>",
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="probabilidad (verde = factible, rojo = viola one-hot)")
    return fig
