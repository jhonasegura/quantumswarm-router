"""
QuantumSwarm Router — app.py
============================
Dashboard Streamlit que compara, sobre el mismo escenario, un VRPTW clásico
(OR-Tools) con un pipeline cuántico híbrido (QUBO + QAOA en qiskit-aer).

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from core.classical_solver import solve_classical
from core.evaluator import Solution
from core.quantum_solver import solve_quantum
from core.scenario import MAX_DRONES, MAX_MISSIONS, MIN_DRONES, MIN_MISSIONS, generate_scenario
from ui import plots

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="QuantumSwarm Router", page_icon="🛸", layout="wide")

# KPIs a mostrar: (clave, etiqueta, formato, menor_es_mejor)
KPI_ROWS = [
    ("costo_total", "Coste total (función unificada)", "{:.1f}", True),
    ("makespan_min", "Makespan (min)", "{:.1f}", True),
    ("tiempo_vuelo_total_min", "Tiempo total de misión, suma flota (min)", "{:.1f}", True),
    ("distancia_total_km", "Distancia total (km)", "{:.2f}", True),
    ("energia_total_wh", "Energía total (Wh)", "{:.1f}", True),
    ("eficiencia_pts_por_10wh", "Eficiencia (pts prioridad / 10 Wh)", "{:.2f}", False),
    ("uso_bateria_max_pct", "Uso de batería del dron más cargado (%)", "{:.0f}", True),
    ("misiones_sin_atender", "Misiones sin atender", "{:.0f}", True),
    ("entregas_tarde", "Entregas fuera de ventana", "{:.0f}", True),
    ("violaciones_bateria", "Violaciones de batería", "{:.0f}", True),
    ("conflictos_espacio_aereo", "Conflictos de espacio aéreo", "{:.0f}", True),
    ("tiempo_computo_s", "Tiempo de cómputo (s)", "{:.2f}", True),
]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Escenario")
    n_drones = st.slider("Drones", MIN_DRONES, MAX_DRONES, 3)
    n_missions = st.slider("Misiones", MIN_MISSIONS, MAX_MISSIONS, 8)
    seed = st.number_input("Semilla", 0, 10_000, 42, step=1)

    st.header("Restricciones")
    wear = st.slider("Factor de desgaste de batería", 0.8, 2.0, 1.0, 0.05,
                     help="Multiplica el consumo en crucero y en estacionario.")
    conflict_pen = st.slider("Penalización por conflicto de espacio aéreo", 0.0, 10.0, 5.0, 0.5,
                             help="Peso γ de los términos x_ik·x_jl del QUBO y del coste unificado.")
    safety = st.slider("Radio de seguridad entre drones (km)", 0.2, 1.5, 0.5, 0.1)

    st.header("Solvers")
    ort_time = st.slider("Límite de tiempo OR-Tools (s)", 1, 10, 2)
    reps = st.slider("Capas QAOA (p)", 1, 3, 2)
    max_qubits = st.slider("Qubits máx. por sub-QUBO", max(6, n_drones), 16, max(12, n_drones),
                           help="Cada lote tiene ⌊qubits/drones⌋ misiones. 2^n amplitudes por simulación.")
    maxiter = st.slider("Iteraciones COBYLA", 30, 300, 120, 10)
    shots = st.select_slider("Shots", [512, 1024, 2048, 4096, 8192], 2048)

    run = st.button("Ejecutar enfoque clásico vs. cuántico", type="primary", use_container_width=True)

params = dict(n_drones=n_drones, n_missions=n_missions, seed=int(seed), wear_factor=wear,
              conflict_penalty=conflict_pen, safety_radius=safety)


@st.cache_data(show_spinner=False)
def build_scenario(**kw):
    return generate_scenario(**kw)


try:
    scn = build_scenario(**params)
except ValueError as exc:
    st.error(f"Parámetros de escenario no válidos: {exc}")
    st.stop()


# ----------------------------------------------------------------------------
# Ejecución en paralelo (cada solver aislado: un fallo no tumba al otro)
# ----------------------------------------------------------------------------
def _safe(fn, name, *args, **kw) -> Solution:
    try:
        return fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Fallo en %s", name)
        return Solution(solver=name, routes={}, traces={}, conflicts=[], kpis={},
                        compute_time=0.0, error=f"{exc}\n\n{traceback.format_exc()}")


if run:
    with st.spinner("Resolviendo con OR-Tools y QAOA en paralelo…"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_c = pool.submit(_safe, solve_classical, "Clásico", scn, time_limit_s=ort_time)
            f_q = pool.submit(_safe, solve_quantum, "Cuántico híbrido", scn, reps=reps,
                              max_qubits=max_qubits, maxiter=maxiter, shots=shots, seed=int(seed))
            st.session_state["results"] = (f_c.result(), f_q.result())
            st.session_state["results_params"] = params

# ----------------------------------------------------------------------------
# Cabecera
# ----------------------------------------------------------------------------
st.title("🛸 QuantumSwarm Router")
st.caption(f"{scn.M} drones · {scn.N} misiones · {scn.N * scn.M} variables binarias de asignación "
           f"· área {scn.area:.0f}×{scn.area:.0f} km · horizonte {scn.horizon:.0f} min")

results = st.session_state.get("results")
if results and st.session_state.get("results_params") != params:
    st.info("Has cambiado parámetros del escenario: los resultados mostrados son de la configuración anterior. "
            "Pulsa ejecutar para actualizarlos.")
    results = None

if not results:
    st.plotly_chart(plots.route_map(scn, None, "Escenario: bases y misiones (tamaño = prioridad)"),
                    use_container_width=True)
    st.write("Ajusta el escenario en la barra lateral y pulsa **Ejecutar enfoque clásico vs. cuántico**.")
    st.stop()

sol_c, sol_q = results
for sol in (sol_c, sol_q):
    if sol.error:
        st.error(f"El solver **{sol.solver}** falló. Detalle técnico:")
        st.code(sol.error)
if sol_c.error and sol_q.error:
    st.stop()

tab_routes, tab_qaoa, tab_model, tab_data = st.tabs(
    ["Rutas y KPIs", "Diagnóstico QAOA", "Formulación matemática", "Datos del escenario"])

# ----------------------------------------------------------------------------
# Pestaña 1: mapas + KPIs
# ----------------------------------------------------------------------------
with tab_routes:
    col_c, col_q = st.columns(2)
    for col, sol in ((col_c, sol_c), (col_q, sol_q)):
        with col:
            if sol.error:
                continue
            st.plotly_chart(plots.route_map(scn, sol, f"{sol.solver} · {sol.extra.get('method', '')}"),
                            use_container_width=True)
            st.plotly_chart(plots.timeline(scn, sol), use_container_width=True)
            k = sol.kpis
            if k["violaciones_bateria"]:
                st.warning(f"🔋 {k['violaciones_bateria']:.0f} dron(es) exceden su batería.")
            if k["conflictos_espacio_aereo"]:
                st.warning(f"⚠️ {k['conflictos_espacio_aereo']:.0f} pérdida(s) de separación "
                           f"< {scn.safety_radius} km.")
            if k["misiones_sin_atender"]:
                st.warning(f"📍 {k['misiones_sin_atender']:.0f} misión(es) sin atender.")
            if not (k["violaciones_bateria"] or k["conflictos_espacio_aereo"] or k["misiones_sin_atender"]):
                st.success("Sin violaciones de batería, conflictos ni misiones descartadas.")
            if sol.extra.get("status", "ok") != "ok":
                st.info(sol.extra["status"])

    if not (sol_c.error or sol_q.error):
        st.subheader("Comparativa de KPIs")
        m1, m2, m3, m4, m5 = st.columns(5)
        kc, kq = sol_c.kpis, sol_q.kpis
        m1.metric("Coste total (cuántico)", f"{kq['costo_total']:.1f}",
                  f"{kq['costo_total'] - kc['costo_total']:+.1f} vs clásico", delta_color="inverse")
        m2.metric("Conflictos (cuántico)", f"{kq['conflictos_espacio_aereo']:.0f}",
                  f"{kq['conflictos_espacio_aereo'] - kc['conflictos_espacio_aereo']:+.0f} vs clásico",
                  delta_color="inverse")
        m3.metric("Energía flota (cuántico)", f"{kq['energia_total_wh']:.0f} Wh",
                  f"{kq['energia_total_wh'] - kc['energia_total_wh']:+.1f} Wh", delta_color="inverse")
        m4.metric("Ratio de aproximación QAOA", f"{sol_q.extra['approx_ratio']:.3f}",
                  f"esperado {sol_q.extra['approx_ratio_expected']:.3f}", delta_color="off")
        m5.metric("Cómputo C / Q", f"{kc['tiempo_computo_s']:.2f} / {kq['tiempo_computo_s']:.2f} s")

        rows = []
        for key, label, fmt, lower in KPI_ROWS:
            a, b = kc[key], kq[key]
            better = "=" if abs(a - b) < 1e-9 else (
                "Cuántico" if (b < a) == lower else "Clásico")
            rows.append({"KPI": label, "Clásico": fmt.format(a), "Cuántico híbrido": fmt.format(b),
                         "Mejor": better})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Ambas soluciones se evalúan con el mismo simulador espaciotemporal. Los tiempos de cómputo "
            "del enfoque cuántico corresponden a SIMULAR el circuito en CPU (coste exponencial en qubits); "
            "no son representativos del tiempo en hardware cuántico."
        )

# ----------------------------------------------------------------------------
# Pestaña 2: diagnóstico QAOA
# ----------------------------------------------------------------------------
with tab_qaoa:
    if sol_q.error:
        st.write("El motor cuántico no produjo resultados.")
    else:
        ex = sol_q.extra
        batches = ex["batches"]
        a, b, c, d = st.columns(4)
        a.metric("Sub-QUBOs", len(batches), f"{ex['max_qubits_used']} qubits máx.", delta_color="off")
        b.metric("Lotes con óptimo exacto", f"{ex['optimal_batches']}/{len(batches)}")
        c.metric("P(factible) media", f"{ex['p_feasible']:.2f}")
        d.metric("Reparaciones de batería", ex["battery_repairs"])
        st.caption(f"Backend: {ex['backend']} · aristas en el grafo de conflictos: {ex['conflict_edges']} · "
                   f"tiempo QAOA: {ex['qaoa_time_s']:.2f} s")
        for w in sorted({w for r in batches for w in r.warnings}):
            st.warning(w)

        st.plotly_chart(plots.convergence(batches), use_container_width=True)

        idx = st.selectbox("Lote a inspeccionar", range(len(batches)),
                           format_func=lambda i: f"Lote {i + 1}: misiones {batches[i].mission_ids}")
        rep = batches[idx]
        st.write(
            f"**{rep.n_qubits} qubits** · penalización one-hot P = {rep.penalty:.1f} · "
            f"profundidad del circuito {rep.circuit_depth or 'n/d'} · {rep.n_rzz} puertas RZZ · "
            f"Ising vía {rep.ising_source} · ratio r = {rep.approx_ratio:.3f} · "
            f"P(óptimo) = {rep.p_optimal:.4f} · E encontrada {rep.energy_found:.2f} "
            f"(óptimo {rep.energy_opt:.2f})"
            + (" · ⚠️ asignación reparada por marginales" if rep.repaired else "")
        )
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**Matriz QUBO** — diagonal: costes lineales y −P; bloques de misión: +2P (one-hot); "
                        "entre drones: conflictos γ·w; mismo dron: compacidad y batería.")
            st.plotly_chart(plots.qubo_heatmap(rep), use_container_width=True)
        with q2:
            st.markdown("**Distribución final del estado QAOA** (10 estados más probables)")
            st.plotly_chart(plots.state_distribution(rep), use_container_width=True)
        st.caption(
            "Con sub-QUBOs pequeños, 2048 shots suelen cubrir todo el subespacio factible, por lo que el "
            "ratio de la mejor muestra tiende a 1. Las métricas que reflejan la calidad real del circuito "
            "son el ratio esperado (sobre ⟨H_C⟩ condicionado a estados factibles), P(factible) y P(óptimo)."
        )

# ----------------------------------------------------------------------------
# Pestaña 3: formulación
# ----------------------------------------------------------------------------
with tab_model:
    st.markdown("**Variables:** $x_{ik}\\in\\{0,1\\}$, la misión $i$ la ejecuta el dron $k$.")
    st.latex(r"""
    \min_x\; \underbrace{\sum_{i,k} c_{ik}x_{ik}}_{\text{aproximación + urgencia}}
    + \lambda_c\sum_k\sum_{i<j} t_k(i,j)\,x_{ik}x_{jk}
    + \gamma\sum_{k\neq l}\sum_{i<j} w_{ij}^{kl}\,x_{ik}x_{jl}
    + \lambda_b\sum_k\Big(\tfrac{L_k+\sum_i e_{ik}x_{ik}}{C_k}\Big)^2
    + P\sum_i\Big(\sum_k x_{ik}-1\Big)^2
    """)
    st.latex(r"x_{ik}=\tfrac{1-Z_{ik}}{2}\;\Rightarrow\; H_C=\sum_u h_u Z_u+\sum_{u<v}J_{uv}Z_uZ_v")
    st.latex(r"|\psi(\boldsymbol\gamma,\boldsymbol\beta)\rangle=\prod_{p=1}^{P}e^{-i\beta_p\sum_u X_u}"
             r"\,e^{-i\gamma_p H_C}\,|+\rangle^{\otimes n}")
    st.markdown(
        "- **Clásico (OR-Tools):** batería y ventanas de tiempo como dimensiones con restricciones duras/blandas; "
        "la búsqueda nunca visita estados inválidos. No puede expresar el término $\\gamma\\,x_{ik}x_{jl}$ "
        "porque el coste de un arco depende de un solo vehículo.\n"
        "- **Cuántico (QAOA):** todas las restricciones viven en la energía. La batería es una penalización "
        "cuadrática suave (sin qubits de holgura) y one-hot una penalización dura con peso $P$. "
        "La secuencia dentro de cada dron se resuelve clásicamente (codificarla exigiría $N^2M$ qubits).\n"
        "- **Descomposición:** las misiones se reparten en lotes de $\\lfloor n_{max}/M\\rfloor$; "
        "las ya asignadas entran como campo lineal en los lotes siguientes."
    )

# ----------------------------------------------------------------------------
# Pestaña 4: datos
# ----------------------------------------------------------------------------
with tab_data:
    st.markdown("**Flota**")
    st.dataframe(pd.DataFrame([{
        "Dron": d.name, "Velocidad (km/min)": round(d.speed, 2), "Batería (Wh)": round(d.battery, 1),
        "Consumo efectivo (Wh/km)": round(scn.energy_per_km(d), 2),
        "Estacionario efectivo (Wh/min)": round(scn.hover_rate(d), 2),
    } for d in scn.drones]), hide_index=True, use_container_width=True)
    st.markdown("**Misiones**")
    st.dataframe(pd.DataFrame([{
        "Misión": m.id, "x": round(m.x, 2), "y": round(m.y, 2), "Prioridad": m.priority,
        "Servicio (min)": round(m.service_time, 1), "Ventana": f"{m.tw_start:.0f}–{m.tw_end:.0f}",
    } for m in scn.missions]), hide_index=True, use_container_width=True)
