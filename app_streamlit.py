"""
app_streamlit.py
=================
Aplicativo Streamlit para predecir DBO con el metamodelo de red neuronal
entrenado sobre el caso Chicamocha T1 (ver metamodelo/nn_trainer.py).

Permite ingresar los predictores del río (parámetros hidráulicos, cinéticos
y de las fuentes puntuales) y obtener la predicción de DBO en uno o varios
puntos a lo largo del tramo (x_km), con su intervalo de predicción por
Conformalized Quantile Regression (CQR, Romano et al. 2019) — ancho
adaptativo según el punto, en vez de un margen fijo para toda la cuenca.

Uso:
    streamlit run app_streamlit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos      import FEATURES
from metamodelo.nn_trainer import MLPDbo, QUANTILES_CQR

OUTPUT_DIR    = _ROOT / "resultados" / "chicamocha_t1_metamodelo"
MODELO_PATH   = OUTPUT_DIR / "nn_dbo_cqr.pt"
SCALER_X_PATH = OUTPUT_DIR / "nn_scaler_x_cqr.joblib"
SCALER_Y_PATH = OUTPUT_DIR / "nn_scaler_y_cqr.joblib"
CQR_PATH      = OUTPUT_DIR / "nn_cqr.json"

CAPAS   = [32, 128]
DROPOUT = 0.010292247147901223

# Orden de las columnas que predice la red — ver metamodelo/nn_trainer.py::QUANTILES_CQR
IDX_QUANTIL = {"q025": 0, "q05": 1, "mediana": 2, "q95": 3, "q975": 4}

X_KM_MIN, X_KM_MAX = 0.0, 28.56819916

# nombre -> (minimo, maximo, valor_defecto, unidad, etiqueta)
PREDICTORES = {
    "alpha_1":         (0.03,  0.40,   0.095, "-",    "α₁ — parámetro hidráulico"),
    "kaaa":            (0.5,   3.0,    1.82,  "d-1", "Reaireación (kaaa)"),
    "kdc":              (0.3,   3.0,    0.565, "d-1", "Oxidación DBO rápida (kdc)"),
    "caudal_bypass":    (0.05,  0.5,    0.27,  "m³/s", "Caudal By-Pass Veolia"),
    "dbo5_bypass":      (5.0,   600.0,  263.0, "mg/L", "DBO5 By-Pass Veolia"),
    "caudal_veolia":    (0.05,  0.50,   0.19,  "m³/s", "Caudal Veolia"),
    "dbo5_veolia":      (5.0,   600.0,  32.75, "mg/L", "DBO5 Veolia (tratada)"),
    "caudal_la_vega":   (0.001, 1.0,    0.03,  "m³/s", "Caudal R. La Vega"),
    "dbo5_la_vega":     (1.0,   80.0,   3.63,  "mg/L", "DBO5 R. La Vega"),
    "caudal_cabecera":  (0.005, 0.300,  0.029, "m³/s", "Caudal cabecera"),
    "dbo5_cabecera":    (0.5,   50.0,   2.5,   "mg/L", "DBO5 cabecera"),
}


@st.cache_resource
def cargar_modelo():
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    cqr      = json.loads(CQR_PATH.read_text(encoding="utf-8"))

    modelo = MLPDbo(n_entrada=len(FEATURES), capas=CAPAS, dropout=DROPOUT, n_salida=len(QUANTILES_CQR))
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location="cpu"))
    modelo.eval()

    return modelo, scaler_x, scaler_y, cqr


def predecir(modelo, scaler_x, scaler_y, valores: dict, x_km: np.ndarray) -> np.ndarray:
    """Devuelve un arreglo (n_puntos, 5) con los cuantiles [q025, q05, mediana, q95, q975]."""
    filas = np.array([
        [valores[f] if f != "x_km" else x for f in FEATURES]
        for x in x_km
    ])
    filas_norm = scaler_x.transform(filas)
    with torch.no_grad():
        pred_norm = modelo(torch.tensor(filas_norm, dtype=torch.float32)).numpy()

    # Reordenar por si la red predice cuantiles fuera de orden ("quantile
    # crossing") — misma corrección que en el entrenamiento (ver nn_trainer.py).
    pred_norm = np.sort(pred_norm, axis=1)
    pred = np.column_stack([
        scaler_y.inverse_transform(pred_norm[:, [i]]).ravel()
        for i in range(pred_norm.shape[1])
    ])
    return np.clip(pred, 0, None)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Metamodelo del software QUAL2K (Tramo inicial Río Chicamocha)",
    page_icon="🌊", layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&family=Work+Sans:wght@400;500&display=swap');

:root {
    --azul-cobalto: #0943B5;
    --azul-oscuro:  #004884;
    --azul-claro:   #3366CC;
    --gris-oscuro:  #333333;
    --blanco:       #FFFFFF;
    --borde-suave:  #C9D6E8;
    --fondo-suave:  #EEF3FA;
}

html, body, [class*="css"] {
    font-family: 'Work Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: var(--gris-oscuro);
    font-size: 1rem;
}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-weight: 600;
    color: var(--azul-oscuro);
}

/* Franja superior — referencia visual a la barra institucional GOV.CO */
.govco-top-bar {
    background-color: var(--azul-cobalto);
    height: 6px;
    width: 100%;
    border-radius: 3px;
    margin-bottom: 1rem;
}

a, a:visited { color: var(--azul-claro); }
a:hover      { color: var(--azul-oscuro); }

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.9rem 1.1rem 0.6rem 1.1rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px !important;
    border: 1px solid var(--borde-suave) !important;
    background-color: var(--blanco) !important;
}
.card-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--azul-oscuro);
    border-bottom: 2px solid var(--azul-cobalto);
    padding-bottom: 0.3rem;
    margin-bottom: 0.5rem;
}

/* Botones — sólidos, bordes ligeramente redondeados, área táctil >= 44px */
div.stButton > button,
div.stDownloadButton > button {
    background-color: var(--azul-oscuro);
    color: var(--blanco) !important;
    border: none;
    border-radius: 6px;
    font-family: 'Work Sans', sans-serif;
    font-weight: 500;
    min-height: 44px;
    transition: background-color 0.15s ease-in-out;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background-color: var(--azul-claro);
    color: var(--blanco) !important;
}
div.stButton > button:focus-visible,
div.stDownloadButton > button:focus-visible {
    outline: 2px solid var(--azul-cobalto);
    outline-offset: 2px;
}

/* Controles nativos (radio, slider, checkbox) heredan el azul institucional */
input, select, textarea { accent-color: var(--azul-oscuro); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="govco-top-bar"></div>', unsafe_allow_html=True)

col_titulo, col_logo = st.columns([4, 1])
with col_titulo:
    st.markdown(
        "<h1 style='text-align: center;'>Metamodelo del software QUAL2K "
        "(Tramo inicial Río Chicamocha)</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #5A6B85; font-size: 0.85rem; margin-top: -0.5rem;'>"
        "Elaborado por: Sergio David Torres Piraquive — Estudiante de la Maestría en Ciencia de Datos, "
        "Escuela Colombiana de Ingeniería Julio Garavito"
        "</p>",
        unsafe_allow_html=True,
    )
with col_logo:
    st.image(str(_ROOT / "assets" / "logo_escuela.png"), width=170)

modelo, scaler_x, scaler_y, cqr = cargar_modelo()

grupos = {
    "Parámetros hidráulicos":    ["alpha_1"],
    "Tasas cinéticas": ["kaaa", "kdc"],
    "Cabecera":                  ["caudal_cabecera", "dbo5_cabecera"],
    "R. La Vega":                ["caudal_la_vega", "dbo5_la_vega"],
    "By-Pass Veolia":            ["caudal_bypass", "dbo5_bypass"],
    "Veolia (tratada)":          ["caudal_veolia", "dbo5_veolia"],
}


def _sync_desde_slider(nombre: str):
    st.session_state[f"{nombre}_num"] = st.session_state[f"{nombre}_slider"]


def _sync_desde_numero(nombre: str, minimo: float, maximo: float):
    valor = min(max(st.session_state[f"{nombre}_num"], minimo), maximo)
    st.session_state[f"{nombre}_num"]    = valor
    st.session_state[f"{nombre}_slider"] = valor


def _campo_predictor(nombre: str) -> float:
    minimo, maximo, defecto, unidad, etiqueta = PREDICTORES[nombre]
    paso = round((maximo - minimo) / 100, 6)

    if f"{nombre}_slider" not in st.session_state:
        st.session_state[f"{nombre}_slider"] = defecto
        st.session_state[f"{nombre}_num"]    = defecto

    st.caption(f"{etiqueta} ({unidad})")
    col_slider, col_num = st.columns([3, 2])
    with col_slider:
        st.slider(
            etiqueta, min_value=minimo, max_value=maximo, step=paso,
            key=f"{nombre}_slider", label_visibility="collapsed",
            on_change=_sync_desde_slider, args=(nombre,),
        )
    with col_num:
        st.number_input(
            "Valor exacto", min_value=minimo, max_value=maximo, step=paso,
            key=f"{nombre}_num", label_visibility="collapsed",
            on_change=_sync_desde_numero, args=(nombre, minimo, maximo),
        )

    return st.session_state[f"{nombre}_slider"]


st.subheader("Predictores del río")

valores: dict[str, float] = {}
columnas = st.columns(3)
for i, (titulo, nombres) in enumerate(grupos.items()):
    with columnas[i % 3]:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{titulo}</div>', unsafe_allow_html=True)
            for nombre in nombres:
                valores[nombre] = _campo_predictor(nombre)

st.divider()
st.subheader("Predicción")

with st.container(border=True):
    col_modo, col_extra = st.columns([1, 2])
    with col_modo:
        modo = st.radio("Punto(s) del río", ["Un solo km", "Varios km", "Perfil completo"])
    with col_extra:
        if modo == "Un solo km":
            x_km = np.array([st.number_input(
                "Distancia longitudinal (km)",
                min_value=X_KM_MIN, max_value=X_KM_MAX, value=17.0, step=0.5,
            )])
        elif modo == "Varios km":
            texto = st.text_input(
                "Lista de km separados por coma",
                value="0, 5, 10, 15, 20, 25, 30",
            )
            try:
                x_km = np.array(sorted(float(v.strip()) for v in texto.split(",") if v.strip()))
                x_km = np.clip(x_km, X_KM_MIN, X_KM_MAX)
            except ValueError:
                st.error("Formato inválido — usa números separados por coma (ej: 0, 5, 10).")
                st.stop()
        else:
            n_puntos = st.slider("Número de puntos", min_value=5, max_value=50, value=20)
            x_km = np.linspace(X_KM_MIN, X_KM_MAX, n_puntos)

        nivel = st.radio("Nivel del intervalo", ["90%", "95%"], index=1, horizontal=True)
        st.caption(f"Intervalo de predicción: {nivel} (Conformalized Quantile Regression — ancho adaptativo)")

    if nivel == "90%":
        idx_lo, idx_hi = IDX_QUANTIL["q05"], IDX_QUANTIL["q95"]
    else:
        idx_lo, idx_hi = IDX_QUANTIL["q025"], IDX_QUANTIL["q975"]
    correccion = cqr[nivel]["q_correction"]
    predecir_click = st.button("Predecir", type="primary", use_container_width=True)

if predecir_click:
    pred   = predecir(modelo, scaler_x, scaler_y, valores, x_km)
    y_pred = pred[:, IDX_QUANTIL["mediana"]]
    lo = np.clip(pred[:, idx_lo] - correccion, 0, None)
    # Piso de ancho no-negativo: cuando la corrección CQR encoge el intervalo
    # (ver metamodelo/nn_trainer.py::entrenar_cqr) puede cruzarlo en puntos de
    # spread muy angosto; el caso límite es un intervalo puntual, nunca uno invertido.
    hi = np.maximum(pred[:, idx_hi] + correccion, lo)

    with st.container(border=True):
        if len(x_km) == 1:
            st.metric(f"DBO predicha en x = {x_km[0]:.2f} km", f"{y_pred[0]:.2f} mg/L")
            st.caption(f"Intervalo {nivel}: [{lo[0]:.2f}, {hi[0]:.2f}] mg/L")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_km, x_km[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor="rgba(9, 67, 181, 0.12)",
                line=dict(color="rgba(255, 255, 255, 0)"),
                hoverinfo="skip",
                name=f"Intervalo {nivel}",
            ))
            fig.add_trace(go.Scatter(
                x=x_km, y=y_pred,
                mode="lines+markers",
                line=dict(color="#004884", width=2.5),
                marker=dict(size=6, color="#004884"),
                name="DBO predicha",
                hovertemplate="x = %{x:.2f} km<br>DBO = %{y:.2f} mg/L<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_white",
                height=380,
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(orientation="h", y=1.15, x=0),
                font=dict(family="Work Sans, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                          color="#333333"),
                xaxis=dict(title="Distancia longitudinal (km)", autorange="reversed"),
                yaxis=dict(title="DBO (mg/L)", rangemode="tozero"),
            )
            col_izq, col_fig, col_der = st.columns([1, 3, 1])
            with col_fig:
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                {
                    "x_km":            x_km,
                    "DBO_mg_L":        y_pred,
                    f"IC_{nivel}_lo":  lo,
                    f"IC_{nivel}_hi":  hi,
                },
                use_container_width=True,
                hide_index=True,
            )
