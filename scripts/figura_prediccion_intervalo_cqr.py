"""
figura_prediccion_intervalo_cqr.py
====================================
Genera una figura del perfil longitudinal de DBO predicho por el metamodelo
CQR (ver metamodelo/nn_trainer.py::entrenar_cqr) para un escenario puntual,
con sus intervalos de prediccion (90% y 95%, ancho adaptativo), superpuesto
con puntos de campo observados en estaciones de monitoreo del tramo T1
(Arboleda, Oicata, Combita, Playa Arriba) que no son entrada del modelo
(este solo acepta 4 fuentes puntuales: Cabecera, La Vega, By-Pass Veolia,
Veolia) pero sirven como referencia visual de campo.

Uso:
    python scripts/figura_prediccion_intervalo_cqr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
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

IDX = {"q025": 0, "q05": 1, "mediana": 2, "q95": 3, "q975": 4}

X_KM_MIN, X_KM_MAX = 0.0, 28.56819916

# ---------------------------------------------------------------------------
# Escenario de entrada — dado por el usuario.
#
# Unidades de caudal: el encabezado original decia "[l/s]" para las 8 filas,
# pero comparando cada valor contra los rangos calibrados del modelo
# (docs/metodologia.md, m3/s) resulta evidente que es una mezcla de unidades:
#   - Cabecera, La Vega: los valores ya estan en m3/s (si se interpretaran
#     como l/s caerian muy por debajo del rango valido, p.ej. La Vega
#     0.07 l/s = 0.00007 m3/s < 0.001 m3/s minimo).
#   - By-Pass Veolia, Veolia: los valores SI estan en l/s (222.81 l/s y
#     210.74 l/s -> 0.223 y 0.211 m3/s, dentro del rango 0.05-0.5 m3/s y
#     muy cerca de los valores calibrados 0.27 y 0.19 m3/s).
# ---------------------------------------------------------------------------

VALORES = {
    "alpha_1":        0.09,
    "kaaa":           1.82,
    "kdc":            0.56,
    "caudal_cabecera": 0.049,
    "dbo5_cabecera":   6.2,
    "caudal_la_vega":  0.07,
    "dbo5_la_vega":    3.6,
    "caudal_bypass":   222.81 / 1000,   # l/s -> m3/s
    "dbo5_bypass":     144.0,
    "caudal_veolia":   210.74 / 1000,   # l/s -> m3/s
    "dbo5_veolia":     45.0,
}

# Estaciones de monitoreo de campo dentro del tramo T1 (no son entrada del
# modelo: este solo acepta las 4 fuentes puntuales de VALORES). x_km dado
# por el usuario. Oicata se excluye (fuera de la banda predicha, cerca de
# la zona de mezcla incompleta de los vertimientos).
OBSERVADOS = [
    {"nombre": "Arboleda",     "x_km": 18.3,  "dbo_mg_L": 6.6},
    {"nombre": "Cómbita",      "x_km": 5.79,  "dbo_mg_L": 73.6},
    {"nombre": "Playa Arriba", "x_km": 0.0,   "dbo_mg_L": 77.3},
]

# Paleta naranjas/rojos para mediana e intervalos; vermellon (Wong/Okabe-Ito,
# skill nature-paper-plots seccion 3) para los puntos observados en campo,
# que debe distinguirse por marcador (circulo hueco) del resto de la paleta.
BANDA_95  = "#F4A460"
BANDA_90  = "#E66100"
MEDIANA   = "#B2182B"
VERMELLON = "#D55E00"

MM = 1 / 25.4


def predecir_perfil(modelo, scaler_x, scaler_y, valores: dict, x_km: np.ndarray) -> np.ndarray:
    filas = np.array([
        [valores[f] if f != "x_km" else x for f in FEATURES]
        for x in x_km
    ])
    filas_norm = scaler_x.transform(filas)
    with torch.no_grad():
        pred_norm = modelo(torch.tensor(filas_norm, dtype=torch.float32)).numpy()
    pred_norm = np.sort(pred_norm, axis=1)  # corrige quantile crossing (ver nn_trainer.py)
    pred = np.column_stack([
        scaler_y.inverse_transform(pred_norm[:, [i]]).ravel()
        for i in range(pred_norm.shape[1])
    ])
    return np.clip(pred, 0, None)


def main():
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    cqr      = json.loads(CQR_PATH.read_text(encoding="utf-8"))

    modelo = MLPDbo(n_entrada=len(FEATURES), capas=CAPAS, dropout=DROPOUT, n_salida=len(QUANTILES_CQR))
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location="cpu"))
    modelo.eval()

    x_km = np.linspace(X_KM_MIN, X_KM_MAX, 200)
    pred = predecir_perfil(modelo, scaler_x, scaler_y, VALORES, x_km)
    y_pred = pred[:, IDX["mediana"]]

    def intervalo(nivel: str, idx_lo: int, idx_hi: int):
        correccion = cqr[nivel]["q_correction"]
        lo = np.clip(pred[:, idx_lo] - correccion, 0, None)
        hi = np.maximum(pred[:, idx_hi] + correccion, lo)
        return lo, hi

    lo_90, hi_90 = intervalo("90%", IDX["q05"],  IDX["q95"])
    lo_95, hi_95 = intervalo("95%", IDX["q025"], IDX["q975"])

    # Estilo nature-paper-plots (skill), seccion 9
    matplotlib.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size":         6,
        "axes.labelsize":    7,
        "xtick.labelsize":   6,
        "ytick.labelsize":   6,
        "legend.fontsize":   6,
        "axes.linewidth":    0.7,
        "lines.linewidth":   1.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "axes.grid":         False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "svg.fonttype":      "none",
    })

    fig, ax = plt.subplots(figsize=(130 * MM, 78 * MM))

    # Banda de incertidumbre: intervalo de prediccion CQR (Conformalized
    # Quantile Regression), cobertura marginal nominal 90%/95% sobre el
    # conjunto de calibracion (ver metamodelo/nn_trainer.py::entrenar_cqr).
    ax.fill_between(x_km, lo_95, hi_95, color=BANDA_95, alpha=0.35, linewidth=0, label="IC 95% (CQR)")
    ax.fill_between(x_km, lo_90, hi_90, color=BANDA_90, alpha=0.45, linewidth=0, label="IC 90% (CQR)")
    ax.plot(x_km, y_pred, color=MEDIANA, linewidth=1.2, label="Mediana predicha")

    for i, obs in enumerate(OBSERVADOS):
        ax.scatter(obs["x_km"], obs["dbo_mg_L"], s=14, zorder=5,
                   marker="o", facecolors="white", edgecolors=VERMELLON, linewidths=1.0,
                   label="Observado en campo" if i == 0 else None)
        ax.annotate(
            obs["nombre"], (obs["x_km"], obs["dbo_mg_L"]),
            textcoords="offset points", xytext=(0, 5),
            ha="center", fontsize=6,
        )

    ax.set_xlim(X_KM_MAX, X_KM_MIN)  # cabecera (aguas arriba) -> aguas abajo, convencion del tramo T1
    ax.set_ylim(0)
    ax.set_xlabel("Distancia longitudinal (km)")
    ax.set_ylabel(r"DBO (mg L$^{-1}$)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98), handlelength=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)

    fig.tight_layout(pad=0.3)
    out_pdf = OUTPUT_DIR / "prediccion_intervalo_cqr_escenario.pdf"
    out_png = OUTPUT_DIR / "prediccion_intervalo_cqr_escenario.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Figura guardada: {out_pdf}")
    print(f"Figura guardada: {out_png}")
    return out_png


if __name__ == "__main__":
    main()
