"""
figura_barras_costo_computacional.py
======================================
Grafica de barras agrupadas: tiempo total de computo (escala log) para
QUAL2K vs. el metamodelo (red neuronal), en funcion del numero de corridas
(N=500, 5 000, 50 000). Complementa a costo_computacional.png (linea
acumulada) con una lectura mas directa para comparar barras.

Usa los datos ya calculados en costo_computacional.xlsx (hoja
'extrapolacion'), generados por caso_estudio_t_rio_jordan/t_rio_jordan_costo_computacional.py
-- no vuelve a correr el benchmark ni QUAL2K.

Uso:
    python scripts/figura_barras_costo_computacional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _ROOT / "resultados" / "t_rio_jordan_metamodelo"
XLSX_PATH  = OUTPUT_DIR / "costo_computacional.xlsx"
OUT_PDF    = OUTPUT_DIR / "barras_costo_computacional.pdf"
OUT_PNG    = OUTPUT_DIR / "barras_costo_computacional.png"

MM = 1 / 25.4

# Mismos colores usados en costo_computacional.png para el mismo contraste
# QUAL2K (mecanicista) vs metamodelo (red neuronal)
COLOR_Q2K = "#E64B35"
COLOR_NN  = "#3C5488"


def str_tiempo(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.1f} s"
    if segundos < 3600:
        return f"{segundos / 60:.1f} min"
    return f"{segundos / 3600:.2f} h"


def main() -> None:
    df = pd.read_excel(XLSX_PATH, sheet_name="extrapolacion")
    n_corridas   = df["n_corridas"].to_numpy()
    t_qual2k     = df["qual2k_total_s"].to_numpy()
    t_nn         = df["nn_total_s"].to_numpy()

    resumen = pd.read_excel(XLSX_PATH, sheet_name="resumen").iloc[0]
    speedup = resumen["speedup_medio"]

    matplotlib.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size":         7,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "axes.linewidth":    0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.grid":         False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "svg.fonttype":      "none",
    })

    fig, ax = plt.subplots(figsize=(140 * MM, 85 * MM))

    x = np.arange(len(n_corridas))
    ancho = 0.36

    barras_q2k = ax.bar(x - ancho / 2, t_qual2k, ancho, color=COLOR_Q2K,
                         label="QUAL2K (mecanicista)")
    barras_nn  = ax.bar(x + ancho / 2, t_nn, ancho, color=COLOR_NN,
                         label="Metamodelo (red neuronal)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}".replace(",", " ") for n in n_corridas])
    ax.set_xlabel("Número de corridas")
    ax.set_ylabel("Tiempo total de cómputo (s, escala log)")
    ax.set_title(f"Costo computacional: QUAL2K vs. metamodelo "
                 f"(aceleración ≈{speedup:,.0f}×)", fontsize=8)

    for barras, tiempos in [(barras_q2k, t_qual2k), (barras_nn, t_nn)]:
        for rect, t in zip(barras, tiempos):
            ax.annotate(str_tiempo(t), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=6.5)

    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 3)  # espacio para las etiquetas superiores

    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.grid(axis="y", which="major", linestyle="-", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Figura guardada: {OUT_PDF}")
    print(f"Figura guardada: {OUT_PNG}")


if __name__ == "__main__":
    main()
