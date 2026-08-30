"""
chicamocha_t1_costo_computacional.py
=====================================
Compara el costo computacional (tiempo de cómputo) entre el modelo
convencional QUAL2K (FORTRAN) y el metamodelo Red Neuronal (MLP,
metamodelo/nn_trainer.py) para el tramo T1 del río Chicamocha.

No evalúa precisión — eso ya se hizo con el lote de mediciones/test set
existente (ver resultados_nn.xlsx). Este script mide únicamente tiempo
de ejecución por escenario, para cuantificar el beneficio del metamodelo
en análisis que requieren muchas corridas (Monte Carlo, calibración
global, análisis de incertidumbre).

Metodología:
  1. Genera N escenarios aleatorios (LHS) sobre las mismas 16 variables
     sensibles usadas para construir la BD de entrenamiento
     (examples/chicamocha_t1_metamodelo_bd.py).
  2. Para cada escenario:
       - Corre QUAL2K completo (escritura config + ejecución FORTRAN +
         lectura de resultados) y mide el tiempo de pared.
       - Evalúa el metamodelo NN sobre ese mismo escenario, para los
         mismos puntos x_km del tramo, y mide el tiempo de pared.
  3. Reporta tiempo promedio ± desviación estándar de cada uno, el
     speedup, y una extrapolación a tamaños típicos de un análisis de
     incertidumbre / calibración (500, 5 000 y 50 000 corridas).

La inferencia del metamodelo se fuerza a CPU (aunque haya GPU disponible)
para comparar en igualdad de condiciones de hardware con QUAL2K, que
corre en CPU. Con GPU el metamodelo sería aún más rápido.

Uso:
    python examples/chicamocha_t1_costo_computacional.py
    python examples/chicamocha_t1_costo_computacional.py --n 30 --seed 7
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

# Reutiliza el generador de escenarios LHS y el runner de QUAL2K ya
# escritos para construir la BD de entrenamiento — misma parametrización,
# mismo código de ejecución, cero duplicación de esa lógica.
import chicamocha_t1_metamodelo_bd as bdmod

from metamodelo.datos      import FEATURES
from metamodelo.nn_trainer import MLPDbo

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

OUTPUT_DIR    = _ROOT / "resultados" / "chicamocha_t1_metamodelo"
MODELO_PATH   = OUTPUT_DIR / "nn_dbo.pt"
SCALER_X_PATH = OUTPUT_DIR / "nn_scaler_x.joblib"
SCALER_Y_PATH = OUTPUT_DIR / "nn_scaler_y.joblib"
BENCH_DIR     = OUTPUT_DIR / "_bench_costo"
EXCEL_PATH    = OUTPUT_DIR / "costo_computacional.xlsx"
FIG_PATH      = OUTPUT_DIR / "costo_computacional.png"

# Arquitectura del modelo guardado — debe coincidir con la usada al
# entrenar (ver bloque __main__ de metamodelo/nn_trainer.py).
_CAPAS_NN = [32, 128]

# Tamaños de análisis típicos para la extrapolación (Monte Carlo /
# calibración global / análisis de incertidumbre).
_TAMANOS_EXTRAPOLACION = [500, 5_000, 50_000]


# ---------------------------------------------------------------------------
# Metamodelo NN — carga e inferencia
# ---------------------------------------------------------------------------

def cargar_modelo_nn():
    modelo = MLPDbo(n_entrada=len(FEATURES), capas=_CAPAS_NN, dropout=0.0)
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location="cpu"))
    modelo.eval()

    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return modelo, scaler_x, scaler_y


def predecir_nn(modelo: MLPDbo, scaler_x, scaler_y, X_df: pd.DataFrame) -> np.ndarray:
    X = scaler_x.transform(X_df[FEATURES].values)
    with torch.no_grad():
        y_norm = modelo(torch.tensor(X, dtype=torch.float32)).numpy()
    return scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# Benchmark principal
# ---------------------------------------------------------------------------

def correr_benchmark(n: int = 30, seed: int = 42) -> pd.DataFrame:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    with open(bdmod.JSON_BASE, encoding="utf-8") as fh:
        config_base = json.load(fh)

    muestras     = bdmod._muestrear_lhs(bdmod.PARAMETROS, n=n, seed=seed)
    nombres_vars = [p.nombre for p in bdmod.PARAMETROS]

    print("=" * 65)
    print("COSTO COMPUTACIONAL — QUAL2K vs METAMODELO (Red Neuronal)")
    print(f"  Escenarios LHS   : {n}")
    print(f"  Semilla          : {seed}")
    print(f"  Modelo NN        : {MODELO_PATH}")
    print("=" * 65)

    modelo_nn, scaler_x, scaler_y = cargar_modelo_nn()

    filas = []
    x_km_ref = None

    for i in range(n):
        valores = {nombre: float(muestras[nombre][i]) for nombre in nombres_vars}

        cfg = copy.deepcopy(config_base)
        for param in bdmod.PARAMETROS:
            bdmod._modificar_config(cfg, param, valores[param.nombre])

        run_dir = str(BENCH_DIR / f"run_{i:03d}")

        # ── QUAL2K ────────────────────────────────────────────────────────
        t0    = time.perf_counter()
        res   = bdmod._worker_simulacion((i, cfg, run_dir))
        t_q2k = time.perf_counter() - t0

        if not res["exito"]:
            print(f"  [{i + 1:>3}/{n}]  QUAL2K FALLÓ — escenario omitido  "
                  f"[{res.get('error', '')}]")
            continue

        df_result = pd.DataFrame({"x_km": res["x_km"], "dbo_mg_L": res["dbo_mg_L"]})

        if x_km_ref is None:
            x_km_ref = df_result["x_km"].values

        # ── Metamodelo NN — mismo escenario, mismos puntos x_km ────────────
        X_esc = pd.DataFrame(
            [{**valores, "x_km": x} for x in x_km_ref]
        )[FEATURES]

        t0 = time.perf_counter()
        predecir_nn(modelo_nn, scaler_x, scaler_y, X_esc)
        t_nn = time.perf_counter() - t0

        speedup = t_q2k / t_nn
        print(f"  [{i + 1:>3}/{n}]  QUAL2K {t_q2k:7.3f} s   |   "
              f"NN {t_nn * 1000:7.2f} ms   |   speedup ×{speedup:,.0f}")

        filas.append({"escenario": i, "t_qual2k_s": t_q2k, "t_nn_s": t_nn})

    shutil.rmtree(BENCH_DIR, ignore_errors=True)

    df = pd.DataFrame(filas)
    if not df.empty:
        # La primera inferencia de la NN incluye el cold-start de PyTorch
        # (init de threads/kernels en CPU) y no es representativa del
        # tiempo en régimen estable — se marca para excluirla del resumen.
        df["nn_cold_start"] = False
        df.loc[df.index[0], "nn_cold_start"] = True

    return df


# ---------------------------------------------------------------------------
# Resumen, extrapolación y salida
# ---------------------------------------------------------------------------

def resumir(df: pd.DataFrame) -> dict:
    cold_start = df["nn_cold_start"]
    df_estable = df.loc[~cold_start]

    t_q2k = df["t_qual2k_s"]
    t_nn  = df_estable["t_nn_s"]

    resumen = {
        "n_escenarios":      len(df),
        "nn_n_excluidos":    int(cold_start.sum()),
        "nn_cold_start_ms":  float(df.loc[cold_start, "t_nn_s"].iloc[0] * 1000) if cold_start.any() else None,
        "qual2k_media_s":    t_q2k.mean(),
        "qual2k_std_s":      t_q2k.std(),
        "nn_media_s":        t_nn.mean(),
        "nn_std_s":          t_nn.std(),
        "speedup_medio":     (df_estable["t_qual2k_s"] / df_estable["t_nn_s"]).mean(),
    }

    print("\n" + "=" * 65)
    print("RESUMEN")
    print("=" * 65)
    if resumen["nn_n_excluidos"]:
        print(f"  (excluido el cold-start de la NN: "
              f"{resumen['nn_cold_start_ms']:.1f} ms en el 1er escenario — "
              f"init de PyTorch, no representativo)")
    print(f"  QUAL2K  : {resumen['qual2k_media_s']:7.3f} ± "
          f"{resumen['qual2k_std_s']:.3f} s/escenario  (n={len(df)})")
    print(f"  NN      : {resumen['nn_media_s'] * 1000:7.2f} ± "
          f"{resumen['nn_std_s'] * 1000:.2f} ms/escenario  (n={len(t_nn)})")
    print(f"  Speedup : ×{resumen['speedup_medio']:,.0f}  (promedio por escenario)")

    print("\n  Extrapolación (ejecución secuencial):")
    print(f"  {'N corridas':>12} | {'QUAL2K':>14} | {'NN':>14}")
    for tam in _TAMANOS_EXTRAPOLACION:
        t_q2k_tot = str_horas(resumen["qual2k_media_s"] * tam)
        t_nn_tot  = str_horas(resumen["nn_media_s"] * tam)
        print(f"  {tam:>12,} | {t_q2k_tot:>14} | {t_nn_tot:>14}")
    print("=" * 65)

    return resumen


def str_horas(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.1f} s"
    if segundos < 3600:
        return f"{segundos / 60:.1f} min"
    return f"{segundos / 3600:.2f} h"


def guardar_resultados(df: pd.DataFrame, resumen: dict):
    df_extrap = pd.DataFrame([
        {
            "n_corridas":      tam,
            "qual2k_total_s":  resumen["qual2k_media_s"] * tam,
            "nn_total_s":      resumen["nn_media_s"] * tam,
            "qual2k_total":    str_horas(resumen["qual2k_media_s"] * tam),
            "nn_total":        str_horas(resumen["nn_media_s"] * tam),
        }
        for tam in _TAMANOS_EXTRAPOLACION
    ])

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="tiempos_por_escenario", index=False)
        pd.DataFrame([resumen]).to_excel(writer, sheet_name="resumen", index=False)
        df_extrap.to_excel(writer, sheet_name="extrapolacion", index=False)
    print(f"\n[costo] Excel guardado : {EXCEL_PATH}")

    _grafica_comparacion(df, resumen)
    print(f"[costo] Figura guardada: {FIG_PATH}")


def _grafica_comparacion(df: pd.DataFrame, resumen: dict):
    """
    Costo computacional acumulado: tiempo total (QUAL2K vs NN) en función
    del número de corridas N. Tramo sólido = medido (cumsum real de los
    escenarios del benchmark); tramo punteado = extrapolado más allá de lo
    medido, usando el tiempo promedio por escenario (NN sin el cold-start).
    """
    # QUAL2K no tiene outliers: usa todos los puntos medidos.
    n_medido       = len(df)
    cum_q2k_medido = df["t_qual2k_s"].cumsum().values
    x_medido       = np.arange(1, n_medido + 1)

    # La NN sí tiene el cold-start de PyTorch en el primer escenario: se
    # excluye también de la curva (no solo del resumen), porque es un
    # costo fijo de una sola vez y no una tasa por corrida — mezclado con
    # el cumsum distorsiona la forma de la curva (la aplana al inicio y
    # la hace ver potencial en vez de lineal). Se grafica aparte como
    # anotación.
    df_estable       = df.loc[~df["nn_cold_start"]]
    n_medido_nn      = len(df_estable)
    cum_nn_medido    = df_estable["t_nn_s"].cumsum().values
    x_medido_nn      = np.arange(1, n_medido_nn + 1)

    n_max    = max(_TAMANOS_EXTRAPOLACION)
    x_extrap = np.geomspace(n_medido, n_max, 50)
    cum_q2k_extrap = cum_q2k_medido[-1] + resumen["qual2k_media_s"] * (x_extrap - n_medido)
    cum_nn_extrap  = cum_nn_medido[-1]  + resumen["nn_media_s"]     * (x_extrap - n_medido_nn)

    color_q2k, color_nn = "#E64B35", "#3C5488"

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x_medido, cum_q2k_medido, color=color_q2k, linewidth=2,
            label="QUAL2K (convencional) — medido")
    ax.plot(x_extrap, cum_q2k_extrap, color=color_q2k, linewidth=2, linestyle="--",
            label="QUAL2K — extrapolado")

    ax.plot(x_medido_nn, cum_nn_medido, color=color_nn, linewidth=2,
            label="Metamodelo NN — medido")
    ax.plot(x_extrap, cum_nn_extrap, color=color_nn, linewidth=2, linestyle="--",
            label="Metamodelo NN — extrapolado")

    for tam in _TAMANOS_EXTRAPOLACION:
        ax.axvline(tam, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Número de corridas (N, escala log)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Tiempo acumulado (s, escala log)", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Costo computacional acumulado — QUAL2K vs Metamodelo (NN)\n"
        f"speedup ×{resumen['speedup_medio']:,.0f} por escenario",
        fontweight="bold", fontsize=12,
    )
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":", color="#E5E5E5", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Fija los límites ANTES de anotar, con margen suficiente para que las
    # etiquetas de N=500/5.000/50.000 (arriba de QUAL2K, debajo de la NN)
    # no queden cortadas por el marco de la gráfica.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0, x1 * 1.8)
    ax.set_ylim(y0 * 0.5, y1 * 2.5)

    for tam in _TAMANOS_EXTRAPOLACION:
        t_q2k_tot = resumen["qual2k_media_s"] * tam
        t_nn_tot  = resumen["nn_media_s"] * tam
        es_ultimo = (tam == n_max)
        ha        = "right" if es_ultimo else "center"
        x_texto   = tam * (0.75 if es_ultimo else 1.0)

        ax.text(x_texto, t_q2k_tot * 1.5, f"N={tam:,}\n{str_horas(t_q2k_tot)}",
                ha=ha, va="bottom", fontsize=8, color=color_q2k)
        ax.text(x_texto, t_nn_tot * 0.35, str_horas(t_nn_tot),
                ha=ha, va="top", fontsize=8, color=color_nn)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chicamocha_t1_costo_computacional",
        description="Compara el tiempo de cómputo QUAL2K vs metamodelo (NN).",
    )
    p.add_argument("--n",    type=int, default=30, help="N° de escenarios LHS (default: 30).")
    p.add_argument("--seed", type=int, default=42, help="Semilla LHS (default: 42).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    df_tiempos = correr_benchmark(n=args.n, seed=args.seed)
    if df_tiempos.empty:
        print("\n[costo] Todas las corridas QUAL2K fallaron — no hay nada que comparar.")
        sys.exit(1)

    resumen_dict = resumir(df_tiempos)
    guardar_resultados(df_tiempos, resumen_dict)
