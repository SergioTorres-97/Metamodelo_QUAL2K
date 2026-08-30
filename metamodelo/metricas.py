"""
metricas.py
============
Calcula y muestra las métricas de evaluación del metamodelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular(y_true, y_pred, nombre: str = "Modelo") -> dict:
    """
    Calcula R², RMSE, MAE y sesgo medio.

    Returns:
        dict con las métricas.
    """
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(np.array(y_pred) - np.array(y_true)))

    print(f"\n{'=' * 45}")
    print(f"  MÉTRICAS — {nombre}")
    print(f"{'=' * 45}")
    print(f"  R²    : {r2:.4f}")
    print(f"  RMSE  : {rmse:.4f} mg/L")
    print(f"  MAE   : {mae:.4f} mg/L")
    print(f"  Sesgo : {bias:+.4f} mg/L")
    print(f"{'=' * 45}\n")

    return {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias}


def intervalo_conformal(y_true, y_pred, alpha: float = 0.10) -> dict:
    """
    Calcula el margen de un intervalo de predicción por split conformal
    prediction (Lei et al., 2018), usando los residuos absolutos de un
    conjunto de calibración (aquí, el conjunto de test) que no participó
    en el entrenamiento ni en la selección de hiperparámetros del modelo.

    Garantía teórica (bajo exchangeability entre calibración y predicción
    futura): P(y_nuevo ∈ [ŷ - qhat, ŷ + qhat]) ≥ 1 - alpha, en muestra
    finita y sin supuestos sobre la distribución del error.

    Args:
        y_true : valores observados del conjunto de calibración.
        y_pred : valores predichos por el modelo para ese mismo conjunto.
        alpha  : nivel de significancia (0.10 → cobertura nominal 90 %).

    Returns:
        dict con qhat (margen en mg/L), alpha, cobertura nominal y n usado.
    """
    residuos = np.abs(np.array(y_pred) - np.array(y_true))
    n = len(residuos)

    nivel = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat  = float(np.quantile(residuos, nivel, method="higher"))

    print(f"[conformal] alpha={alpha:.2f}  (cobertura nominal {1 - alpha:.0%})  "
          f"→ margen qhat = ±{qhat:.4f} mg/L  (n calibración = {n:,})")

    return {"alpha": alpha, "cobertura_nominal": 1 - alpha, "qhat": qhat, "n": n}


def intervalo_cqr(y_true, y_lo, y_hi, alpha: float = 0.10) -> dict:
    """
    Calcula la corrección de calibración de un intervalo de predicción por
    Conformalized Quantile Regression (Romano, Patterson & Candès, 2019,
    "Conformalized Quantile Regression", NeurIPS).

    A diferencia de intervalo_conformal (margen fijo qhat para toda la
    cuenca), aquí la red ya predice cuantiles bajo/alto que varían según la
    entrada (más angostos donde el modelo tiene más confianza); esta función
    solo calcula la corrección escalar necesaria para garantizar la
    cobertura marginal nominal sobre el conjunto de calibración.

    Score de no-conformidad: E_i = max(y_lo_i - y_i, y_i - y_hi_i)
    Intervalo final: [y_lo - q_correction, y_hi + q_correction]

    Garantía teórica (bajo exchangeability): P(y_nuevo ∈ [y_lo - q_correction,
    y_hi + q_correction]) ≥ 1 - alpha, en muestra finita.

    Args:
        y_true : valores observados del conjunto de calibración.
        y_lo   : cuantil bajo predicho por la red para ese mismo conjunto.
        y_hi   : cuantil alto predicho por la red para ese mismo conjunto.
        alpha  : nivel de significancia (0.10 → cobertura nominal 90 %).

    Returns:
        dict con q_correction (ajuste en mg/L), alpha, cobertura nominal y n usado.
    """
    y_true = np.asarray(y_true)
    y_lo   = np.asarray(y_lo)
    y_hi   = np.asarray(y_hi)

    E = np.maximum(y_lo - y_true, y_true - y_hi)
    n = len(E)

    nivel        = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_correction = float(np.quantile(E, nivel, method="higher"))

    print(f"[cqr] alpha={alpha:.2f}  (cobertura nominal {1 - alpha:.0%})  "
          f"→ corrección = {q_correction:+.4f} mg/L  (n calibración = {n:,})")

    return {"alpha": alpha, "cobertura_nominal": 1 - alpha,
            "q_correction": q_correction, "n": n}
