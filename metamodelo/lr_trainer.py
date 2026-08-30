"""
lr_trainer.py
=============
Entrena un metamodelo de Regresión Lineal (OLS / Ridge / Lasso) para predecir
DBO (mg/L) en función de las 16 variables sensibles + distancia x_km.

Target en escala original (mg/L), sin transformación logarítmica.

Validación completa de supuestos estadísticos:
    1. Linealidad        — residuos vs ajustados + test RESET (Ramsey)
    2. Independencia     — estadístico Durbin-Watson
    3. Homocedasticidad  — test Breusch-Pagan + gráfica Scale-Location
    4. Normalidad        — Anderson-Darling + Kolmogorov-Smirnov + Q-Q
    5. Multicolinealidad — VIF por variable + número de condición + heatmap corr.
    6. Outliers/influencia — distancia de Cook + leverage + residuos studentizados

Nota sobre los tests en dataset grande:
    Los tests que requieren inversiones de matrices (RESET, Cook, leverage) se
    evalúan sobre una submuestra del conjunto de entrenamiento (n_supuestos,
    default 10 000 filas).  VIF, Durbin-Watson y Breusch-Pagan se computan
    sobre la submuestra.

Split de datos:
    Train  80 % — ajusta los parámetros del modelo
    Test   20 % — evaluación final honesta

Uso:
    Editar los parámetros en el bloque __main__ al final del archivo y ejecutar:
        python metamodelo/lr_trainer.py
    Para cambiar variante: tipo_modelo = "ols" | "ridge" | "lasso".
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

# En Windows, si stdout se redirige a un archivo (p.ej. `> log.txt`), Python usa
# la codificación de la consola (cp1252) en vez de UTF-8, y falla al imprimir
# caracteres como "→". Se fuerza UTF-8 para que la ejecución no se caiga a mitad
# de la busqueda/entrenamiento por un simple print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import anderson, kstest, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
    OLSInfluence,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos    import cargar_datos, FEATURES, TARGET
from metamodelo.metricas import calcular

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

OUTPUT_DIR  = _ROOT / "resultados" / "t_rio_jordan_metamodelo"
MODELO_PATH = OUTPUT_DIR / "lr_dbo.joblib"
FIGS_DIR    = OUTPUT_DIR / "figuras_lr"

_NOMBRE_CORTO = "LR"

# Paleta de colores (misma que los demás trainers)
_C_AZUL  = "#3C5488"
_C_ROJO  = "#E64B35"
_C_NARANJA = "#F39B7F"
_C_CYAN  = "#4DBBD5"


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def entrenar(
    tipo_modelo: str   = "ols",   # "ols" | "ridge" | "lasso"
    alpha:       float = 1.0,     # alpha inicial (ignorado si CV activa)
    cv_alpha:    int   = 5,       # folds para búsqueda de alpha (Ridge/Lasso)
    test_size:   float = 0.20,
    seed:        int   = 42,
    n_perm:      int   = 30,
    n_supuestos: int   = 10_000,  # filas para tests costosos
) -> LinearRegression | Ridge | Lasso:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    nombre_largo = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso"}[tipo_modelo]
    print("=" * 60)
    print(f"  METAMODELO DBO — Regresión Lineal ({nombre_largo})")
    print("=" * 60)

    # ── 1. Split por sim_id — train/test ─────────────────────────────────────
    df      = cargar_datos()
    sim_ids = df["sim_id"].unique()

    ids_train, ids_test = train_test_split(
        sim_ids, test_size=test_size, random_state=seed
    )

    train_df = df[df["sim_id"].isin(ids_train)]
    test_df  = df[df["sim_id"].isin(ids_test)]

    print(f"\n[datos] Train : {len(train_df):,} filas ({len(ids_train)} sims)")
    print(f"[datos] Test  : {len(test_df):,} filas ({len(ids_test)} sims)\n")

    # ── 2. Variables y target en escala original (mg/L) ──────────────────────
    X_train = train_df[FEATURES].values
    X_test  = test_df[FEATURES].values

    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    print(f"[datos] Target DBO — rango train: "
          f"[{y_train.min():.1f}, {y_train.max():.1f}] mg/L\n")

    # ── 3. Entrenar modelo ───────────────────────────────────────────────────
    alpha_final: float | None = None
    t0 = time.time()

    if tipo_modelo == "ols":
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

    elif tipo_modelo == "ridge":
        alphas_grid = np.logspace(-4, 4, 60)
        cv_model    = RidgeCV(alphas=alphas_grid, cv=cv_alpha)
        cv_model.fit(X_train, y_train)
        alpha_final = float(cv_model.alpha_)
        print(f"[Ridge] α óptimo (RidgeCV {cv_alpha}-fold): {alpha_final:.6g}")
        modelo = Ridge(alpha=alpha_final)
        modelo.fit(X_train, y_train)

    else:  # lasso
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = LassoCV(cv=cv_alpha, max_iter=20_000, random_state=seed, n_jobs=-1)
            modelo.fit(X_train, y_train)
        alpha_final = float(modelo.alpha_)
        n_nz = int(np.sum(modelo.coef_ != 0))
        print(f"[Lasso] α óptimo (LassoCV {cv_alpha}-fold): {alpha_final:.6g}")
        print(f"[Lasso] Coeficientes ≠ 0: {n_nz} / {len(FEATURES)}")

    print(f"[{nombre_largo}] Entrenamiento completado en {time.time() - t0:.1f} s")

    # ── 4. Predicciones en mg/L ───────────────────────────────────────────────
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    metricas_train = calcular(y_train, y_pred_train, nombre=f"Train ({nombre_largo})")
    metricas_test  = calcular(y_test,  y_pred_test,  nombre=f"Test ({nombre_largo})")

    # ── 5. Validación de supuestos (en mg/L) ──────────────────────────────────
    residuos_train  = y_train - y_pred_train
    ajustados_train = y_pred_train

    resultados_sup = _validar_supuestos(
        X             = X_train,
        y             = y_train,
        residuos      = residuos_train,
        ajustados     = ajustados_train,
        feature_names = FEATURES,
        nombre        = nombre_largo,
        n_max         = n_supuestos,
        seed          = seed,
    )

    # ── 6. Coeficientes en escala original ───────────────────────────────────
    coefs = pd.Series(modelo.coef_, index=FEATURES)

    # ── 7. Permutation Importance (en mg/L) ───────────────────────────────────
    print(f"\n[{nombre_largo}] Calculando permutation importance ({n_perm} repeticiones)...")
    perm = permutation_importance(
        modelo, X_test, y_test,
        n_repeats    = n_perm,
        random_state = seed,
        scoring      = "r2",
        n_jobs       = -1,
    )
    imp_mean = pd.Series(perm.importances_mean, index=FEATURES)
    imp_std  = pd.Series(perm.importances_std,  index=FEATURES)

    print(f"[{nombre_largo}] Top-5 features (ΔR² en mg/L):")
    for feat, val in imp_mean.sort_values(ascending=False).head(5).items():
        print(f"       {feat:<25} Δ R² = {val:+.4f}")

    # ── 8. Gráficas ───────────────────────────────────────────────────────────
    _grafica_obs_vs_pred(y_test, y_pred_test, nombre_largo)
    _grafica_residuos_mg_L(y_test, y_pred_test, nombre_largo)
    _grafica_coeficientes(coefs, nombre_largo)
    _grafica_importancia_permutation(imp_mean, imp_std, nombre_largo)
    _grafica_no_linealidad(train_df)

    # ── 9. Excel ──────────────────────────────────────────────────────────────
    _exportar_excel(
        metricas_train       = metricas_train,
        metricas_test        = metricas_test,
        y_test               = y_test,
        y_pred               = y_pred_test,
        coefs                = coefs,
        imp_mean             = imp_mean,
        imp_std              = imp_std,
        resultados_sup       = resultados_sup,
        config               = dict(
            tipo_modelo      = nombre_largo,
            alpha            = alpha_final if alpha_final is not None else "N/A",
            cv_folds_alpha   = cv_alpha if tipo_modelo != "ols" else "N/A",
            n_supuestos      = n_supuestos,
            target_transform = "ninguna (mg/L directo)",
            features_scaled  = "ninguna (escala original)",
        ),
    )

    # ── 10. Guardar modelo ────────────────────────────────────────────────────
    joblib.dump({"modelo": modelo}, MODELO_PATH)
    print(f"\n[{nombre_largo}] Modelo guardado  : {MODELO_PATH}")

    return modelo


# ---------------------------------------------------------------------------
# Validación de supuestos
# ---------------------------------------------------------------------------

def _validar_supuestos(
    X:             np.ndarray,
    y:             np.ndarray,
    residuos:      np.ndarray,
    ajustados:     np.ndarray,
    feature_names: list[str],
    nombre:        str,
    n_max:         int,
    seed:          int,
) -> dict:
    """
    Ejecuta los 6 supuestos de la regresión lineal y devuelve un dict con
    todos los resultados para su exportación a Excel y PDF.
    """
    rng = np.random.default_rng(seed)
    n   = len(residuos)

    # Submuestra para tests que requieren inversión de matrices
    idx_sub     = rng.choice(n, size=min(n, n_max), replace=False)
    X_sub       = X[idx_sub]
    y_sub       = y[idx_sub]
    resid_sub   = residuos[idx_sub]
    ajust_sub   = ajustados[idx_sub]

    print("\n" + "=" * 60)
    print("  VALIDACIÓN DE SUPUESTOS — Regresión Lineal")
    print(f"  (submuestra para tests costosos: {len(idx_sub):,} / {n:,} filas)")
    print("=" * 60)

    res = {}
    res["linealidad"]      = _sup_linealidad(X_sub, y_sub, resid_sub, ajust_sub, nombre)
    res["independencia"]   = _sup_independencia(residuos)
    res["homocedasticidad"]= _sup_homocedasticidad(resid_sub, X_sub, ajust_sub, nombre)
    res["normalidad"]      = _sup_normalidad(residuos, nombre, seed)
    res["multicolinealidad"]= _sup_multicolinealidad(X_sub, feature_names)
    res["outliers"]        = _sup_outliers(X_sub, y_sub, nombre, n_total=n)

    # Resumen global
    print("\n── Resumen de supuestos ───────────────────────────────────")
    veredictos = {
        "Linealidad (RESET)":             res["linealidad"]["conclusion"],
        "Independencia (DW)":             res["independencia"]["conclusion"],
        "Homocedasticidad (BP)":          res["homocedasticidad"]["conclusion"],
        "Normalidad (Anderson-Darling)":  res["normalidad"]["anderson_darling"]["conclusion"],
        "Normalidad (KS)":                res["normalidad"]["ks"]["conclusion"],
        "Multicolinealidad (VIF)":        res["multicolinealidad"]["conclusion_vif"],
        "Multicolinealidad (condición)":  res["multicolinealidad"]["conclusion_cond"],
    }
    for nombre_sup, veredicto in veredictos.items():
        icono = "✓" if "CUMPLIDO" in veredicto else ("!" if "ADVERT" in veredicto else "✗")
        print(f"  {icono}  {nombre_sup:<38} {veredicto}")

    return res


# ── Supuesto 1: Linealidad ────────────────────────────────────────────────────

def _sup_linealidad(X_sub, y_sub, resid_sub, ajust_sub, nombre):
    print("\n── Supuesto 1: Linealidad ─────────────────────────────────")

    # Test RESET (Ramsey): añade ŷ², ŷ³ y testa si mejoran el ajuste
    X_sm     = sm.add_constant(X_sub)
    ols_res  = sm.OLS(y_sub, X_sm).fit()

    try:
        from statsmodels.stats.diagnostic import linear_reset
        reset    = linear_reset(ols_res, power=3, use_f=True)
        f_reset  = float(reset.statistic)
        p_reset  = float(reset.pvalue)
        conclusion = "VIOLADO (no-linealidad detectada)" if p_reset < 0.05 else "CUMPLIDO"
        print(f"  Test RESET (Ramsey) : F = {f_reset:.4f}  |  p = {p_reset:.4e}  →  {conclusion}")
    except Exception as exc:
        f_reset, p_reset = float("nan"), float("nan")
        conclusion = f"Error: {exc}"
        print(f"  Test RESET: {conclusion}")

    _grafica_residuos_vs_ajustados(ajust_sub, resid_sub, nombre)

    return {
        "test": "RESET (Ramsey)",
        "estadistico_F": f_reset,
        "p_valor": p_reset,
        "H0": "El modelo lineal está bien especificado",
        "conclusion": conclusion,
        "nota": "p < 0.05 → existe estructura no lineal no capturada",
    }


# ── Supuesto 2: Independencia ──────────────────────────────────────────────────

def _sup_independencia(residuos):
    print("\n── Supuesto 2: Independencia (Durbin-Watson) ──────────────")

    dw = float(durbin_watson(residuos))

    if dw < 1.5:
        conclusion = "VIOLADO (autocorrelación positiva)"
    elif dw > 2.5:
        conclusion = "VIOLADO (autocorrelación negativa)"
    else:
        conclusion = "CUMPLIDO"

    print(f"  DW = {dw:.4f}  →  {conclusion}")
    print("  (referencia: DW ≈ 2.0 = sin autocorrelación; zona aceptable 1.5–2.5)")
    print("  Nota: datos espaciales (x_km) pueden presentar autocorrelación estructural.")

    return {
        "test": "Durbin-Watson",
        "estadistico_DW": dw,
        "p_valor": "N/A",
        "H0": "Errores independientes (sin autocorrelación)",
        "conclusion": conclusion,
        "nota": "DW ≈ 2: independencia; <1.5: autocorr. positiva; >2.5: negativa",
    }


# ── Supuesto 3: Homocedasticidad ──────────────────────────────────────────────

def _sup_homocedasticidad(resid_sub, X_sub, ajust_sub, nombre):
    print("\n── Supuesto 3: Homocedasticidad (Breusch-Pagan) ──────────")

    X_sm = sm.add_constant(X_sub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(resid_sub, X_sm)

    conclusion = "VIOLADO (heterocedasticidad)" if lm_p < 0.05 else "CUMPLIDO (homocedasticidad)"
    print(f"  Breusch-Pagan : LM = {lm_stat:.4f}  |  p = {lm_p:.4e}  →  {conclusion}")

    _grafica_scale_location(ajust_sub, resid_sub, nombre)

    return {
        "test": "Breusch-Pagan",
        "estadistico_LM": lm_stat,
        "estadistico_F": f_stat,
        "p_valor_LM": lm_p,
        "p_valor_F": f_p,
        "H0": "Varianza de los errores es constante (homocedasticidad)",
        "conclusion": conclusion,
        "nota": "p < 0.05 → varianza no constante (heterocedasticidad)",
    }


# ── Supuesto 4: Normalidad ────────────────────────────────────────────────────

def _sup_normalidad(residuos, nombre, seed):
    print("\n── Supuesto 4: Normalidad de residuos ─────────────────────")

    n = len(residuos)

    # Anderson-Darling (sin límite de n, más sensible en las colas)
    ad         = anderson(residuos, dist="norm")
    ad_stat    = ad.statistic
    ad_critico = ad.critical_values[2]   # nivel 5%
    ad_violado = ad_stat > ad_critico
    conc_ad    = "VIOLADO" if ad_violado else "CUMPLIDO"
    print(f"  Anderson-Darling (n={n:,}): A² = {ad_stat:.4f}  "
          f"(crítico 5% = {ad_critico:.4f})  →  {conc_ad}")

    # Kolmogorov-Smirnov contra N(0, 1) sobre residuos estandarizados
    resid_z = (residuos - residuos.mean()) / residuos.std()
    ks_stat, ks_p = kstest(resid_z, "norm")
    conc_ks = "VIOLADO" if ks_p < 0.05 else "CUMPLIDO"
    print(f"  Kolmogorov-Smirnov (n={n:,}): D = {ks_stat:.6f}  |  p = {ks_p:.4e}  →  {conc_ks}")
    print("  Nota: con n grande ambos tests son muy sensibles a desviaciones leves.")

    _grafica_qq(residuos, nombre, seed)
    _grafica_histograma_residuos(residuos, nombre)

    return {
        "anderson_darling": {
            "test": "Anderson-Darling",
            "n_muestra": n,
            "estadistico_A2": ad_stat,
            "critico_5pct": ad_critico,
            "H0": "Residuos normalmente distribuidos",
            "conclusion": conc_ad,
        },
        "ks": {
            "test": "Kolmogorov-Smirnov",
            "n_muestra": n,
            "estadistico_D": ks_stat,
            "p_valor": ks_p,
            "H0": "Residuos normalmente distribuidos",
            "conclusion": conc_ks,
        },
    }


# ── Supuesto 5: Multicolinealidad ────────────────────────────────────────────

def _sup_multicolinealidad(X_sub, feature_names):
    print("\n── Supuesto 5: Multicolinealidad (VIF + Cond. Number) ──────")

    # VIF — sólo depende de X; submuestra es suficiente para estimar corr.
    X_sm      = sm.add_constant(X_sub)
    vif_vals  = [variance_inflation_factor(X_sm, i + 1) for i in range(len(feature_names))]
    vif_df    = pd.DataFrame({"variable": feature_names, "VIF": vif_vals})
    vif_df    = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    for _, row in vif_df.iterrows():
        v = row["VIF"]
        flag = "  ⚠ SEVERO (>10)" if v > 10 else ("  ↑ moderado (5-10)" if v > 5 else "")
        print(f"  VIF  {row['variable']:<25} : {v:6.2f}{flag}")

    # Número de condición
    cond_num = float(np.linalg.cond(X_sub))
    if cond_num > 1000:
        conc_cond = "VIOLADO (condición > 1000, multicolinealidad severa)"
    elif cond_num > 30:
        conc_cond = "ADVERTENCIA (condición 30–1000, multicolinealidad moderada)"
    else:
        conc_cond = "CUMPLIDO"
    print(f"\n  Número de condición : {cond_num:.2f}  →  {conc_cond}")

    conc_vif = "VIOLADO (VIF > 10)" if vif_df["VIF"].max() > 10 else (
        "ADVERTENCIA (VIF 5–10)" if vif_df["VIF"].max() > 5 else "CUMPLIDO"
    )

    _grafica_vif(vif_df)
    _grafica_correlacion(X_sub, feature_names)

    return {
        "vif_df": vif_df,
        "condicion": cond_num,
        "conclusion_vif": conc_vif,
        "conclusion_cond": conc_cond,
    }


# ── Supuesto 6: Outliers e influencia ────────────────────────────────────────

def _sup_outliers(X_sub, y_sub, nombre, n_total):
    print("\n── Supuesto 6: Outliers e influencia ──────────────────────")

    n = len(y_sub)
    p = X_sub.shape[1]

    X_sm    = sm.add_constant(X_sub)
    ols_res = sm.OLS(y_sub, X_sm).fit()
    infl    = OLSInfluence(ols_res)

    cook_d      = infl.cooks_distance[0]
    leverage    = infl.hat_matrix_diag
    student_res = infl.resid_studentized_internal

    umbral_cook = 4.0 / n
    umbral_lev  = 2.0 * (p + 1) / n
    umbral_stud = 3.0

    n_cook  = int(np.sum(cook_d > umbral_cook))
    n_lev   = int(np.sum(leverage > umbral_lev))
    n_stud  = int(np.sum(np.abs(student_res) > umbral_stud))

    print(f"  Distancia de Cook > 4/n     ({umbral_cook:.5f}): "
          f"{n_cook:,} pts ({100 * n_cook / n:.1f} %)")
    print(f"  Leverage > 2(p+1)/n         ({umbral_lev:.5f}): "
          f"{n_lev:,} pts ({100 * n_lev / n:.1f} %)")
    print(f"  |Resid. studentizado| > 3               : "
          f"{n_stud:,} pts ({100 * n_stud / n:.1f} %)")

    _grafica_cook(cook_d, umbral_cook, nombre)
    _grafica_leverage_vs_residuos(leverage, student_res, umbral_lev, nombre)

    return {
        "n_submuestra": n,
        "n_total": n_total,
        "umbral_cook": umbral_cook,
        "umbral_leverage": umbral_lev,
        "umbral_studentizado": umbral_stud,
        "n_outliers_cook": n_cook,
        "n_outliers_leverage": n_lev,
        "n_outliers_studentizado": n_stud,
        "pct_cook": round(100 * n_cook / n, 2),
        "pct_leverage": round(100 * n_lev / n, 2),
        "pct_studentizado": round(100 * n_stud / n, 2),
    }


# ---------------------------------------------------------------------------
# Gráficas — supuestos
# ---------------------------------------------------------------------------

def _grafica_residuos_vs_ajustados(ajustados, residuos, nombre):
    from scipy.stats import binned_statistic
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ajustados, residuos, alpha=0.3, s=6, color=_C_AZUL, edgecolors="none")
    ax.axhline(0, color=_C_ROJO, linewidth=1.8, linestyle="--")

    bin_med, bin_edges, _ = binned_statistic(ajustados, residuos, statistic="median", bins=50)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(bin_centers, bin_med, color=_C_NARANJA, linewidth=2.0, label="Mediana por bin")

    ax.set_xlabel("Valores ajustados (log1p DBO)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Residuos (log1p DBO)", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Residuos vs Ajustados — {nombre}\n"
        "(patrón curvo → no-linealidad; dispersión creciente → heterocedasticidad)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup1_linealidad_residuos_ajustados.png")


def _grafica_scale_location(ajustados, residuos, nombre):
    from scipy.stats import binned_statistic
    sqrt_abs = np.sqrt(np.abs(residuos))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ajustados, sqrt_abs, alpha=0.3, s=6, color=_C_CYAN, edgecolors="none")

    bin_med, bin_edges, _ = binned_statistic(ajustados, sqrt_abs, statistic="median", bins=50)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(bin_centers, bin_med, color=_C_ROJO, linewidth=2.0, label="Mediana por bin")

    ax.set_xlabel("Valores ajustados (log1p DBO)", fontweight="bold", fontsize=11)
    ax.set_ylabel("√ |Residuos|", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Scale-Location — {nombre}\n"
        "(línea horizontal → homocedasticidad; tendencia → heterocedasticidad)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup3_homocedasticidad_scale_location.png")


def _grafica_qq(residuos, nombre, seed):
    rng     = np.random.default_rng(seed)
    n       = len(residuos)
    sub     = rng.choice(residuos, size=min(n, 10_000), replace=False)
    resid_z = (sub - sub.mean()) / sub.std()

    (osm, osr), (slope, intercept, _) = stats.probplot(resid_z, dist="norm", plot=None)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(osm, osr, alpha=0.4, s=6, color=_C_AZUL, edgecolors="none", label="Cuantiles muestra")
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, color=_C_ROJO, linewidth=2.0, label="Línea normal teórica")
    ax.set_xlabel("Cuantiles teóricos N(0,1)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Cuantiles de la muestra", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Q-Q Plot de Residuos — {nombre}\n"
        "(puntos sobre la línea → normalidad; colas desviadas → no-normalidad)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup4_normalidad_qq.png")


def _grafica_histograma_residuos(residuos, nombre):
    resid_z = (residuos - residuos.mean()) / residuos.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid_z, bins=80, density=True, alpha=0.7,
            color=_C_AZUL, edgecolor="white", linewidth=0.3, label="Residuos estandarizados")

    x_n = np.linspace(resid_z.min(), resid_z.max(), 300)
    ax.plot(x_n, norm.pdf(x_n), color=_C_ROJO, linewidth=2.0, label="N(0, 1)")

    ax.set_xlabel("Residuos estandarizados", fontweight="bold", fontsize=11)
    ax.set_ylabel("Densidad", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Distribución de Residuos — {nombre}\n"
        "(histograma ≈ campana → normalidad)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup4_normalidad_histograma.png")


def _grafica_vif(vif_df):
    df_plot = vif_df.sort_values("VIF", ascending=True)
    colores = [
        _C_ROJO if v > 10 else (_C_NARANJA if v > 5 else _C_AZUL)
        for v in df_plot["VIF"]
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_plot["variable"], df_plot["VIF"], color=colores, edgecolor="white", height=0.65)
    ax.axvline(5,  color=_C_NARANJA, linewidth=1.5, linestyle="--", label="VIF = 5 (moderado)")
    ax.axvline(10, color=_C_ROJO,    linewidth=1.5, linestyle="--", label="VIF = 10 (severo)")
    ax.set_xlabel("Factor de Inflación de la Varianza (VIF)", fontweight="bold", fontsize=11)
    ax.set_title(
        "Multicolinealidad — VIF por variable\n"
        "(> 10: problemático, 5–10: moderado, < 5: aceptable)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_axisbelow(True)
    _estilo_grid(ax, axis="x")
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup5_multicolinealidad_vif.png")


def _grafica_correlacion(X_sub, feature_names):
    corr = pd.DataFrame(X_sub, columns=feature_names).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "Correlación entre variables predictoras\n"
        "(|r| > 0.8 puede indicar multicolinealidad)",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup5_multicolinealidad_correlacion.png")


def _grafica_cook(cook_d, umbral, nombre):
    n         = len(cook_d)
    idx_influy = np.where(cook_d > umbral)[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.vlines(range(n), 0, cook_d, color="#CCCCCC", linewidth=0.3, alpha=0.6)
    if len(idx_influy):
        ax.vlines(idx_influy, 0, cook_d[idx_influy], color=_C_ROJO, linewidth=0.8)
    ax.axhline(umbral, color=_C_ROJO, linewidth=1.5, linestyle="--",
               label=f"Umbral 4/n = {umbral:.5f}")
    ax.set_xlabel("Índice de observación (submuestra)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Distancia de Cook", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Distancia de Cook — {nombre}\n"
        f"(sobre umbral: {len(idx_influy):,} / {n:,} puntos influyentes)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup6_outliers_cook.png")


def _grafica_leverage_vs_residuos(leverage, student_res, umbral_lev, nombre):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(leverage, student_res, alpha=0.4, s=8, color=_C_AZUL, edgecolors="none")
    ax.axvline(umbral_lev, color=_C_NARANJA, linewidth=1.5, linestyle="--",
               label=f"Leverage = 2(p+1)/n = {umbral_lev:.4f}")
    ax.axhline( 3, color=_C_ROJO, linewidth=1.5, linestyle="--", label="|Student| = 3")
    ax.axhline(-3, color=_C_ROJO, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Leverage (h_ii)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Residuos Studentizados", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Leverage vs Residuos Studentizados — {nombre}\n"
        "(cuadrante sup. der.: observaciones influyentes y atípicas)",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "sup6_outliers_leverage.png")


# ---------------------------------------------------------------------------
# Gráficas generales (obs vs pred, residuos mg/L, coefs, importancia)
# ---------------------------------------------------------------------------

def _grafica_no_linealidad(train_df: pd.DataFrame, n_puntos: int = 3_000):
    """
    Scatter de 4 features clave vs DBO en mg/L para evidenciar
    la relación no lineal que la regresión lineal no puede capturar.
    """
    features_clave = ["dbo5_bypass", "caudal_bypass", "kaaa", "x_km"]
    sub = train_df.sample(min(n_puntos, len(train_df)), random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feat in zip(axes.flat, features_clave):
        ax.scatter(sub[feat], sub[TARGET], s=4, alpha=0.3,
                   color=_C_AZUL, edgecolors="none")
        ax.set_xlabel(feat, fontweight="bold", fontsize=10)
        ax.set_ylabel("DBO (mg/L)", fontweight="bold", fontsize=10)
        ax.set_title(f"{feat} vs DBO", fontsize=10, fontweight="bold")
        _estilo_grid(ax)

    fig.suptitle(
        "Relación entre predictores clave y DBO\n"
        "(si no es una línea recta, la regresión lineal no puede capturarla)",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    _guardar(FIGS_DIR / "extra_no_linealidad.png")


def _grafica_obs_vs_pred(y_true, y_pred, nombre):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=8, color=_C_AZUL, edgecolors="none")
    lim_max = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color=_C_ROJO, linewidth=1.8,
            linestyle="--", label="1:1")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("DBO simulada QUAL2K (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel(f"DBO predicha {nombre} (mg/L)", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Observado vs Predicho — conjunto de prueba\n{nombre}",
        fontweight="bold", fontsize=12,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "obs_vs_pred.png")


def _grafica_residuos_mg_L(y_true, y_pred, nombre):
    residuos = np.array(y_pred) - np.array(y_true)
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuos, alpha=0.4, s=8, color=_C_AZUL, edgecolors="none")
    ax.axhline(0, color=_C_ROJO, linewidth=1.8, linestyle="--")
    ax.set_xlim(0)
    ax.set_xlabel("DBO predicha (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Residuo  (pred − real)  mg/L", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Residuos en escala original — conjunto de prueba\n{nombre}",
        fontweight="bold", fontsize=12,
    )
    _estilo_grid(ax)
    plt.tight_layout()
    _guardar(FIGS_DIR / "residuos_mg_L.png")


def _grafica_coeficientes(coefs: pd.Series, nombre: str):
    df_plot = coefs.sort_values(ascending=True)
    colores = [_C_ROJO if v < 0 else _C_AZUL for v in df_plot.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_plot.index, df_plot.values, color=colores, edgecolor="white", height=0.65)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel(
        "Coeficiente (variables en escala original, target log1p)",
        fontweight="bold", fontsize=10,
    )
    ax.set_title(
        f"Coeficientes de Regresión — {nombre}\n"
        "(rojo: efecto negativo; azul: efecto positivo)",
        fontweight="bold", fontsize=11,
    )
    ax.set_axisbelow(True)
    _estilo_grid(ax, axis="x")
    plt.tight_layout()
    _guardar(FIGS_DIR / "coeficientes.png")


def _grafica_importancia_permutation(imp_mean: pd.Series, imp_std: pd.Series, nombre: str):
    orden   = imp_mean.sort_values(ascending=True)
    errores = imp_std[orden.index]
    colores = [_C_ROJO if v < 0 else _C_AZUL for v in orden.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        orden.index, orden.values,
        xerr      = errores.values,
        color     = colores,
        edgecolor = "white",
        height    = 0.65,
        capsize   = 3,
        error_kw  = {"elinewidth": 1.2, "ecolor": "#555555"},
    )
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("ΔR²  =  R²_base − R²_permutado", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Importancia de variables — {nombre} (Permutation Importance)\n"
        "(mayor ΔR² → más influyente  ·  negativo → variable irrelevante)",
        fontweight="bold", fontsize=11,
    )
    ax.set_axisbelow(True)
    _estilo_grid(ax, axis="x")
    plt.tight_layout()
    _guardar(FIGS_DIR / "importancia_permutation.png")


# ---------------------------------------------------------------------------
# Helpers de estilo
# ---------------------------------------------------------------------------

def _estilo_grid(ax, axis="both"):
    ax.minorticks_on()
    ax.grid(which="major", axis=axis, linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", axis=axis, linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)


def _guardar(path: Path):
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[{_NOMBRE_CORTO}] Figura guardada: {path}")


# ---------------------------------------------------------------------------
# Exportar Excel
# ---------------------------------------------------------------------------

def _exportar_excel(
    metricas_train:  dict,
    metricas_test:   dict,
    y_test:          np.ndarray,
    y_pred:          np.ndarray,
    coefs:           pd.Series,
    imp_mean:        pd.Series,
    imp_std:         pd.Series,
    resultados_sup:  dict,
    config:          dict,
):
    excel_path = OUTPUT_DIR / "resultados_lr.xlsx"

    # ── Hoja 1: métricas ──────────────────────────────────────────────────────
    df_metricas = pd.DataFrame([
        {"conjunto": "Train", **metricas_train},
        {"conjunto": "Test",  **metricas_test},
    ])[["conjunto", "r2", "rmse", "mae", "bias"]]
    df_metricas.columns = ["Conjunto", "R²", "RMSE (mg/L)", "MAE (mg/L)", "Sesgo (mg/L)"]

    # ── Hoja 2: predicciones ──────────────────────────────────────────────────
    df_pred = pd.DataFrame({
        "dbo_real_mg_L":     y_test,
        "dbo_predicha_mg_L": y_pred,
        "residuo_mg_L":      y_pred - y_test,
    })

    # ── Hoja 3: supuestos — resumen ───────────────────────────────────────────
    filas_sup = []
    s = resultados_sup

    filas_sup.append({
        "supuesto": "1 — Linealidad",
        "test": s["linealidad"]["test"],
        "estadístico": f"F = {s['linealidad']['estadistico_F']:.4f}",
        "p_valor": s["linealidad"]["p_valor"],
        "H0": s["linealidad"]["H0"],
        "conclusion": s["linealidad"]["conclusion"],
        "nota": s["linealidad"]["nota"],
    })
    filas_sup.append({
        "supuesto": "2 — Independencia",
        "test": s["independencia"]["test"],
        "estadístico": f"DW = {s['independencia']['estadistico_DW']:.4f}",
        "p_valor": s["independencia"]["p_valor"],
        "H0": s["independencia"]["H0"],
        "conclusion": s["independencia"]["conclusion"],
        "nota": s["independencia"]["nota"],
    })
    filas_sup.append({
        "supuesto": "3 — Homocedasticidad",
        "test": s["homocedasticidad"]["test"],
        "estadístico": f"LM = {s['homocedasticidad']['estadistico_LM']:.4f}",
        "p_valor": s["homocedasticidad"]["p_valor_LM"],
        "H0": s["homocedasticidad"]["H0"],
        "conclusion": s["homocedasticidad"]["conclusion"],
        "nota": s["homocedasticidad"]["nota"],
    })
    ad = s["normalidad"]["anderson_darling"]
    filas_sup.append({
        "supuesto": "4a — Normalidad (AD)",
        "test": ad["test"],
        "estadístico": f"A² = {ad['estadistico_A2']:.4f}",
        "p_valor": f"crítico 5% = {ad['critico_5pct']:.4f}",
        "H0": ad["H0"],
        "conclusion": ad["conclusion"],
        "nota": f"n = {ad['n_muestra']:,}",
    })
    ks = s["normalidad"]["ks"]
    filas_sup.append({
        "supuesto": "4b — Normalidad (KS)",
        "test": ks["test"],
        "estadístico": f"D = {ks['estadistico_D']:.6f}",
        "p_valor": ks["p_valor"],
        "H0": ks["H0"],
        "conclusion": ks["conclusion"],
        "nota": f"n = {ks['n_muestra']:,}",
    })
    filas_sup.append({
        "supuesto": "5 — Multicolinealidad (VIF)",
        "test": "VIF + Cond. Number",
        "estadístico": f"VIF_max = {s['multicolinealidad']['vif_df']['VIF'].max():.2f}",
        "p_valor": "N/A",
        "H0": "Sin multicolinealidad (VIF < 10)",
        "conclusion": s["multicolinealidad"]["conclusion_vif"],
        "nota": f"Cond. num = {s['multicolinealidad']['condicion']:.2f}  →  {s['multicolinealidad']['conclusion_cond']}",
    })
    filas_sup.append({
        "supuesto": "6 — Outliers/influencia",
        "test": "Cook + Leverage + Student",
        "estadístico": f"Cook>{s['outliers']['umbral_cook']:.5f}: {s['outliers']['pct_cook']:.1f}%",
        "p_valor": "N/A",
        "H0": "Sin puntos influyentes",
        "conclusion": "VER DETALLE",
        "nota": (f"Leverage>umbral: {s['outliers']['pct_leverage']:.1f}%  |  "
                 f"|Student|>3: {s['outliers']['pct_studentizado']:.1f}%"),
    })

    df_sup = pd.DataFrame(filas_sup)

    # ── Hoja 4: VIF detallado ─────────────────────────────────────────────────
    df_vif = s["multicolinealidad"]["vif_df"].copy()
    df_vif["interpretacion"] = df_vif["VIF"].apply(
        lambda v: "Severo (>10)" if v > 10 else ("Moderado (5-10)" if v > 5 else "Aceptable (<5)")
    )

    # ── Hoja 5: coeficientes ──────────────────────────────────────────────────
    df_coefs = pd.DataFrame({
        "variable":     coefs.index,
        "coeficiente":  coefs.values,
        "abs_coef":     np.abs(coefs.values),
    }).sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

    # ── Hoja 6: importancia ───────────────────────────────────────────────────
    df_imp = pd.DataFrame({
        "variable":         imp_mean.index,
        "importancia_mean": imp_mean.values,
        "importancia_std":  imp_std.values,
    }).sort_values("importancia_mean", ascending=False)

    # ── Hoja 7: configuración ─────────────────────────────────────────────────
    df_config = pd.DataFrame([{"parametro": k, "valor": v} for k, v in config.items()])

    # ── Escribir ──────────────────────────────────────────────────────────────
    sheets = [
        ("metricas",        df_metricas),
        ("predicciones",    df_pred),
        ("supuestos",       df_sup),
        ("VIF",             df_vif),
        ("coeficientes",    df_coefs),
        ("importancia",     df_imp),
        ("configuracion",   df_config),
    ]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for col_idx, col_name in enumerate(df.columns, start=1):
                ws.column_dimensions[
                    ws.cell(row=1, column=col_idx).column_letter
                ].width = max(len(str(col_name)) + 2, 16)

    print(f"[{_NOMBRE_CORTO}] Excel guardado: {excel_path}")


if __name__ == "__main__":
    entrenar(
        tipo_modelo = "ols",   # "ols", "ridge" o "lasso"
        alpha       = 1.0,
        cv_alpha    = 5,
        test_size   = 0.20,
        seed        = 42,
        n_perm      = 30,
        n_supuestos = 10_000,
    )
