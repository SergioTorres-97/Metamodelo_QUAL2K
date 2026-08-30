"""
lr_diagnostico.py
=================
Script simple para entender por qué la regresión lineal no funciona
bien como metamodelo de DBO.

Valida los 6 supuestos clásicos y muestra dónde falla.

Uso:
    python scripts/lr_diagnostico.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import anderson, kstest

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from metamodelo.datos import cargar_datos, FEATURES, TARGET

FIGS_DIR = _ROOT / "resultados" / "chicamocha_t1_metamodelo" / "figuras_lr_diagnostico"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Cargar datos y ajustar OLS
# =============================================================================

print("Cargando datos...")
df = cargar_datos()

# Split por sim_id (igual que los demás modelos)
sim_ids = df["sim_id"].unique()
ids_train, ids_test = train_test_split(sim_ids, test_size=0.20, random_state=42)
train_df = df[df["sim_id"].isin(ids_train)]
test_df  = df[df["sim_id"].isin(ids_test)]

# Features en magnitudes originales (sin escalar)
X_train = train_df[FEATURES].values
X_test  = test_df[FEATURES].values

y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values

# Ajustar OLS
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones y residuos en mg/L
y_pred_train = modelo.predict(X_train)
residuos     = y_train - y_pred_train

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test,  modelo.predict(X_test))

print(f"\nR² train: {r2_train:.4f}")
print(f"R² test : {r2_test:.4f}")


# =============================================================================
# Supuesto 1 — Linealidad
# Test: RESET (Ramsey). Detecta si falta estructura no lineal en el modelo.
# =============================================================================

def sup_linealidad(X_train, y_train, y_pred_train, residuos):
    print("\n" + "="*55)
    print("SUPUESTO 1 — LINEALIDAD")
    print("="*55)

    X_sm    = sm.add_constant(X_train)
    ols_res = sm.OLS(y_train, X_sm).fit()
    reset   = linear_reset(ols_res, power=3, use_f=True)

    print(f"  Test RESET: F = {reset.statistic:.2f}, p = {reset.pvalue:.2e}")
    if reset.pvalue < 0.05:
        print("  → VIOLADO: el modelo lineal deja estructura sin capturar.")
        print("    Los predictores tienen relaciones NO LINEALES con DBO.")
    else:
        print("  → Cumplido.")

    # Gráfica: residuos vs ajustados
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred_train, residuos, s=5, alpha=0.3, color="#3C5488")
    ax.axhline(0, color="#E64B35", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Valores ajustados (mg/L)")
    ax.set_ylabel("Residuos")
    ax.set_title("Supuesto 1 — Linealidad: Residuos vs Ajustados\n"
                 "(patrón curvo o en abanico → no linealidad)")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "sup1_linealidad.png", dpi=200)
    plt.close()
    print(f"  Figura: {FIGS_DIR / 'sup1_linealidad.png'}")


# =============================================================================
# Supuesto 2 — Independencia
# Test: Durbin-Watson. Detecta autocorrelación en los residuos.
# =============================================================================

def sup_independencia(residuos):
    print("\n" + "="*55)
    print("SUPUESTO 2 — INDEPENDENCIA (Durbin-Watson)")
    print("="*55)

    dw = durbin_watson(residuos)
    print(f"  DW = {dw:.4f}  (rango aceptable: 1.5 – 2.5, ideal ≈ 2.0)")
    if dw < 1.5:
        print("  → VIOLADO: autocorrelación positiva.")
        print("    Filas consecutivas del mismo río tienen DBO similar (estructura espacial).")
    elif dw > 2.5:
        print("  → VIOLADO: autocorrelación negativa.")
    else:
        print("  → Cumplido.")


# =============================================================================
# Supuesto 3 — Homocedasticidad
# Test: Breusch-Pagan. La varianza del error debe ser constante.
# =============================================================================

def sup_homocedasticidad(X_train, residuos, y_pred_train):
    print("\n" + "="*55)
    print("SUPUESTO 3 — HOMOCEDASTICIDAD (Breusch-Pagan)")
    print("="*55)

    X_sm = sm.add_constant(X_train)
    lm, p, _, _ = het_breuschpagan(residuos, X_sm)
    print(f"  LM = {lm:.2f}, p = {p:.2e}")
    if p < 0.05:
        print("  → VIOLADO: la varianza del error NO es constante.")
        print("    El modelo comete errores más grandes en ciertos rangos de DBO.")
    else:
        print("  → Cumplido.")

    # Gráfica: scale-location
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred_train, np.sqrt(np.abs(residuos)), s=5, alpha=0.3, color="#4DBBD5")
    ax.set_xlabel("Valores ajustados (mg/L)")
    ax.set_ylabel("√|Residuos|")
    ax.set_title("Supuesto 3 — Homocedasticidad: Scale-Location\n"
                 "(línea plana = varianza constante)")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "sup3_homocedasticidad.png", dpi=200)
    plt.close()
    print(f"  Figura: {FIGS_DIR / 'sup3_homocedasticidad.png'}")


# =============================================================================
# Supuesto 4 — Normalidad de residuos
# Tests: Shapiro-Wilk y Kolmogorov-Smirnov. Q-Q plot.
# =============================================================================

def sup_normalidad(residuos):
    print("\n" + "="*55)
    print("SUPUESTO 4 — NORMALIDAD DE RESIDUOS")
    print("="*55)

    resid_z = (residuos - residuos.mean()) / residuos.std()

    # Anderson-Darling (sin límite de n, más sensible en las colas)
    ad = anderson(residuos, dist="norm")
    # scipy devuelve estadístico y valores críticos al 15%, 10%, 5%, 2.5%, 1%
    # comparamos contra el nivel 5% (índice 2)
    ad_stat   = ad.statistic
    ad_critico = ad.critical_values[2]   # nivel 5%
    ad_violado = ad_stat > ad_critico
    print(f"  Anderson-Darling (n={len(residuos):,}): A² = {ad_stat:.4f}  "
          f"(crítico 5% = {ad_critico:.4f})  →  {'VIOLADO' if ad_violado else 'Cumplido'}")

    # Kolmogorov-Smirnov
    D, p_ks = kstest(resid_z, "norm")
    print(f"  Kolmogorov-Smirnov  (n={len(residuos):,}): D = {D:.6f}, p = {p_ks:.2e}")

    if ad_violado or p_ks < 0.05:
        print("  → VIOLADO: los residuos no siguen distribución normal.")
        print("    Nota: con n grande ambos tests son muy sensibles a desviaciones leves.")
    else:
        print("  → Cumplido.")

    # Q-Q plot (muestra 10000 puntos para que sea legible)
    sub_z = resid_z[:10_000]
    fig, ax = plt.subplots(figsize=(5, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(sub_z, dist="norm", plot=None)
    ax.scatter(osm, osr, s=5, alpha=0.4, color="#3C5488")
    ax.plot([osm.min(), osm.max()],
            [slope * osm.min() + intercept, slope * osm.max() + intercept],
            color="#E64B35", linewidth=2)
    ax.set_xlabel("Cuantiles teóricos N(0,1)")
    ax.set_ylabel("Cuantiles de la muestra")
    ax.set_title("Supuesto 4 — Normalidad: Q-Q Plot\n"
                 "(puntos sobre la línea = normalidad)")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "sup4_normalidad_qq.png", dpi=200)
    plt.close()
    print(f"  Figura: {FIGS_DIR / 'sup4_normalidad_qq.png'}")


# =============================================================================
# Supuesto 5 — Multicolinealidad
# Métrica: VIF. VIF > 10 es problemático.
# =============================================================================

def sup_multicolinealidad(X_train):
    print("\n" + "="*55)
    print("SUPUESTO 5 — MULTICOLINEALIDAD (VIF)")
    print("="*55)

    X_sm  = sm.add_constant(X_train)
    vifs  = [variance_inflation_factor(X_sm, i + 1) for i in range(len(FEATURES))]
    df_vif = pd.DataFrame({"variable": FEATURES, "VIF": vifs})
    df_vif = df_vif.sort_values("VIF", ascending=False)

    for _, row in df_vif.iterrows():
        flag = "  ⚠ ALTO" if row["VIF"] > 10 else ("  ↑ mod" if row["VIF"] > 5 else "")
        print(f"  {row['variable']:<25} VIF = {row['VIF']:6.2f}{flag}")

    if df_vif["VIF"].max() > 10:
        print("  → VIOLADO: hay variables con multicolinealidad severa.")
    elif df_vif["VIF"].max() > 5:
        print("  → ADVERTENCIA: multicolinealidad moderada.")
    else:
        print("  → Cumplido.")


# =============================================================================
# Supuesto 6 — Outliers e influencia
# Métricas: distancia de Cook, leverage, residuos studentizados.
# =============================================================================

def sup_outliers(X_train, y_train):
    print("\n" + "="*55)
    print("SUPUESTO 6 — OUTLIERS E INFLUENCIA")
    print("="*55)

    from statsmodels.stats.outliers_influence import OLSInfluence
    X_sm    = sm.add_constant(X_train)
    ols_res = sm.OLS(y_train, X_sm).fit()
    infl    = OLSInfluence(ols_res)

    n, p        = X_train.shape
    cook        = infl.cooks_distance[0]
    leverage    = infl.hat_matrix_diag
    studentized = infl.resid_studentized_internal

    umbral_cook = 4 / n
    umbral_lev  = 2 * (p + 1) / n

    pct_cook = 100 * np.mean(cook > umbral_cook)
    pct_lev  = 100 * np.mean(leverage > umbral_lev)
    pct_stud = 100 * np.mean(np.abs(studentized) > 3)

    print(f"  Cook > 4/n         : {pct_cook:.1f}% de los puntos son influyentes")
    print(f"  Leverage > 2(p+1)/n: {pct_lev:.1f}% de los puntos tienen alto leverage")
    print(f"  |Student| > 3      : {pct_stud:.1f}% son outliers")

    if pct_cook > 5:
        print("  → Hay una proporción relevante de puntos influyentes.")
    else:
        print("  → Cumplido (pocos puntos influyentes).")


# =============================================================================
# EXTRA — Por qué falla: relación no lineal entre features y DBO
# =============================================================================

def grafica_no_linealidad(train_df):
    print("\n" + "="*55)
    print("DIAGNÓSTICO EXTRA — No linealidad visible")
    print("="*55)

    # Muestra 4 features clave vs DBO original
    features_clave = ["dbo5_bypass", "caudal_bypass", "kaaa", "x_km"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, feat in zip(axes.flat, features_clave):
        sub = train_df.sample(3000, random_state=42)
        ax.scatter(sub[feat], sub[TARGET], s=4, alpha=0.3, color="#3C5488")
        ax.set_xlabel(feat)
        ax.set_ylabel("DBO (mg/L)")
        ax.set_title(f"{feat} vs DBO")

    fig.suptitle("Relación entre predictores y DBO\n"
                 "(si no es una línea recta, la regresión lineal no puede capturarla)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "extra_no_linealidad.png", dpi=200)
    plt.close()
    print(f"  Figura: {FIGS_DIR / 'extra_no_linealidad.png'}")
    print("  → Si ves curvas, nubes, o patrones en zigzag: la relación es no lineal.")


# =============================================================================
# Ejecutar todo
# =============================================================================

sup_linealidad(X_train, y_train, y_pred_train, residuos)
sup_independencia(residuos)
sup_homocedasticidad(X_train, residuos, y_pred_train)
sup_normalidad(residuos)
sup_multicolinealidad(X_train)
sup_outliers(X_train, y_train)
grafica_no_linealidad(train_df)

print("\n" + "="*55)
print("CONCLUSIÓN")
print("="*55)
print(f"  R² = {r2_test:.4f} en escala mg/L.")
print("  La regresión lineal falla porque la relación entre")
print("  los inputs del QUAL2K y la DBO es intrínsecamente")
print("  no lineal (ecuaciones diferenciales acopladas).")
print("  Esto se confirma con el test RESET y las gráficas.")
print("="*55)
