"""
catboost_trainer.py
====================
Entrena un metamodelo CatBoost para predecir DBO en función de las
11 variables sensibles + distancia x_km.

Diferencias respecto a LGBM/XGBoost:
  - Ordered boosting: construye cada árbol sobre un subconjunto aleatorio
    de los datos anteriores, reduciendo el sesgo de target leakage.
  - Symmetric trees (oblivious): todos los nodos de un nivel usan el mismo
    split → inferencia muy rápida y regularización implícita.
  - No requiere log1p en el target: el ordered boosting es menos propenso
    al sesgo por cola derecha, pero se mantiene para comparabilidad directa
    con LGBM y XGBoost.

Split de datos:
    Train  70 % — actualiza los pesos
    Val    10 % — early stopping
    Test   20 % — evaluación final honesta

Uso:
    Editar los parámetros en el bloque __main__ al final del archivo y ejecutar:
        python metamodelo/catboost_trainer.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# En Windows, si stdout se redirige a un archivo (p.ej. `> log.txt`), Python usa
# la codificación de la consola (cp1252) en vez de UTF-8, y falla al imprimir
# caracteres como "Δ". Se fuerza UTF-8 para que la ejecución no se caiga a mitad
# de la busqueda/entrenamiento por un simple print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos    import cargar_datos, FEATURES, TARGET
from metamodelo.metricas import calcular

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

OUTPUT_DIR  = _ROOT / "resultados" / "t_rio_jordan_metamodelo"
MODELO_PATH = OUTPUT_DIR / "catboost_dbo.joblib"
FIGS_DIR    = OUTPUT_DIR / "figuras_catboost"

# Tope de iteraciones durante la busqueda de hiperparametros; el early stopping
# decide cuantas se usan realmente en cada trial (igual que en el entrenamiento final).
_ITERATIONS_BUSQUEDA = 2000


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar(
    iterations:       int   = 1000,
    depth:            int   = 6,       # profundidad árboles simétricos
    learning_rate:    float = 0.05,
    l2_leaf_reg:      float = 3.0,     # regularización L2 en las hojas
    subsample:        float = 0.8,     # fracción de filas por árbol
    rsm:              float = 0.8,     # fracción de features por nivel (colsample)
    min_data_in_leaf: int   = 20,      # mínimo de muestras por hoja
    early_stopping:   int   = 50,
    test_size:        float = 0.20,
    val_size:         float = 0.10,
    seed:             int   = 42,
    n_perm:           int   = 30,
    buscar:           bool  = False,
    n_trials:         int   = 20,
) -> CatBoostRegressor:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  METAMODELO DBO — CatBoost")
    print("=" * 55)

    # ── 1. Split por sim_id ───────────────────────────────────────────────────
    df      = cargar_datos()
    sim_ids = df["sim_id"].unique()

    ids_trainval, ids_test = train_test_split(
        sim_ids, test_size=test_size, random_state=seed
    )
    val_rel = val_size / (1 - test_size)
    ids_train, ids_val = train_test_split(
        ids_trainval, test_size=val_rel, random_state=seed
    )

    train_df = df[df["sim_id"].isin(ids_train)]
    val_df   = df[df["sim_id"].isin(ids_val)]
    test_df  = df[df["sim_id"].isin(ids_test)]

    print(f"\n[datos] Train : {len(train_df):,} filas ({len(ids_train)} sims)")
    print(f"[datos] Val   : {len(val_df):,} filas ({len(ids_val)} sims)")
    print(f"[datos] Test  : {len(test_df):,} filas ({len(ids_test)} sims)\n")

    # CatBoost es invariante a la escala; se usan los valores directos
    X_train = train_df[FEATURES].values
    X_val   = val_df[FEATURES].values
    X_test  = test_df[FEATURES].values

    # Target — log1p para comprimir la cola derecha y comparar en igualdad
    # de condiciones con LGBM y XGBoost
    y_train_orig = train_df[TARGET].values
    y_val_orig   = val_df[TARGET].values
    y_test_orig  = test_df[TARGET].values

    y_train = np.log1p(y_train_orig)
    y_val   = np.log1p(y_val_orig)

    print(f"[datos] Target log1p — rango train: "
          f"[{y_train.min():.3f}, {y_train.max():.3f}]  "
          f"(original: [{y_train_orig.min():.1f}, {y_train_orig.max():.1f}] mg/L)\n")

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    # Busqueda bayesiana con Optuna (opcional) — split unico train/val
    if buscar:
        study = _buscar_optuna(
            X_train, y_train, X_val, y_val,
            n_trials       = n_trials,
            early_stopping = early_stopping,
            seed           = seed,
        )
        params_opt       = study.best_params
        depth            = params_opt["depth"]
        learning_rate    = params_opt["learning_rate"]
        l2_leaf_reg      = params_opt["l2_leaf_reg"]
        subsample        = params_opt["subsample"]
        rsm              = params_opt["rsm"]
        min_data_in_leaf = params_opt["min_data_in_leaf"]
        iterations       = _ITERATIONS_BUSQUEDA
    else:
        study = None

    modelo = CatBoostRegressor(
        iterations        = iterations,
        depth             = depth,
        learning_rate     = learning_rate,
        l2_leaf_reg       = l2_leaf_reg,
        subsample         = subsample,
        rsm               = rsm,
        min_data_in_leaf  = min_data_in_leaf,
        loss_function     = "RMSE",
        eval_metric       = "RMSE",
        early_stopping_rounds = early_stopping,
        random_seed       = seed,
        thread_count      = -1,
        verbose           = 100,
    )

    print(f"[CAT] depth         : {depth}  (árboles simétricos)")
    print(f"[CAT] Regularización: l2_leaf_reg={l2_leaf_reg}  "
          f"min_data_in_leaf={min_data_in_leaf}")
    print(f"[CAT] Early stopping: {early_stopping} rondas sin mejora en val\n")
    print(f"[CAT] Entrenando con {len(X_train):,} filas...")

    modelo.fit(
        X_train, y_train,
        eval_set        = (X_val, y_val),
        use_best_model  = True,
    )

    iteraciones_usadas = modelo.best_iteration_
    print(f"\n[CAT] Mejor iteración : {iteraciones_usadas} árboles  "
          f"(val RMSE = {modelo.best_score_['validation']['RMSE']:.4f})")

    # ── 3. Predicciones — invertir log1p → mg/L ──────────────────────────────
    y_pred_train = np.expm1(modelo.predict(X_train))
    y_pred_test  = np.expm1(modelo.predict(X_test))

    metricas_train = calcular(y_train_orig, y_pred_train, nombre="Train")
    metricas_test  = calcular(y_test_orig,  y_pred_test,  nombre="Test")

    # ── 4. Permutation Importance (en mg/L — escala original) ────────────────
    # permutation_importance ya paraleliza a nivel de repeticiones/features con
    # n_jobs=-1 (un proceso joblib por núcleo). Si cada predict interno de
    # CatBoost además usa todos los hilos (thread_count=-1 por defecto), cada
    # uno de esos procesos sobre-suscribe los núcleos varias veces (visto con
    # LGBM: ~20 min extra por esta causa). A diferencia de LGBM, un modelo
    # CatBoost ya entrenado no permite `set_params` (CatBoostError), así que
    # se fuerza thread_count=1 pasándolo directo en cada llamada a predict.
    def _r2_mg_L(estimator, X, y_orig):
        pred = estimator.predict(X, thread_count=1)
        return r2_score(y_orig, np.expm1(pred))

    print(f"[CAT] Calculando permutation importance ({n_perm} repeticiones)...")
    perm = permutation_importance(
        modelo, X_val, y_val_orig,
        n_repeats    = n_perm,
        random_state = seed,
        scoring      = _r2_mg_L,
        n_jobs       = -1,
    )
    imp_mean = pd.Series(perm.importances_mean, index=FEATURES)
    imp_std  = pd.Series(perm.importances_std,  index=FEATURES)

    print("[CAT] Top-5 features por importancia (ΔR² en mg/L):")
    for feat, val in imp_mean.sort_values(ascending=False).head(5).items():
        print(f"       {feat:<25} Δ R² = {val:+.4f}")

    # ── 5. Gráficas ───────────────────────────────────────────────────────────
    _grafica_curvas(modelo)
    _grafica_obs_vs_pred(y_test_orig, y_pred_test)
    _grafica_residuos(y_test_orig, y_pred_test)
    _grafica_importancia_nativa(modelo)
    _grafica_importancia_permutation(imp_mean, imp_std)

    # ── 6. Excel ──────────────────────────────────────────────────────────────
    _exportar_excel(
        metricas_train    = metricas_train,
        metricas_test     = metricas_test,
        modelo            = modelo,
        y_test            = y_test_orig,
        y_pred            = y_pred_test,
        imp_mean          = imp_mean,
        imp_std           = imp_std,
        study             = study,
        config            = dict(
            iterations=iterations, depth=depth,
            learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
            subsample=subsample, rsm=rsm,
            min_data_in_leaf=min_data_in_leaf,
            iteraciones_usadas=iteraciones_usadas,
            buscar_optuna=buscar, n_trials=n_trials if buscar else 0,
            metodo_busqueda="optuna_split_unico_early_stopping" if buscar else "manual",
            target_transform="log1p",
        ),
    )

    # ── 7. Guardar modelo ─────────────────────────────────────────────────────
    joblib.dump(modelo, MODELO_PATH)
    print(f"[CAT] Modelo guardado  : {MODELO_PATH}")

    return modelo


# ---------------------------------------------------------------------------
# Busqueda bayesiana con Optuna — split unico train/val + early stopping
# ---------------------------------------------------------------------------

def _callback_trazabilidad(log_path: Path):
    """
    Callback de Optuna que registra cada trial apenas termina:
      - imprime una linea en consola (numero, R2, duracion, si es el mejor hasta ahora)
      - agrega una fila al CSV `log_path` (append), para no perder el avance
        si la busqueda se interrumpe o tarda mucho.
    """
    def _callback(study, trial) -> None:
        valor    = trial.value if trial.value is not None else float("nan")
        dur_s    = trial.duration.total_seconds() if trial.duration else float("nan")
        es_mejor = trial.number == study.best_trial.number
        marca    = "  <- mejor hasta ahora" if es_mejor else ""

        print(f"[CAT]   trial {trial.number:>3}/{len(study.trials) - 1}  "
              f"R2_val={valor:.4f}  ({dur_s:6.1f} s){marca}")

        fila = {
            "trial":      trial.number,
            "r2_val":     valor,
            "duracion_s": dur_s,
            "estado":     trial.state.name,
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            **trial.params,
        }
        escribir_encabezado = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fila.keys()))
            if escribir_encabezado:
                writer.writeheader()
            writer.writerow(fila)

    return _callback


def _buscar_optuna(
    X_train:        np.ndarray,
    y_train:        np.ndarray,
    X_val:          np.ndarray,
    y_val:          np.ndarray,
    n_trials:       int,
    early_stopping: int,
    seed:           int,
):
    """
    Optimiza hiperparametros de CatBoost con TPE sobre el mismo split
    train/val (agrupado por sim_id) que usa el entrenamiento final.

    Cada trial usa early stopping nativo contra el conjunto de validacion,
    por lo que el costo por trial queda acotado (a diferencia de una
    validacion cruzada k-fold, que multiplicaria el costo de entrenamiento
    por k folds sin aprovechar el early stopping).

    Devuelve el objeto `study` completo (no solo los mejores parametros)
    para poder exportar el historial de todos los trials.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna o una de sus dependencias no esta instalada. "
            "Instala las dependencias del proyecto antes de usar --buscar."
        ) from exc

    def objective(trial) -> float:
        params = {
            "depth":            trial.suggest_int("depth", 4, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "subsample":        trial.suggest_float("subsample", 0.60, 1.00),
            "rsm":              trial.suggest_float("rsm", 0.60, 1.00),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        }

        estimador = CatBoostRegressor(
            iterations             = _ITERATIONS_BUSQUEDA,
            **params,
            loss_function          = "RMSE",
            eval_metric            = "RMSE",
            early_stopping_rounds  = early_stopping,
            random_seed            = seed,
            thread_count           = -1,
            verbose                = False,
        )
        estimador.fit(
            X_train, y_train,
            eval_set       = (X_val, y_val),
            use_best_model = True,
        )
        return float(r2_score(y_val, estimador.predict(X_val)))

    sampler  = optuna.samplers.TPESampler(seed=seed)
    study    = optuna.create_study(direction="maximize", sampler=sampler)
    log_path = OUTPUT_DIR / "optuna_trials_catboost.csv"
    if log_path.exists():
        log_path.unlink()  # log limpio en cada corrida

    print(f"\n[CAT] Busqueda bayesiana Optuna: {n_trials} trials "
          f"(split unico train/val, early stopping={early_stopping})...")
    print(f"[CAT] Trazabilidad por trial    : {log_path}")
    t0 = time.time()
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False,
        callbacks=[_callback_trazabilidad(log_path)],
    )

    print(f"[CAT] Busqueda completada en {time.time() - t0:.1f} s "
          f"({(time.time() - t0) / n_trials:.1f} s/trial en promedio)")
    print(f"[CAT] Mejor R2 val (log1p): {study.best_value:.4f}  (trial {study.best_trial.number})")
    print("[CAT] Mejores hiperparametros:")
    for k, v in study.best_params.items():
        print(f"       {k:<22} : {v}")
    print()

    return study


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def _grafica_curvas(modelo: CatBoostRegressor):
    evals      = modelo.get_evals_result()
    rmse_train = evals["learn"]["RMSE"]
    rmse_val   = evals["validation"]["RMSE"]
    mejor      = int(modelo.best_iteration_) + 1
    rondas     = range(1, len(rmse_train) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rondas, rmse_train, color="#3C5488", linewidth=1.8, label="Train")
    ax.plot(rondas, rmse_val,   color="#E64B35", linewidth=1.8, label="Validación")
    ax.axvline(mejor, color="#00A087", linewidth=1.5, linestyle="--",
               label=f"Mejor árbol = {mejor}")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Número de árboles", fontweight="bold", fontsize=11)
    ax.set_ylabel("RMSE (escala log1p DBO)", fontweight="bold", fontsize=11)
    ax.set_title("Curvas de aprendizaje — Train vs Validación",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "curvas_aprendizaje.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[CAT] Figura guardada: {path}")


def _grafica_obs_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    lim_max = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="#E64B35", linewidth=1.8,
            linestyle="--", label="1:1")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("DBO simulada QUAL2K (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel("DBO predicha CatBoost (mg/L)", fontweight="bold", fontsize=11)
    ax.set_title("Observado vs Predicho — conjunto de prueba",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "obs_vs_pred.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[CAT] Figura guardada: {path}")


def _grafica_residuos(y_true, y_pred):
    residuos = np.array(y_pred) - np.array(y_true)
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuos, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    ax.axhline(0, color="#E64B35", linewidth=1.8, linestyle="--")
    ax.set_xlim(0)
    ax.set_xlabel("DBO predicha (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Residuo (pred − real)  mg/L", fontweight="bold", fontsize=11)
    ax.set_title("Residuos — conjunto de prueba", fontweight="bold", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "residuos.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[CAT] Figura guardada: {path}")


def _grafica_importancia_nativa(modelo: CatBoostRegressor):
    importancias = pd.Series(
        modelo.get_feature_importance(), index=FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importancias.index, importancias.values,
            color="#3C5488", edgecolor="white", height=0.65)
    ax.set_xlim(0)
    ax.set_xlabel("Importancia (%)", fontweight="bold", fontsize=11)
    ax.set_title("Importancia nativa — CatBoost (PredictionValuesChange)",
                 fontweight="bold", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", axis="x", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", axis="x", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIGS_DIR / "importancia_nativa.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[CAT] Figura guardada: {path}")


def _grafica_importancia_permutation(imp_mean: pd.Series, imp_std: pd.Series):
    orden   = imp_mean.sort_values(ascending=True)
    errores = imp_std[orden.index]
    colores = ["#E64B35" if v < 0 else "#3C5488" for v in orden.values]

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
    ax.set_xlabel("ΔR²  =  R²_base − R²_permutado",
                  fontweight="bold", fontsize=11)
    ax.set_title("Importancia de variables — CatBoost (Permutation Importance)\n"
                 "(mayor ΔR² → más influyente  ·  negativo → feature irrelevante)",
                 fontweight="bold", fontsize=11)
    ax.minorticks_on()
    ax.grid(which="major", axis="x", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", axis="x", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIGS_DIR / "importancia_permutation.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[CAT] Figura guardada: {path}")


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def _exportar_excel(
    metricas_train: dict,
    metricas_test:  dict,
    modelo:         CatBoostRegressor,
    y_test:         np.ndarray,
    y_pred:         np.ndarray,
    imp_mean:       pd.Series,
    imp_std:        pd.Series,
    config:         dict,
    study=None,
):
    excel_path = OUTPUT_DIR / "resultados_catboost.xlsx"

    df_metricas = pd.DataFrame([
        {"conjunto": "Train", **metricas_train},
        {"conjunto": "Test",  **metricas_test},
    ])[["conjunto", "r2", "rmse", "mae", "bias"]]
    df_metricas.columns = ["Conjunto", "R²", "RMSE (mg/L)", "MAE (mg/L)", "Sesgo (mg/L)"]

    df_pred = pd.DataFrame({
        "dbo_real_mg_L":     y_test,
        "dbo_predicha_mg_L": y_pred,
        "residuo_mg_L":      y_pred - y_test,
    })

    evals      = modelo.get_evals_result()
    rmse_train = evals["learn"]["RMSE"]
    rmse_val   = evals["validation"]["RMSE"]
    df_curva = pd.DataFrame({
        "arbol":      range(1, len(rmse_train) + 1),
        "RMSE_train": rmse_train,
        "RMSE_val":   rmse_val,
    })

    df_imp = pd.DataFrame({
        "variable":         imp_mean.index,
        "importancia_mean": imp_mean.values,
        "importancia_std":  imp_std.values,
    }).sort_values("importancia_mean", ascending=False)

    df_config = pd.DataFrame([
        {"parametro": k, "valor": v} for k, v in config.items()
    ])

    hojas = [
        ("metricas",      df_metricas),
        ("predicciones",  df_pred),
        ("curva_perdida", df_curva),
        ("importancia",   df_imp),
        ("configuracion", df_config),
    ]

    # Trazabilidad completa de la busqueda de hiperparametros (un trial por fila)
    if study is not None:
        df_trials = study.trials_dataframe()
        df_trials = df_trials.rename(columns=lambda c: c.replace("params_", ""))
        hojas.append(("busqueda_optuna", df_trials))

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet, df in hojas:
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            for col_idx, col_name in enumerate(df.columns, start=1):
                ws.column_dimensions[
                    ws.cell(row=1, column=col_idx).column_letter
                ].width = max(len(str(col_name)) + 2, 14)

    print(f"[CAT] Excel guardado   : {excel_path}")


if __name__ == "__main__":
    entrenar(
        iterations       = 1000,
        depth            = 6,
        learning_rate    = 0.05,
        l2_leaf_reg      = 3.0,
        subsample        = 0.8,
        rsm              = 0.8,
        min_data_in_leaf = 20,
        early_stopping   = 50,
        test_size        = 0.20,
        val_size         = 0.10,
        seed             = 42,
        n_perm           = 30,
        buscar           = True,
        n_trials         = 20,
    )
