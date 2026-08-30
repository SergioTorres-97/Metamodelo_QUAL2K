"""
lgbm_trainer.py
================
Entrena un metamodelo LightGBM para predecir DBO en función de las
11 variables sensibles + distancia x_km.

Ventajas sobre XGBoost en datasets grandes:
  - Histogram-based: agrupa valores continuos en bins antes de buscar splits
    → mucho más rápido con n > 10 000 filas.
  - Leaf-wise growth (vs level-wise en XGBoost): crece el nodo con mayor
    ganancia, convergiendo más rápido con árboles más profundos.
  - Menor consumo de memoria.

Transformación del target:
    La DBO tiene distribución asimétrica (cola derecha larga).
    Se aplica log1p antes de entrenar y expm1 al predecir, de modo que
    el modelo minimiza RMSE en escala logarítmica → residuos absolutos
    más uniformes a lo largo de todo el rango.

Split de datos:
    Train  70 % — actualiza los pesos
    Val    10 % — early stopping
    Test   20 % — evaluación final honesta

Uso:
    Editar los parámetros en el bloque __main__ al final del archivo y ejecutar:
        python metamodelo/lgbm_trainer.py
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
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import lightgbm as lgb

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos    import cargar_datos, FEATURES, TARGET
from metamodelo.metricas import calcular

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

OUTPUT_DIR  = _ROOT / "resultados" / "t_rio_jordan_metamodelo"
MODELO_PATH = OUTPUT_DIR / "lgbm_dbo.joblib"
FIGS_DIR    = OUTPUT_DIR / "figuras_lgbm"

# Tope de arboles durante la busqueda de hiperparametros; el early stopping
# decide cuantos se usan realmente en cada trial (igual que en el entrenamiento final).
# Subido de 2000 a 6000: en la corrida anterior el entrenamiento final llegó al
# tope de 2000 árboles sin que el early stopping llegara a activarse (val RMSE
# seguía bajando), es decir el modelo no había convergido todavía.
_N_ESTIMATORS_BUSQUEDA = 6000


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar(
    n_estimators:  int   = 800,
    num_leaves:    int   = 63,      # max hojas por árbol (leaf-wise)
    learning_rate: float = 0.05,
    subsample:     float = 0.8,     # fracción de filas por árbol
    colsample:     float = 0.8,     # fracción de features por árbol
    reg_alpha:     float = 0.1,     # L1
    reg_lambda:    float = 1.0,     # L2
    min_child_samples: int = 20,    # mínimo de muestras por hoja
    max_depth:      int   = -1,     # -1 = sin límite (leaf-wise lo controla num_leaves)
    early_stopping: int  = 40,
    test_size:     float = 0.20,
    val_size:      float = 0.10,
    seed:          int   = 42,
    n_perm:        int   = 30,
    buscar:        bool  = False,
    n_trials:      int   = 50,
) -> lgb.LGBMRegressor:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  METAMODELO DBO — LightGBM")
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

    # Features — LightGBM es invariante a la escala; se usan los valores directos
    X_train = train_df[FEATURES].values
    X_val   = val_df[FEATURES].values
    X_test  = test_df[FEATURES].values

    # Target — log1p para comprimir la cola derecha de la DBO
    # Las métricas finales se reportan en mg/L (escala original) tras expm1
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
        params_opt         = study.best_params
        num_leaves        = params_opt["num_leaves"]
        max_depth         = params_opt["max_depth"]
        learning_rate     = params_opt["learning_rate"]
        subsample         = params_opt["subsample"]
        colsample         = params_opt["colsample_bytree"]
        reg_alpha         = params_opt["reg_alpha"]
        reg_lambda        = params_opt["reg_lambda"]
        min_child_samples = params_opt["min_child_samples"]
        n_estimators      = _N_ESTIMATORS_BUSQUEDA
    else:
        study = None

    modelo = lgb.LGBMRegressor(
        n_estimators       = n_estimators,
        num_leaves         = num_leaves,
        max_depth          = max_depth,
        learning_rate      = learning_rate,
        subsample          = subsample,
        colsample_bytree   = colsample,
        reg_alpha          = reg_alpha,
        reg_lambda         = reg_lambda,
        min_child_samples  = min_child_samples,
        metric             = "rmse",
        random_state       = seed,
        n_jobs             = -1,
        verbosity          = -1,
    )

    print(f"[LGB] num_leaves    : {num_leaves}  (leaf-wise growth)")
    print(f"[LGB] Regularización: alpha={reg_alpha}  lambda={reg_lambda}  "
          f"min_child_samples={min_child_samples}")
    print(f"[LGB] Early stopping: {early_stopping} rondas sin mejora en val\n")
    print(f"[LGB] Entrenando con {len(X_train):,} filas...")

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    modelo.fit(
        X_train, y_train,
        eval_set        = [(X_train, y_train), (X_val, y_val)],
        eval_names      = ["train", "val"],
        callbacks       = callbacks,
    )

    arboles_usados = modelo.best_iteration_
    print(f"\n[LGB] Mejor iteración : {arboles_usados} árboles")

    # ── 3. Predicciones — invertir log1p → mg/L ───────────────────────────────
    y_pred_train = np.expm1(modelo.predict(X_train))
    y_pred_test  = np.expm1(modelo.predict(X_test))

    metricas_train = calcular(y_train_orig, y_pred_train, nombre="Train")
    metricas_test  = calcular(y_test_orig,  y_pred_test,  nombre="Test")

    # ── 4. Permutation Importance (en mg/L — escala original) ────────────────
    # El scorer aplica expm1 a las predicciones del modelo (log1p) y compara
    # contra y_val_orig en mg/L → importancia interpretable en la escala física.
    def _r2_mg_L(estimator, X, y_orig):
        return r2_score(y_orig, np.expm1(estimator.predict(X)))

    # permutation_importance ya paraleliza a nivel de repeticiones/features con
    # n_jobs=-1 (un proceso joblib por núcleo). Si el modelo conserva su propio
    # n_jobs=-1, cada uno de esos procesos vuelve a lanzar hilos internos de
    # LightGBM, sobre-suscribiendo los núcleos varias veces y volviendo esta
    # etapa mucho más lenta que el entrenamiento mismo. Se fuerza n_jobs=1 en
    # el modelo mientras dura el cálculo y se restaura después.
    modelo.n_jobs = 1
    print(f"[LGB] Calculando permutation importance ({n_perm} repeticiones)...")
    perm = permutation_importance(
        modelo, X_val, y_val_orig,
        n_repeats    = n_perm,
        random_state = seed,
        scoring      = _r2_mg_L,
        n_jobs       = -1,
    )
    modelo.n_jobs = -1
    imp_mean = pd.Series(perm.importances_mean, index=FEATURES)
    imp_std  = pd.Series(perm.importances_std,  index=FEATURES)

    print("[LGB] Top-5 features por importancia (ΔR² en mg/L):")
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
        metricas_train = metricas_train,
        metricas_test  = metricas_test,
        modelo         = modelo,
        y_test         = y_test_orig,
        y_pred         = y_pred_test,
        imp_mean       = imp_mean,
        imp_std        = imp_std,
        study          = study,
        config         = dict(
            n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, subsample=subsample,
            colsample=colsample, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, min_child_samples=min_child_samples,
            max_depth=max_depth,
            arboles_usados=arboles_usados, buscar_optuna=buscar,
            n_trials=n_trials if buscar else 0,
            metodo_busqueda="optuna_split_unico_early_stopping" if buscar else "manual",
            target_transform="log1p",
        ),
    )

    # ── 7. Guardar modelo ─────────────────────────────────────────────────────
    joblib.dump(modelo, MODELO_PATH)
    print(f"[LGB] Modelo guardado  : {MODELO_PATH}")

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

        print(f"[LGB]   trial {trial.number:>3}/{len(study.trials) - 1}  "
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
    Optimiza hiperparametros de LightGBM con TPE sobre el mismo split
    train/val (agrupado por sim_id) que usa el entrenamiento final.

    Cada trial usa early stopping nativo contra el conjunto de validacion,
    por lo que el costo por trial queda acotado (a diferencia de una
    validacion cruzada k-fold, que multiplicaria el costo de entrenamiento
    por k folds sin aprovechar el early stopping).

    `num_leaves` se explora en un rango moderado (15-63): con leaf-wise
    growth, hojas por encima de ~100 generan arboles mucho mas profundos
    y costosos sin mejora relevante en este dataset (17 features), y eran
    el principal cuello de botella de tiempo en la busqueda anterior.

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
            "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
            "max_depth":         trial.suggest_categorical("max_depth", [-1, 4, 6, 8]),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample":         trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.10, 20.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        estimador = lgb.LGBMRegressor(
            n_estimators = _N_ESTIMATORS_BUSQUEDA,
            **params,
            metric       = "rmse",
            random_state = seed,
            n_jobs       = -1,
            verbosity    = -1,
        )
        estimador.fit(
            X_train, y_train,
            eval_set  = [(X_val, y_val)],
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)],
        )
        return float(r2_score(y_val, estimador.predict(X_val)))

    sampler  = optuna.samplers.TPESampler(seed=seed)
    study    = optuna.create_study(direction="maximize", sampler=sampler)
    log_path = OUTPUT_DIR / "optuna_trials_lgbm.csv"
    if log_path.exists():
        log_path.unlink()  # log limpio en cada corrida

    print(f"\n[LGB] Busqueda bayesiana Optuna: {n_trials} trials "
          f"(split unico train/val, early stopping={early_stopping})...")
    print(f"[LGB] Trazabilidad por trial    : {log_path}")
    t0 = time.time()
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False,
        callbacks=[_callback_trazabilidad(log_path)],
    )

    print(f"[LGB] Busqueda completada en {time.time() - t0:.1f} s "
          f"({(time.time() - t0) / n_trials:.1f} s/trial en promedio)")
    print(f"[LGB] Mejor R2 val (log1p): {study.best_value:.4f}  (trial {study.best_trial.number})")
    print("[LGB] Mejores hiperparametros:")
    for k, v in study.best_params.items():
        print(f"       {k:<22} : {v}")
    print()

    return study


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def _grafica_curvas(modelo: lgb.LGBMRegressor):
    """Curva de aprendizaje RMSE train vs val por ronda de boosting."""
    evals      = modelo.evals_result_
    # La clave coincide con el metric declarado en el constructor ("rmse")
    metric_key = next(iter(evals["train"]))
    rmse_train = evals["train"][metric_key]
    rmse_val   = evals["val"][metric_key]
    mejor      = int(np.argmin(rmse_val)) + 1
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
    print(f"[LGB] Figura guardada: {path}")


def _grafica_obs_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    lim_max = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="#E64B35", linewidth=1.8,
            linestyle="--", label="1:1")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("DBO simulada QUAL2K (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel("DBO predicha LightGBM (mg/L)", fontweight="bold", fontsize=11)
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
    print(f"[LGB] Figura guardada: {path}")


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
    print(f"[LGB] Figura guardada: {path}")


def _grafica_importancia_nativa(modelo: lgb.LGBMRegressor):
    """Importancia nativa LightGBM basada en ganancia de los splits."""
    importancias = pd.Series(
        modelo.feature_importances_, index=FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importancias.index, importancias.values,
            color="#3C5488", edgecolor="white", height=0.65)
    ax.set_xlim(0)
    ax.set_xlabel("Importancia (ganancia)", fontweight="bold", fontsize=11)
    ax.set_title("Importancia nativa — LightGBM (ganancia)",
                 fontweight="bold", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", axis="x", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", axis="x", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIGS_DIR / "importancia_nativa.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[LGB] Figura guardada: {path}")


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
    ax.set_title("Importancia de variables — LightGBM (Permutation Importance)\n"
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
    print(f"[LGB] Figura guardada: {path}")


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def _exportar_excel(
    metricas_train: dict,
    metricas_test:  dict,
    modelo:         lgb.LGBMRegressor,
    y_test:         np.ndarray,
    y_pred:         np.ndarray,
    imp_mean:       pd.Series,
    imp_std:        pd.Series,
    config:         dict,
    study=None,
):
    excel_path = OUTPUT_DIR / "resultados_lgbm.xlsx"

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

    evals      = modelo.evals_result_
    metric_key = next(iter(modelo.evals_result_["train"]))
    df_curva = pd.DataFrame({
        "arbol":      range(1, len(evals["train"][metric_key]) + 1),
        "RMSE_train": evals["train"][metric_key],
        "RMSE_val":   evals["val"][metric_key],
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

    print(f"[LGB] Excel guardado   : {excel_path}")


if __name__ == "__main__":
    entrenar(
        n_estimators      = 800,
        num_leaves        = 63,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample         = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        min_child_samples = 20,
        max_depth         = -1,
        early_stopping    = 40,
        test_size         = 0.20,
        val_size          = 0.10,
        seed              = 42,
        n_perm            = 30,
        buscar            = True,
        n_trials          = 20,
    )
