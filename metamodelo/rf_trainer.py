"""
rf_trainer.py
==============
Entrena un metamodelo Random Forest (o ExtraTrees) para predecir DBO en
función de las 16 variables sensibles + distancia x_km.

Transformación del target:
    La DBO tiene distribución asimétrica (cola derecha larga).
    Se aplica log1p antes de entrenar y expm1 al predecir, de modo que
    el modelo minimiza RMSE en escala logarítmica → residuos absolutos
    más uniformes a lo largo de todo el rango.

Mejoras v2 (sobre la versión anterior):
  1. **Optuna (Bayesian TPE)** reemplaza RandomizedSearchCV → encuentra mejores
     hiperparámetros con ~3× menos evaluaciones.
  2. **OOB + warm_start como "early stopping" de la búsqueda**: cada trial
     crece los árboles en bloques y se detiene apenas el R² OOB deja de
     mejorar, en vez de evaluar con GroupKFold (5 folds × árboles completos),
     que resultaba inviable a esta escala (>3 días sin terminar, ver
     docs/metodologia.md §5.3.1). El OOB no necesita partición adicional:
     cada árbol ya se valida con las filas que no vio en su muestra bootstrap.
  3. **Espacio de búsqueda ampliado**:
       - ccp_alpha  : poda por complejidad-costo → regulariza cada árbol
       - min_samples_split : controla cuándo se parte un nodo
       - max_samples : fracción de filas usadas en cada árbol (bag fraction)
  4. **ExtraTreesRegressor** disponible con `--modelo extra`:
       cortes aleatorios (no óptimos) → mayor diversidad, a veces supera a RF
       en conjuntos grandes.

Split de datos:
    Train  70 % — entrena los árboles
    Val    10 % — permutation importance y métricas intermedias
    Test   20 % — evaluación final honesta

Uso:
    Editar los parámetros en el bloque __main__ al final del archivo y ejecutar:
        python metamodelo/rf_trainer.py
    Para usar ExtraTrees cambiar tipo_modelo = "extra".
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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos    import cargar_datos, FEATURES, TARGET
from metamodelo.metricas import calcular

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

OUTPUT_DIR  = _ROOT / "resultados" / "chicamocha_t1_metamodelo"
MODELO_PATH = OUTPUT_DIR / "rf_dbo.joblib"
FIGS_DIR    = OUTPUT_DIR / "figuras_rf"

# Tipo alias para los dos modelos soportados
_ModeloArbol = RandomForestRegressor | ExtraTreesRegressor


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------

def entrenar(
    tipo_modelo:      str   = "rf",     # "rf" o "extra"
    n_estimators:     int   = 500,
    max_depth:        int   = None,
    min_samples_leaf: int   = 5,
    min_samples_split:int   = 2,
    max_features:     str   = "sqrt",
    max_samples:      float = None,     # None → 1.0 (usa todas las filas)
    ccp_alpha:        float = 0.0,      # poda costo-complejidad
    bootstrap:        bool  = True,
    test_size:        float = 0.20,
    val_size:         float = 0.10,
    seed:             int   = 42,
    n_perm:           int   = 30,
    buscar:           bool  = False,
    n_trials:         int   = 80,       # iteraciones Optuna
) -> _ModeloArbol:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    nombre = "ExtraTrees" if tipo_modelo == "extra" else "Random Forest"
    print("=" * 55)
    print(f"  METAMODELO DBO — {nombre}")
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

    # Features — árboles son invariantes a la escala; se usan los valores directos
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

    # ── 2. Búsqueda bayesiana con Optuna (opcional) ───────────────────────────
    if buscar:
        params_opt = _buscar_optuna(
            X_train, y_train,
            tipo_modelo = tipo_modelo,
            n_trials    = n_trials,
            seed        = seed,
        )
        n_estimators      = params_opt["n_estimators"]
        max_depth         = params_opt["max_depth"]
        min_samples_leaf  = params_opt["min_samples_leaf"]
        min_samples_split = params_opt["min_samples_split"]
        max_features      = params_opt["max_features"]
        max_samples       = params_opt["max_samples"]
        ccp_alpha         = params_opt["ccp_alpha"]

    # ── 3. Curva OOB con warm_start ───────────────────────────────────────────
    if tipo_modelo == "rf":
        print("[RF] Generando curva OOB (warm_start)...")
        oob_scores, ns_oob = _curva_oob_warm(
            X_train, y_train,
            tipo_modelo       = tipo_modelo,
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            min_samples_leaf  = min_samples_leaf,
            min_samples_split = min_samples_split,
            max_features      = max_features,
            max_samples       = max_samples,
            ccp_alpha         = ccp_alpha,
            seed              = seed,
            paso              = max(1, n_estimators // 50),
        )
        n_convergencia = _detectar_convergencia(ns_oob, oob_scores)
        print(f"[RF] OOB converge ≈ {n_convergencia} árboles")
    else:
        oob_scores, ns_oob, n_convergencia = [], [], 0

    # ── 4. Modelo final ───────────────────────────────────────────────────────
    kwargs_comunes = dict(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_leaf  = min_samples_leaf,
        min_samples_split = min_samples_split,
        max_features      = max_features,
        ccp_alpha         = ccp_alpha,
        random_state      = seed,
        n_jobs            = -1,
    )

    if tipo_modelo == "rf":
        modelo = RandomForestRegressor(
            **kwargs_comunes,
            bootstrap   = bootstrap,
            max_samples = max_samples,
            oob_score   = True,
        )
    else:
        modelo = ExtraTreesRegressor(
            **kwargs_comunes,
            bootstrap   = bootstrap,
            max_samples = max_samples if bootstrap else None,
        )

    print(f"\n[{nombre}] n_estimators      : {n_estimators}")
    print(f"[{nombre}] max_depth         : {max_depth if max_depth else 'sin límite'}")
    print(f"[{nombre}] min_samples_leaf  : {min_samples_leaf}")
    print(f"[{nombre}] min_samples_split : {min_samples_split}")
    print(f"[{nombre}] max_features      : {max_features}")
    print(f"[{nombre}] max_samples       : {max_samples if max_samples else '100 %'}")
    print(f"[{nombre}] ccp_alpha         : {ccp_alpha:.6f}")
    print(f"[{nombre}] Entrenando con {len(X_train):,} filas...\n")

    t0 = time.time()
    modelo.fit(X_train, y_train)
    print(f"[{nombre}] Entrenamiento completado en {time.time() - t0:.1f} s")
    if hasattr(modelo, "oob_score_"):
        print(f"[{nombre}] R² OOB (escala log1p): {modelo.oob_score_:.4f}")

    # ── 5. Predicciones — invertir log1p → mg/L ───────────────────────────────
    y_pred_train = np.expm1(modelo.predict(X_train))
    y_pred_val   = np.expm1(modelo.predict(X_val))
    y_pred_test  = np.expm1(modelo.predict(X_test))

    metricas_train = calcular(y_train_orig, y_pred_train, nombre="Train")
    metricas_val   = calcular(y_val_orig,   y_pred_val,   nombre="Val")
    metricas_test  = calcular(y_test_orig,  y_pred_test,  nombre="Test")

    # ── 6. Permutation Importance (en mg/L — escala original) ────────────────
    # El scorer aplica expm1 a las predicciones del modelo (log1p) y compara
    # contra y_val_orig en mg/L → importancia interpretable en la escala física.
    def _r2_mg_L(estimator, X, y_orig):
        return r2_score(y_orig, np.expm1(estimator.predict(X)))

    print(f"\n[{nombre}] Calculando permutation importance ({n_perm} repeticiones)...")
    perm = permutation_importance(
        modelo, X_val, y_val_orig,
        n_repeats    = n_perm,
        random_state = seed,
        scoring      = _r2_mg_L,
        n_jobs       = -1,
    )
    imp_mean = pd.Series(perm.importances_mean, index=FEATURES)
    imp_std  = pd.Series(perm.importances_std,  index=FEATURES)

    print(f"[{nombre}] Top-5 features por importancia (ΔR² en mg/L):")
    for feat, val in imp_mean.sort_values(ascending=False).head(5).items():
        print(f"       {feat:<25} Δ R² = {val:+.4f}")

    # ── 7. Gráficas ───────────────────────────────────────────────────────────
    if tipo_modelo == "rf":
        _grafica_oob(ns_oob, oob_scores, n_convergencia)
    _grafica_obs_vs_pred(y_test_orig, y_pred_test, nombre)
    _grafica_residuos(y_test_orig, y_pred_test, nombre)
    _grafica_importancia_nativa(modelo, nombre)
    _grafica_importancia_permutation(imp_mean, imp_std, nombre)

    # ── 8. Excel ──────────────────────────────────────────────────────────────
    oob_r2 = round(modelo.oob_score_, 4) if hasattr(modelo, "oob_score_") else "N/A"
    _exportar_excel(
        metricas_train    = metricas_train,
        metricas_val      = metricas_val,
        metricas_test     = metricas_test,
        ns_oob            = ns_oob,
        oob_scores        = oob_scores,
        y_test            = y_test_orig,
        y_pred            = y_pred_test,
        imp_mean          = imp_mean,
        imp_std           = imp_std,
        config            = dict(
            tipo_modelo       = nombre,
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            min_samples_leaf  = min_samples_leaf,
            min_samples_split = min_samples_split,
            max_features      = max_features,
            max_samples       = max_samples,
            ccp_alpha         = round(ccp_alpha, 6),
            bootstrap         = bootstrap,
            oob_r2            = oob_r2,
            n_convergencia    = n_convergencia,
            buscar_optuna     = buscar,
            n_trials          = n_trials if buscar else 0,
            metodo_busqueda   = "optuna_oob_warmstart_early_stopping" if buscar else "manual",
            target_transform  = "log1p",
        ),
    )

    # ── 9. Guardar modelo ─────────────────────────────────────────────────────
    joblib.dump(modelo, MODELO_PATH)
    print(f"\n[{nombre}] Modelo guardado  : {MODELO_PATH}")

    return modelo


# ---------------------------------------------------------------------------
# Búsqueda bayesiana con Optuna + OOB/warm_start como "early stopping"
# ---------------------------------------------------------------------------

# Tope de árboles por trial; warm_start + OOB deciden cuántos se usan
# realmente (igual que el early stopping nativo en los modelos boosting).
_N_ESTIMATORS_BUSQUEDA = 1200
_PASO_BUSQUEDA         = 50     # árboles añadidos por bloque
_PACIENCIA_BUSQUEDA    = 5      # bloques sin mejora en OOB antes de detener el trial
_TOL_BUSQUEDA          = 0.001  # mejora mínima en R² OOB para no contar como estancado


def _callback_trazabilidad(log_path: Path, nombre: str):
    """
    Callback de Optuna que registra cada trial apenas termina:
      - imprime una línea en consola (número, R² OOB, árboles usados,
        duración, si es el mejor hasta ahora)
      - agrega una fila al CSV `log_path` (append), para poder revisar el
        avance sin perderlo si la búsqueda se interrumpe o tarda mucho.
    """
    def _callback(study, trial) -> None:
        valor    = trial.value if trial.value is not None else float("nan")
        dur_s    = trial.duration.total_seconds() if trial.duration else float("nan")
        n_arb    = trial.user_attrs.get("n_estimators", "N/A")
        es_mejor = (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.number == study.best_trial.number
        )
        marca    = "  <- mejor hasta ahora" if es_mejor else ""

        print(f"[{nombre}]   trial {trial.number:>3}/{len(study.trials) - 1}  "
              f"R2_oob={valor:.4f}  arboles={n_arb:<5}  "
              f"({dur_s:6.1f} s)  {trial.state.name}{marca}")

        fila = {
            "trial":       trial.number,
            "r2_oob":      valor,
            "n_estimators": n_arb,
            "duracion_s":  dur_s,
            "estado":      trial.state.name,
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
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
    X_train:     np.ndarray,
    y_train:     np.ndarray,   # escala log1p
    tipo_modelo: str,
    n_trials:    int,
    seed:        int,
) -> dict:
    """
    Optimización bayesiana (TPE) usando el R² OOB como señal de validación.

    Random Forest / ExtraTrees ya dejan ~37 % de las filas fuera de la
    muestra bootstrap de cada árbol (out-of-bag), así que ese subconjunto
    cumple el mismo papel que un fold de validación, sin necesitar un
    GroupKFold explícito ni una partición adicional.

    Cada trial crece los árboles en bloques de `_PASO_BUSQUEDA` con
    warm_start y se detiene apenas el R² OOB deja de mejorar más de
    `_TOL_BUSQUEDA` durante `_PACIENCIA_BUSQUEDA` bloques consecutivos —
    el equivalente, para RF/ExtraTrees, del early stopping nativo que usan
    los modelos boosting. Antes, la búsqueda evaluaba con GroupKFold
    (5 folds × hasta 1200 árboles completos, sin freno) y resultó inviable
    a esta escala de datos (ver docs/metodologia.md §5.3.1).

    Persistencia: el estudio se guarda en un SQLite (`optuna_rf.db`, un
    `study_name` distinto por `tipo_modelo`) en vez de mantenerse solo en
    memoria. Si el proceso se interrumpe a mitad de camino (p. ej. lo mata
    el entorno de ejecución), basta con volver a correr el script: retoma
    los trials ya completados en vez de empezar de cero. `n_trials` se
    interpreta como el total acumulado de la búsqueda (no como trials
    adicionales por corrida), así que llamadas repetidas con el mismo
    valor no vuelven a gastar presupuesto ya usado.
    """
    Clase = RandomForestRegressor if tipo_modelo == "rf" else ExtraTreesRegressor

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":         trial.suggest_categorical("max_depth", [None, 10, 20, 30, 50]),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features":      trial.suggest_categorical("max_features",
                                     ["sqrt", "log2", 0.3, 0.5, 0.7]),
            "max_samples":       trial.suggest_float("max_samples", 0.5, 1.0),
            "ccp_alpha":         trial.suggest_float("ccp_alpha", 0.0, 0.05, log=False),
        }

        estimador = Clase(
            n_estimators = 0,
            **params,
            bootstrap    = True,
            oob_score    = True,
            warm_start   = True,
            random_state = seed,
            n_jobs       = -1,
        )

        mejor_oob  = -np.inf
        sin_mejora = 0
        n_actual   = 0

        for _ in range(_N_ESTIMATORS_BUSQUEDA // _PASO_BUSQUEDA):
            n_actual += _PASO_BUSQUEDA
            estimador.n_estimators = n_actual
            estimador.fit(X_train, y_train)
            oob = estimador.oob_score_

            trial.report(oob, n_actual)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if oob > mejor_oob + _TOL_BUSQUEDA:
                mejor_oob  = oob
                sin_mejora = 0
            else:
                sin_mejora += 1

            if sin_mejora >= _PACIENCIA_BUSQUEDA:
                break

        trial.set_user_attr("n_estimators", n_actual)
        return float(mejor_oob)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials = 5,
        n_warmup_steps   = _PACIENCIA_BUSQUEDA * _PASO_BUSQUEDA,
    )

    db_path     = OUTPUT_DIR / "optuna_rf.db"
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(
        direction      = "maximize",
        sampler        = sampler,
        pruner         = pruner,
        storage        = storage_url,
        study_name     = f"rf_dbo_{tipo_modelo}",
        load_if_exists = True,
    )

    nombre = "ExtraTrees" if tipo_modelo == "extra" else "Random Forest"

    # Si el proceso murió a mitad de un trial (p. ej. lo mató el entorno de
    # ejecución), ese trial queda "atascado" en estado RUNNING para siempre
    # — Optuna no lo detecta ni lo reintenta solo. Se marca como fallido
    # para que no cuente como progreso ni bloquee la reanudación.
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
            print(f"[{nombre}] Trial {t.number} quedó inconcluso "
                  f"(corrida anterior interrumpida) — descartado.")

    log_path    = OUTPUT_DIR / f"optuna_trials_{tipo_modelo}.csv"
    n_previos   = sum(
        1 for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )
    n_faltantes = max(0, n_trials - n_previos)

    if n_previos:
        print(f"\n[{nombre}] Estudio existente en {db_path.name}: "
              f"{n_previos} trials ya registrados — retomando.")

    if n_faltantes == 0:
        print(f"[{nombre}] Ya se alcanzaron los {n_trials} trials objetivo, "
              f"no se ejecutan trials nuevos.")
    else:
        print(f"\n[{nombre}] Búsqueda bayesiana Optuna: {n_faltantes} trials nuevos "
              f"de {n_trials} totales ({n_previos} ya completados) — "
              f"R² OOB, warm_start early stopping, tope {_N_ESTIMATORS_BUSQUEDA} árboles...")
        print(f"[{nombre}] Trazabilidad por trial    : {log_path}")
        t0 = time.time()

        study.optimize(
            objective, n_trials=n_faltantes, show_progress_bar=False,
            callbacks=[_callback_trazabilidad(log_path, nombre)],
        )

        print(f"[{nombre}] Búsqueda completada en {time.time() - t0:.1f} s")

    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]

    print(f"[{nombre}] Mejor R² OOB (log1p) : {study.best_value:.4f}  "
          f"(trial {study.best_trial.number})")
    print(f"[{nombre}] Mejores hiperparámetros:")
    for k, v in best.items():
        print(f"       {k:<22} : {v}")
    print()

    return best


# ---------------------------------------------------------------------------
# Curva OOB con warm_start
# ---------------------------------------------------------------------------

def _curva_oob_warm(
    X_train:          np.ndarray,
    y_train:          np.ndarray,   # escala log1p
    tipo_modelo:      str,
    n_estimators:     int,
    max_depth:        int | None,
    min_samples_leaf: int,
    min_samples_split:int,
    max_features:     str,
    max_samples:      float | None,
    ccp_alpha:        float,
    seed:             int,
    paso:             int,
) -> tuple[list, list]:
    """
    Construye la curva R² OOB añadiendo árboles de a `paso` con warm_start.
    El OOB score se calcula en escala log1p (consistente con el entrenamiento).
    """
    rf = RandomForestRegressor(
        n_estimators      = 0,
        max_depth         = max_depth,
        min_samples_leaf  = min_samples_leaf,
        min_samples_split = min_samples_split,
        max_features      = max_features,
        max_samples       = max_samples,
        ccp_alpha         = ccp_alpha,
        bootstrap         = True,
        oob_score         = True,
        warm_start        = True,
        random_state      = seed,
        n_jobs            = -1,
    )

    ns, scores = [], []
    for n in range(paso, n_estimators + 1, paso):
        rf.n_estimators = n
        rf.fit(X_train, y_train)
        ns.append(n)
        scores.append(rf.oob_score_)

    return scores, ns


def _detectar_convergencia(ns: list, scores: list, ventana: int = 5) -> int:
    """
    Detecta el n_estimators donde el OOB R² deja de mejorar más de 0.001
    durante `ventana` puntos consecutivos.
    """
    for i in range(ventana, len(scores)):
        mejora = max(scores[i - ventana:i + 1]) - scores[i - ventana]
        if mejora < 0.001:
            return ns[i - ventana]
    return ns[-1]


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def _grafica_oob(ns: list, scores: list, n_conv: int):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, scores, color="#3C5488", linewidth=1.8, label="R² OOB")
    ax.axvline(n_conv, color="#E64B35", linewidth=1.5, linestyle="--",
               label=f"Convergencia ≈ {n_conv} árboles")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Número de árboles", fontweight="bold", fontsize=11)
    ax.set_ylabel("R² OOB (escala log1p DBO)", fontweight="bold", fontsize=11)
    ax.set_title("Convergencia OOB — Random Forest",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "curva_oob.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[RF] Figura guardada: {path}")


def _grafica_obs_vs_pred(y_true, y_pred, nombre: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    lim_max = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="#E64B35", linewidth=1.8,
            linestyle="--", label="1:1")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("DBO simulada QUAL2K (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel(f"DBO predicha {nombre} (mg/L)", fontweight="bold", fontsize=11)
    ax.set_title(f"Observado vs Predicho — conjunto de prueba\n{nombre}",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "obs_vs_pred.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[RF] Figura guardada: {path}")


def _grafica_residuos(y_true, y_pred, nombre: str):
    residuos = np.array(y_pred) - np.array(y_true)
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuos, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    ax.axhline(0, color="#E64B35", linewidth=1.8, linestyle="--")
    ax.set_xlim(0)
    ax.set_xlabel("DBO predicha (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Residuo  (pred − real)  mg/L", fontweight="bold", fontsize=11)
    ax.set_title(f"Residuos — conjunto de prueba\n{nombre}",
                 fontweight="bold", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "residuos.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[RF] Figura guardada: {path}")


def _grafica_importancia_nativa(modelo: _ModeloArbol, nombre: str):
    importancias = pd.Series(
        modelo.feature_importances_, index=FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importancias.index, importancias.values,
            color="#3C5488", edgecolor="white", height=0.65)
    ax.set_xlim(0)
    ax.set_xlabel("Importancia (reducción de impureza, MDI)",
                  fontweight="bold", fontsize=11)
    ax.set_title(f"Importancia nativa — {nombre} (MDI)",
                 fontweight="bold", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", axis="x", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", axis="x", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIGS_DIR / "importancia_nativa.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[RF] Figura guardada: {path}")


def _grafica_importancia_permutation(
    imp_mean: pd.Series, imp_std: pd.Series, nombre: str
):
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
    ax.set_title(
        f"Importancia de variables — {nombre} (Permutation Importance)\n"
        "(mayor ΔR² → más influyente  ·  negativo → feature irrelevante)",
        fontweight="bold", fontsize=11,
    )
    ax.minorticks_on()
    ax.grid(which="major", axis="x", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", axis="x", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIGS_DIR / "importancia_permutation.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[RF] Figura guardada: {path}")


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def _exportar_excel(
    metricas_train: dict,
    metricas_val:   dict,
    metricas_test:  dict,
    ns_oob:         list,
    oob_scores:     list,
    y_test:         np.ndarray,
    y_pred:         np.ndarray,
    imp_mean:       pd.Series,
    imp_std:        pd.Series,
    config:         dict,
):
    excel_path = OUTPUT_DIR / "resultados_rf.xlsx"

    df_metricas = pd.DataFrame([
        {"conjunto": "Train", **metricas_train},
        {"conjunto": "Val",   **metricas_val},
        {"conjunto": "Test",  **metricas_test},
    ])[["conjunto", "r2", "rmse", "mae", "bias"]]
    df_metricas.columns = ["Conjunto", "R²", "RMSE (mg/L)", "MAE (mg/L)", "Sesgo (mg/L)"]

    df_pred = pd.DataFrame({
        "dbo_real_mg_L":     y_test,
        "dbo_predicha_mg_L": y_pred,
        "residuo_mg_L":      y_pred - y_test,
    })

    if ns_oob:
        df_oob = pd.DataFrame({"n_arboles": ns_oob, "R2_OOB": oob_scores})
    else:
        df_oob = pd.DataFrame({"n_arboles": [], "R2_OOB": []})

    df_imp = pd.DataFrame({
        "variable":         imp_mean.index,
        "importancia_mean": imp_mean.values,
        "importancia_std":  imp_std.values,
    }).sort_values("importancia_mean", ascending=False)

    df_config = pd.DataFrame([
        {"parametro": k, "valor": v} for k, v in config.items()
    ])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet, df in [
            ("metricas",      df_metricas),
            ("predicciones",  df_pred),
            ("curva_oob",     df_oob),
            ("importancia",   df_imp),
            ("configuracion", df_config),
        ]:
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            for col_idx, col_name in enumerate(df.columns, start=1):
                ws.column_dimensions[
                    ws.cell(row=1, column=col_idx).column_letter
                ].width = max(len(str(col_name)) + 2, 14)

    print(f"[RF] Excel guardado: {excel_path}")


if __name__ == "__main__":
    entrenar(
        tipo_modelo       = "rf",   # "rf" o "extra"
        n_estimators      = 500,
        max_depth         = None,
        min_samples_leaf  = 5,
        min_samples_split = 2,
        max_features      = "sqrt",
        max_samples       = None,
        ccp_alpha         = 0.0,
        test_size         = 0.20,
        val_size          = 0.10,
        seed              = 42,
        n_perm            = 30,
        buscar            = True,
        n_trials          = 20,
    )
