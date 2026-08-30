"""
nn_trainer.py
==============
Entrena una red neuronal (MLP) como metamodelo para predecir DBO.

Arquitectura:
    Linear → ReLU → Dropout  (× n capas ocultas)  → Linear (salida)

Normalización:
    Features y target → StandardScaler ajustado solo en train.

Split:
    Train  70 %  |  Val  10 %  |  Test  20 %  (por sim_id, no por fila)

Uso:
    Editar los parámetros en el bloque __main__ al final del archivo y ejecutar:
        python metamodelo/nn_trainer.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# La salida puede quedar redirigida a un archivo con encoding cp1252 (Windows),
# lo que rompe los prints con caracteres como "→". Forzar UTF-8 evita ese crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos    import cargar_datos, FEATURES, TARGET
from metamodelo.metricas import calcular, intervalo_conformal, intervalo_cqr

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

OUTPUT_DIR      = _ROOT / "resultados" / "t_rio_jordan_metamodelo"
MODELO_PATH     = OUTPUT_DIR / "nn_dbo.pt"
SCALER_X_PATH   = OUTPUT_DIR / "nn_scaler_x.joblib"
SCALER_Y_PATH   = OUTPUT_DIR / "nn_scaler_y.joblib"
FIGS_DIR        = OUTPUT_DIR / "figuras_nn"
CHECKPOINT_PATH = OUTPUT_DIR / "nn_checkpoint.pt"
CONFORMAL_PATH  = OUTPUT_DIR / "nn_conformal.json"

# ---------------------------------------------------------------------------
# Rutas de salida — modelo de cuantiles (CQR), independiente del puntual
# ---------------------------------------------------------------------------

QUANTILES_CQR       = [0.025, 0.05, 0.5, 0.95, 0.975]  # idx: 0=q025 1=q05 2=mediana 3=q95 4=q975
FIGS_CQR_DIR        = OUTPUT_DIR / "figuras_nn_cqr"
MODELO_CQR_PATH     = OUTPUT_DIR / "nn_dbo_cqr.pt"
SCALER_X_CQR_PATH   = OUTPUT_DIR / "nn_scaler_x_cqr.joblib"
SCALER_Y_CQR_PATH   = OUTPUT_DIR / "nn_scaler_y_cqr.joblib"
CHECKPOINT_CQR_PATH = OUTPUT_DIR / "nn_checkpoint_cqr.pt"
CQR_PATH            = OUTPUT_DIR / "nn_cqr.json"

# ---------------------------------------------------------------------------
# Arquitectura MLP
# ---------------------------------------------------------------------------

class MLPDbo(nn.Module):
    """MLP simple: Linear → ReLU → Dropout por cada capa oculta.

    n_salida=1 (default) reproduce el modelo puntual original. n_salida>1 se
    usa para el modelo de cuantiles (CQR, ver entrenar_cqr): cada neurona de
    salida predice uno de QUANTILES_CQR, en el mismo orden.
    """

    def __init__(self, n_entrada: int, capas: list[int], dropout: float, n_salida: int = 1):
        super().__init__()
        self.n_salida = n_salida

        bloques = []
        prev    = n_entrada
        for neuronas in capas:
            bloques += [
                nn.Linear(prev, neuronas),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = neuronas
        bloques.append(nn.Linear(prev, n_salida))

        self.red = nn.Sequential(*bloques)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.red(x)
        return out.squeeze(1) if self.n_salida == 1 else out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DboDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, y_orig: np.ndarray | None = None):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32)
        self.y_orig = torch.tensor(y_orig, dtype=torch.float32) if y_orig is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.y_orig is not None:
            return self.X[idx], self.y[idx], self.y_orig[idx]
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Pérdida RMSE
# ---------------------------------------------------------------------------

def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def rmse_loss_weighted(pred: torch.Tensor, target: torch.Tensor,
                       w: torch.Tensor) -> torch.Tensor:
    """RMSE ponderado: w = 1 + y_orig/y_max, así valores altos pesan más."""
    return torch.sqrt(torch.mean(w * (pred - target) ** 2))


def pinball_loss_weighted(pred: torch.Tensor, target: torch.Tensor,
                           w: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """
    Pinball (check) loss para regresión cuantílica, ponderada igual que
    rmse_loss_weighted (w = 1 + y_orig/y_max), sumada sobre todos los
    cuantiles de QUANTILES_CQR y promediada sobre el batch.

    pred: (batch, n_cuantiles) — una columna por cuantil, mismo orden que
    `quantiles`. target/w: (batch,).
    """
    target_col = target.unsqueeze(1)                    # (batch, 1) — broadcast
    w_col      = w.unsqueeze(1)
    errores    = target_col - pred                       # (batch, n_cuantiles)
    q          = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).unsqueeze(0)
    perdidas   = torch.maximum((q - 1) * errores, q * errores)
    return torch.mean(w_col * perdidas)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar(
    capas:          list[int],
    epochs:         int,
    lr:             float,
    batch_size:     int,
    dropout:        float,
    test_size:      float,
    val_size:       float,
    seed:           int,
    paciencia:      int,
    n_perm:            int   = 30,
    weight_decay:      float = 0.0,
    buscar:            bool  = False,
    n_trials:          int   = 8,
    epochs_busqueda:   int   = 60,
    paciencia_busqueda: int  = 20,
) -> MLPDbo:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  METAMODELO DBO — Red Neuronal (MLP)")
    print(f"  Dispositivo : {device}")
    print("=" * 55)

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    df = cargar_datos()

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

    # Features — StandardScaler ajustado solo en train
    scaler_x = StandardScaler()
    X_train  = scaler_x.fit_transform(train_df[FEATURES].values)
    X_val    = scaler_x.transform(val_df[FEATURES].values)
    X_test   = scaler_x.transform(test_df[FEATURES].values)
    joblib.dump(scaler_x, SCALER_X_PATH)

    # Target — StandardScaler en escala original
    y_train_orig = train_df[TARGET].values.reshape(-1, 1)
    y_val_orig   = val_df[TARGET].values.reshape(-1, 1)
    y_test_orig  = test_df[TARGET].values.reshape(-1, 1)

    scaler_y = StandardScaler()
    y_train  = scaler_y.fit_transform(y_train_orig).ravel()
    y_val    = scaler_y.transform(y_val_orig).ravel()
    joblib.dump(scaler_y, SCALER_Y_PATH)

    y_train_orig = y_train_orig.ravel()
    y_val_orig   = y_val_orig.ravel()
    y_test_orig  = y_test_orig.ravel()

    print(f"[datos] Target — rango train: "
          f"[{y_train_orig.min():.1f}, {y_train_orig.max():.1f}] mg/L\n")

    # Busqueda bayesiana con Optuna (opcional) — split unico train/val
    if buscar:
        study = _buscar_optuna(
            X_train, y_train, y_train_orig,
            X_val,   y_val,
            n_entrada  = len(FEATURES),
            n_trials   = n_trials,
            paciencia  = paciencia_busqueda,
            max_epochs = epochs_busqueda,
            seed       = seed,
            device     = device,
        )
        best_params  = study.best_params
        n_capas_best = best_params["n_capas"]
        capas        = [best_params[f"neuronas_{i}"] for i in range(n_capas_best)]
        lr           = best_params["lr"]
        dropout      = best_params["dropout"]
        batch_size   = best_params["batch_size"]
        weight_decay = best_params["weight_decay"]
    else:
        study = None

    # DataLoaders
    y_max = float(y_train_orig.max())
    loader_train = torch.utils.data.DataLoader(
        DboDataset(X_train, y_train, y_train_orig), batch_size=batch_size, shuffle=True
    )
    loader_val = torch.utils.data.DataLoader(
        DboDataset(X_val, y_val), batch_size=batch_size * 4
    )

    # ── 2. Modelo, optimizador y scheduler ───────────────────────────────────
    modelo    = MLPDbo(n_entrada=len(FEATURES), capas=capas, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = 0.5,        # lr ← lr × 0.5 cuando val RMSE no mejora
        patience  = 15,         # espera 15 epochs antes de reducir
        min_lr    = 1e-6,
    )

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"[red] Arquitectura  : {len(FEATURES)} → {' → '.join(str(c) for c in capas)} → 1")
    print(f"[red] Parámetros    : {n_params:,}")
    print(f"[red] Epochs        : {epochs}  |  batch {batch_size}  |  lr inicial {lr}")
    print(f"[red] Scheduler     : ReduceLROnPlateau (factor=0.5, paciencia_lr=15, min_lr=1e-6)")
    print(f"[red] Dropout       : {dropout}  |  loss RMSE  |  optimizer Adam\n")

    # ── 3. Bucle de entrenamiento ──────────────────────────────────────────────
    hist_train, hist_val, hist_lr = [], [], []
    mejor_val         = np.inf
    mejor_epoch       = 1
    epochs_sin_mejora = 0
    mejor_estado      = None
    epoch_inicio      = 1

    # Reanudar desde checkpoint si el entrenamiento se pauso/interrumpio antes
    # (misma arquitectura e hiperparametros que esta corrida).
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        if ckpt.get("capas") == capas and ckpt.get("dropout") == dropout \
                and ckpt.get("lr") == lr and ckpt.get("batch_size") == batch_size:
            modelo.load_state_dict(ckpt["modelo_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            torch.set_rng_state(ckpt["rng_state"])
            epoch_inicio       = ckpt["epoch"] + 1
            hist_train         = ckpt["hist_train"]
            hist_val            = ckpt["hist_val"]
            hist_lr             = ckpt["hist_lr"]
            mejor_val           = ckpt["mejor_val"]
            mejor_epoch          = ckpt["mejor_epoch"]
            epochs_sin_mejora   = ckpt["epochs_sin_mejora"]
            mejor_estado         = ckpt["mejor_estado"]
            print(f"[red] Checkpoint encontrado — reanudando desde epoch {epoch_inicio} "
                  f"(mejor val hasta ahora: {mejor_val:.4f}  en epoch {mejor_epoch})\n")
        else:
            print("[red] Checkpoint encontrado pero con hiperparametros distintos — se ignora.\n")

    for epoch in range(epoch_inicio, epochs + 1):

        modelo.train()
        for xb, yb, yb_orig in loader_train:
            xb, yb, yb_orig = xb.to(device), yb.to(device), yb_orig.to(device)
            w = 1.0 + yb_orig / y_max
            optimizer.zero_grad()
            loss = rmse_loss_weighted(modelo(xb), yb, w)
            loss.backward()
            optimizer.step()

        # Evaluar en modo eval (Dropout desactivado)
        modelo.eval()
        rmse_train = _evaluar_rmse(modelo, loader_train, device)
        rmse_val   = _evaluar_rmse(modelo, loader_val,   device)

        scheduler.step(rmse_val)

        hist_train.append(rmse_train)
        hist_val.append(rmse_val)
        hist_lr.append(optimizer.param_groups[0]["lr"])

        if rmse_val < mejor_val:
            mejor_val         = rmse_val
            mejor_epoch       = epoch
            mejor_estado      = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        # Checkpoint automatico cada epoch: si el entrenamiento se pausa o se
        # interrumpe, la proxima corrida con los mismos hiperparametros lo retoma.
        torch.save({
            "epoch":             epoch,
            "capas":             capas,
            "dropout":           dropout,
            "lr":                lr,
            "batch_size":        batch_size,
            "modelo_state":      modelo.state_dict(),
            "optimizer_state":   optimizer.state_dict(),
            "scheduler_state":   scheduler.state_dict(),
            "rng_state":         torch.get_rng_state(),
            "hist_train":        hist_train,
            "hist_val":          hist_val,
            "hist_lr":           hist_lr,
            "mejor_val":         mejor_val,
            "mejor_epoch":       mejor_epoch,
            "epochs_sin_mejora": epochs_sin_mejora,
            "mejor_estado":      mejor_estado,
        }, CHECKPOINT_PATH)

        if epoch % 10 == 0 or epoch == 1:
            lr_actual = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>4d}/{epochs} | "
                  f"RMSE train {rmse_train:7.4f} | "
                  f"RMSE val {rmse_val:7.4f}  (norm) | "
                  f"lr {lr_actual:.2e}"
                  + ("  ← mejor" if epochs_sin_mejora == 0 else ""))

        if epochs_sin_mejora >= paciencia:
            print(f"\n[red] Early stopping en epoch {epoch} "
                  f"(sin mejora por {paciencia} epochs).")
            break

    modelo.load_state_dict(mejor_estado)
    print(f"\n[red] Mejor epoch      : {mejor_epoch}  (val RMSE = {mejor_val:.4f}  norm)")

    # Guardar el modelo final YA, antes de los pasos lentos que siguen
    # (permutation importance, graficas, Excel) — si alguno de esos falla o el
    # proceso se interrumpe, el modelo entrenado no se pierde.
    torch.save(modelo.state_dict(), MODELO_PATH)
    print(f"[red] Modelo guardado   : {MODELO_PATH}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # ── 4. Predicciones en escala original (mg/L) ────────────────────────────
    modelo.eval()
    with torch.no_grad():
        y_pred_norm    = modelo(torch.tensor(X_test,  dtype=torch.float32).to(device)).cpu().numpy()
        y_pred_tr_norm = modelo(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()

    y_pred    = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
    y_pred_tr = scaler_y.inverse_transform(y_pred_tr_norm.reshape(-1, 1)).ravel()

    metricas_train = calcular(y_train_orig, y_pred_tr, nombre="Train")
    metricas_test  = calcular(y_test_orig,  y_pred,    nombre="Test")

    # ── 4.1 Intervalo de predicción — split conformal prediction ────────────
    # Se calibra con el conjunto de test, que no participó en el entrenamiento
    # ni en la seleccion de hiperparametros (early stopping usa validacion).
    print("[red] Calculando margenes de prediccion (conformal prediction)...")
    conformal_90 = intervalo_conformal(y_test_orig, y_pred, alpha=0.10)
    conformal_95 = intervalo_conformal(y_test_orig, y_pred, alpha=0.05)

    with open(CONFORMAL_PATH, "w", encoding="utf-8") as fh:
        json.dump({"90%": conformal_90, "95%": conformal_95}, fh, indent=2)
    print(f"[red] Conformal guardado : {CONFORMAL_PATH}")

    # ── 5. Permutation Importance ─────────────────────────────────────────────
    print(f"[red] Calculando permutation importance ({n_perm} repeticiones)...")
    imp_mean, imp_std = _permutation_importance_nn(
        modelo, X_val, y_val_orig, scaler_y, device, n_perm, seed, FEATURES
    )

    # ── 6. Gráficas ───────────────────────────────────────────────────────────
    _grafica_curvas(hist_train, hist_val, hist_lr, mejor_epoch)
    _grafica_obs_vs_pred(y_test_orig, y_pred)
    _grafica_residuos(y_test_orig, y_pred)
    _grafica_importancia(imp_mean, imp_std)

    # ── 7. Exportar resultados ────────────────────────────────────────────────
    _exportar_excel(
        metricas_train = metricas_train,
        metricas_test  = metricas_test,
        hist_train     = hist_train,
        hist_val       = hist_val,
        hist_lr        = hist_lr,
        y_test         = y_test_orig,
        y_pred         = y_pred,
        imp_mean       = imp_mean,
        imp_std        = imp_std,
        conformal      = {"90%": conformal_90, "95%": conformal_95},
        study          = study,
        config         = dict(capas=capas, epochs=epochs, lr=lr,
                              batch_size=batch_size, dropout=dropout,
                              weight_decay=weight_decay,
                              buscar_optuna=buscar, n_trials=n_trials if buscar else 0,
                              metodo_busqueda="optuna_split_unico_early_stopping_pruning" if buscar else "manual"),
    )

    print(f"[red] Scaler X guardado : {SCALER_X_PATH}")
    print(f"[red] Scaler Y guardado : {SCALER_Y_PATH}")

    return modelo


# ---------------------------------------------------------------------------
# Entrenamiento — modelo de cuantiles (CQR)
# ---------------------------------------------------------------------------

def entrenar_cqr(
    capas:        list[int],
    epochs:       int,
    lr:           float,
    batch_size:   int,
    dropout:      float,
    test_size:    float,
    val_size:     float,
    seed:         int,
    paciencia:    int,
    weight_decay: float = 0.0,
) -> MLPDbo:
    """
    Entrena un MLP que predice directamente los cuantiles QUANTILES_CQR de
    DBO (pinball loss) en vez de un único valor puntual (RMSE), y calibra
    los cuantiles bajo/alto con Conformalized Quantile Regression (CQR,
    Romano et al. 2019) para obtener intervalos de predicción de ancho
    variable — más angostos donde el modelo tiene más confianza, en vez del
    margen fijo qhat de intervalo_conformal (ver entrenar()).

    Usa el mismo split por sim_id, arquitectura base y esquema de
    ponderación (valores altos de DBO pesan más) que el modelo puntual, para
    que ambos sean comparables. No repite la búsqueda de hiperparámetros de
    Optuna: reutiliza los hiperparámetros ya encontrados para el modelo
    puntual como punto de partida razonable.

    Artefactos independientes del modelo puntual (no sobrescribe nn_dbo.pt
    ni nn_conformal.json): MODELO_CQR_PATH, SCALER_X_CQR_PATH,
    SCALER_Y_CQR_PATH, CQR_PATH, figuras en FIGS_CQR_DIR.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_CQR_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  METAMODELO DBO — Red Neuronal de Cuantiles (CQR)")
    print(f"  Cuantiles   : {QUANTILES_CQR}")
    print(f"  Dispositivo : {device}")
    print("=" * 55)

    # ── 1. Datos — mismo split por sim_id que el modelo puntual ─────────────
    df = cargar_datos()

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

    print(f"\n[cqr] Train : {len(train_df):,} filas ({len(ids_train)} sims)")
    print(f"[cqr] Val   : {len(val_df):,} filas ({len(ids_val)} sims)")
    print(f"[cqr] Test  : {len(test_df):,} filas ({len(ids_test)} sims)\n")

    scaler_x = StandardScaler()
    X_train  = scaler_x.fit_transform(train_df[FEATURES].values)
    X_val    = scaler_x.transform(val_df[FEATURES].values)
    X_test   = scaler_x.transform(test_df[FEATURES].values)
    joblib.dump(scaler_x, SCALER_X_CQR_PATH)

    y_train_orig = train_df[TARGET].values.reshape(-1, 1)
    y_val_orig   = val_df[TARGET].values.reshape(-1, 1)
    y_test_orig  = test_df[TARGET].values.reshape(-1, 1)

    scaler_y = StandardScaler()
    y_train  = scaler_y.fit_transform(y_train_orig).ravel()
    y_val    = scaler_y.transform(y_val_orig).ravel()
    joblib.dump(scaler_y, SCALER_Y_CQR_PATH)

    y_train_orig = y_train_orig.ravel()
    y_val_orig   = y_val_orig.ravel()
    y_test_orig  = y_test_orig.ravel()

    y_max = float(y_train_orig.max())
    loader_train = torch.utils.data.DataLoader(
        DboDataset(X_train, y_train, y_train_orig), batch_size=batch_size, shuffle=True
    )
    loader_val = torch.utils.data.DataLoader(
        DboDataset(X_val, y_val, y_val_orig), batch_size=batch_size * 4
    )

    # ── 2. Modelo, optimizador y scheduler ───────────────────────────────────
    modelo = MLPDbo(
        n_entrada=len(FEATURES), capas=capas, dropout=dropout, n_salida=len(QUANTILES_CQR)
    ).to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"[cqr] Arquitectura  : {len(FEATURES)} → {' → '.join(str(c) for c in capas)} "
          f"→ {len(QUANTILES_CQR)} (cuantiles)")
    print(f"[cqr] Parámetros    : {n_params:,}")
    print(f"[cqr] Epochs        : {epochs}  |  batch {batch_size}  |  lr inicial {lr}")
    print(f"[cqr] Dropout       : {dropout}  |  loss pinball ponderada  |  optimizer Adam\n")

    # ── 3. Bucle de entrenamiento ──────────────────────────────────────────────
    hist_train, hist_val, hist_lr = [], [], []
    mejor_val         = np.inf
    mejor_epoch       = 1
    epochs_sin_mejora = 0
    mejor_estado      = None
    epoch_inicio      = 1

    if CHECKPOINT_CQR_PATH.exists():
        ckpt = torch.load(CHECKPOINT_CQR_PATH, map_location=device, weights_only=False)
        if ckpt.get("capas") == capas and ckpt.get("dropout") == dropout \
                and ckpt.get("lr") == lr and ckpt.get("batch_size") == batch_size:
            modelo.load_state_dict(ckpt["modelo_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            torch.set_rng_state(ckpt["rng_state"])
            epoch_inicio       = ckpt["epoch"] + 1
            hist_train         = ckpt["hist_train"]
            hist_val           = ckpt["hist_val"]
            hist_lr            = ckpt["hist_lr"]
            mejor_val          = ckpt["mejor_val"]
            mejor_epoch        = ckpt["mejor_epoch"]
            epochs_sin_mejora  = ckpt["epochs_sin_mejora"]
            mejor_estado       = ckpt["mejor_estado"]
            print(f"[cqr] Checkpoint encontrado — reanudando desde epoch {epoch_inicio} "
                  f"(mejor val hasta ahora: {mejor_val:.4f}  en epoch {mejor_epoch})\n")
        else:
            print("[cqr] Checkpoint encontrado pero con hiperparametros distintos — se ignora.\n")

    for epoch in range(epoch_inicio, epochs + 1):

        modelo.train()
        for xb, yb, yb_orig in loader_train:
            xb, yb, yb_orig = xb.to(device), yb.to(device), yb_orig.to(device)
            w = 1.0 + yb_orig / y_max
            optimizer.zero_grad()
            loss = pinball_loss_weighted(modelo(xb), yb, w, QUANTILES_CQR)
            loss.backward()
            optimizer.step()

        modelo.eval()
        pinball_train = _evaluar_pinball(modelo, loader_train, device, QUANTILES_CQR)
        pinball_val   = _evaluar_pinball(modelo, loader_val,   device, QUANTILES_CQR)

        scheduler.step(pinball_val)

        hist_train.append(pinball_train)
        hist_val.append(pinball_val)
        hist_lr.append(optimizer.param_groups[0]["lr"])

        if pinball_val < mejor_val:
            mejor_val         = pinball_val
            mejor_epoch       = epoch
            mejor_estado      = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        torch.save({
            "epoch":             epoch,
            "capas":             capas,
            "dropout":           dropout,
            "lr":                lr,
            "batch_size":        batch_size,
            "modelo_state":      modelo.state_dict(),
            "optimizer_state":   optimizer.state_dict(),
            "scheduler_state":   scheduler.state_dict(),
            "rng_state":         torch.get_rng_state(),
            "hist_train":        hist_train,
            "hist_val":          hist_val,
            "hist_lr":           hist_lr,
            "mejor_val":         mejor_val,
            "mejor_epoch":       mejor_epoch,
            "epochs_sin_mejora": epochs_sin_mejora,
            "mejor_estado":      mejor_estado,
        }, CHECKPOINT_CQR_PATH)

        if epoch % 10 == 0 or epoch == 1:
            lr_actual = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>4d}/{epochs} | "
                  f"pinball train {pinball_train:7.4f} | "
                  f"pinball val {pinball_val:7.4f}  (norm) | "
                  f"lr {lr_actual:.2e}"
                  + ("  ← mejor" if epochs_sin_mejora == 0 else ""))

        if epochs_sin_mejora >= paciencia:
            print(f"\n[cqr] Early stopping en epoch {epoch} "
                  f"(sin mejora por {paciencia} epochs).")
            break

    modelo.load_state_dict(mejor_estado)
    print(f"\n[cqr] Mejor epoch      : {mejor_epoch}  (val pinball = {mejor_val:.4f}  norm)")

    torch.save(modelo.state_dict(), MODELO_CQR_PATH)
    print(f"[cqr] Modelo guardado   : {MODELO_CQR_PATH}")

    if CHECKPOINT_CQR_PATH.exists():
        CHECKPOINT_CQR_PATH.unlink()

    # ── 4. Predicciones en escala original (mg/L), una columna por cuantil ──
    modelo.eval()
    with torch.no_grad():
        pred_test_norm  = modelo(torch.tensor(X_test,  dtype=torch.float32).to(device)).cpu().numpy()
        pred_train_norm = modelo(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()

    # Las 5 salidas se predicen de forma independiente, sin restricción de
    # monotonicidad entre ellas: nada impide que la red prediga q05 > q95
    # ("quantile crossing") en algunos puntos, sobre todo con datos atípicos.
    # Se corrige reordenando cada fila (operador de rearreglo de Chernozhukov,
    # Fernández-Val & Galichon, 2010, "Quantile and Probability Curves Without
    # Crossing", Econometrica) — como StandardScaler es una transformación
    # afín con escala positiva, ordenar antes o después de desescalar es
    # equivalente.
    pred_test_norm  = np.sort(pred_test_norm,  axis=1)
    pred_train_norm = np.sort(pred_train_norm, axis=1)

    def _desescalar(pred_norm: np.ndarray) -> np.ndarray:
        # StandardScaler es afín (escala + traslado): aplicarlo columna a
        # columna sobre cada cuantil conserva su orden y unidades (mg/L).
        return np.column_stack([
            scaler_y.inverse_transform(pred_norm[:, [i]]).ravel()
            for i in range(pred_norm.shape[1])
        ])

    pred_test  = _desescalar(pred_test_norm)
    pred_train = _desescalar(pred_train_norm)

    idx_q025, idx_q05, idx_mediana, idx_q95, idx_q975 = range(5)
    y_pred_mediana    = pred_test[:, idx_mediana]
    y_pred_tr_mediana = pred_train[:, idx_mediana]

    metricas_train = calcular(y_train_orig, y_pred_tr_mediana, nombre="Train (mediana)")
    metricas_test  = calcular(y_test_orig,  y_pred_mediana,    nombre="Test (mediana)")

    # ── 4.1 Calibración CQR sobre el conjunto de test ────────────────────────
    print("[cqr] Calibrando intervalos (Conformalized Quantile Regression)...")
    cqr_90 = intervalo_cqr(y_test_orig, pred_test[:, idx_q05],  pred_test[:, idx_q95],  alpha=0.10)
    cqr_95 = intervalo_cqr(y_test_orig, pred_test[:, idx_q025], pred_test[:, idx_q975], alpha=0.05)

    cqr_resultado = {
        "90%": {**cqr_90, "cuantil_lo": 0.05,  "cuantil_hi": 0.95},
        "95%": {**cqr_95, "cuantil_lo": 0.025, "cuantil_hi": 0.975},
    }
    with open(CQR_PATH, "w", encoding="utf-8") as fh:
        json.dump(cqr_resultado, fh, indent=2)
    print(f"[cqr] Calibración guardada : {CQR_PATH}")

    # Cobertura empírica del intervalo calibrado sobre el propio test — solo
    # diagnóstico (la calibración se hizo con este mismo conjunto, así que se
    # espera que coincida con la nominal; la validación honesta es sobre
    # datos futuros, como hace intervalo_conformal).
    lo_90 = np.clip(pred_test[:, idx_q05]  - cqr_90["q_correction"], 0, None)
    hi_90 = pred_test[:, idx_q95] + cqr_90["q_correction"]
    lo_95 = np.clip(pred_test[:, idx_q025] - cqr_95["q_correction"], 0, None)
    hi_95 = pred_test[:, idx_q975] + cqr_95["q_correction"]

    # Cuando la corrección es negativa (los cuantiles crudos ya sobre-cubren
    # el conjunto de calibración) puede encoger el intervalo hasta cruzarlo
    # en filas de spread muy angosto. Se pone un piso de ancho no-negativo:
    # el caso límite es un intervalo puntual (lo = hi), nunca uno invertido.
    hi_90 = np.maximum(hi_90, lo_90)
    hi_95 = np.maximum(hi_95, lo_95)

    cobertura_90 = float(np.mean((y_test_orig >= lo_90) & (y_test_orig <= hi_90)))
    cobertura_95 = float(np.mean((y_test_orig >= lo_95) & (y_test_orig <= hi_95)))
    ancho_90 = hi_90 - lo_90
    ancho_95 = hi_95 - lo_95
    print(f"[cqr] Cobertura empírica 90% (en test) : {cobertura_90:.1%}  "
          f"|  ancho promedio {ancho_90.mean():.2f} mg/L (min {ancho_90.min():.2f} / max {ancho_90.max():.2f})")
    print(f"[cqr] Cobertura empírica 95% (en test) : {cobertura_95:.1%}  "
          f"|  ancho promedio {ancho_95.mean():.2f} mg/L (min {ancho_95.min():.2f} / max {ancho_95.max():.2f})")

    # ── 5. Gráficas ───────────────────────────────────────────────────────────
    _grafica_curvas_cqr(hist_train, hist_val, hist_lr, mejor_epoch)
    _grafica_obs_vs_pred_cqr(y_test_orig, y_pred_mediana)
    _grafica_ancho_intervalo_cqr(y_pred_mediana, ancho_90, ancho_95)

    # ── 6. Exportar resultados ────────────────────────────────────────────────
    _exportar_excel_cqr(
        metricas_train = metricas_train,
        metricas_test  = metricas_test,
        hist_train     = hist_train,
        hist_val       = hist_val,
        hist_lr        = hist_lr,
        y_test         = y_test_orig,
        pred_test      = pred_test,
        intervalos     = {"lo_90": lo_90, "hi_90": hi_90, "lo_95": lo_95, "hi_95": hi_95},
        cqr            = cqr_resultado,
        cobertura      = {"90%": cobertura_90, "95%": cobertura_95},
        ancho          = {"90%": ancho_90, "95%": ancho_95},
        config         = dict(capas=capas, epochs=epochs, lr=lr,
                              batch_size=batch_size, dropout=dropout,
                              weight_decay=weight_decay, quantiles=QUANTILES_CQR),
    )

    print(f"[cqr] Scaler X guardado : {SCALER_X_CQR_PATH}")
    print(f"[cqr] Scaler Y guardado : {SCALER_Y_CQR_PATH}")

    return modelo


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _evaluar_rmse(modelo, loader, device) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            pred   = modelo(xb)
            total += torch.sum((pred - yb) ** 2).item()
            n     += len(yb)
    return (total / n) ** 0.5


def _evaluar_pinball(modelo, loader, device, quantiles: list[float]) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, yb_orig in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred   = modelo(xb)
            errores  = yb.unsqueeze(1) - pred
            q        = torch.tensor(quantiles, device=device, dtype=pred.dtype).unsqueeze(0)
            perdidas = torch.maximum((q - 1) * errores, q * errores)
            total   += perdidas.sum().item()
            n       += perdidas.numel()
    return total / n


class _NNWrapper(BaseEstimator, RegressorMixin):
    """Envuelve MLPDbo para que sklearn.inspection.permutation_importance lo use como estimador."""

    def __init__(self, modelo: MLPDbo, scaler_y: StandardScaler, device):
        self.modelo   = modelo
        self.scaler_y = scaler_y
        self.device   = device

    def fit(self, X, y):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.modelo.eval()
        with torch.no_grad():
            pred_norm = self.modelo(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        return self.scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).ravel()


def _permutation_importance_nn(
    modelo:     MLPDbo,
    X_val:      np.ndarray,
    y_val_orig: np.ndarray,
    scaler_y:   StandardScaler,
    device:     object,
    n_perm:     int,
    seed:       int,
    features:   list[str],
) -> tuple[pd.Series, pd.Series]:

    wrapper = _NNWrapper(modelo, scaler_y, device)

    r2_base = r2_score(y_val_orig, wrapper.predict(X_val))

    perm = permutation_importance(
        wrapper, X_val, y_val_orig,
        n_repeats    = n_perm,
        random_state = seed,
        scoring      = "r2",
        n_jobs       = 1,
    )
    imp_mean = pd.Series(perm.importances_mean, index=features, name="mean")
    imp_std  = pd.Series(perm.importances_std,  index=features, name="std")

    print(f"[red] R² base (val, mg/L) : {r2_base:.4f}")
    print("[red] Top-5 features por importancia (ΔR²):")
    for feat, val in imp_mean.sort_values(ascending=False).head(5).items():
        print(f"       {feat:<25} Δ R² = {val:+.4f}")

    return imp_mean, imp_std


# ---------------------------------------------------------------------------
# Busqueda bayesiana con Optuna — split unico train/val + pruning por epoca
# ---------------------------------------------------------------------------

def _callback_trazabilidad(log_path: Path):
    """
    Callback de Optuna que registra cada trial apenas termina:
      - imprime una linea en consola (numero, RMSE val, duracion, si es el
        mejor hasta ahora)
      - agrega una fila al CSV `log_path` (append), para no perder el avance
        si la busqueda se interrumpe o tarda mucho.
    """
    def _callback(study, trial) -> None:
        valor    = trial.value if trial.value is not None else float("nan")
        dur_s    = trial.duration.total_seconds() if trial.duration else float("nan")
        es_mejor = trial.number == study.best_trial.number
        marca    = "  <- mejor hasta ahora" if es_mejor else ""

        print(f"[red]   trial {trial.number:>3}/{len(study.trials) - 1}  "
              f"RMSE_val={valor:.4f}  ({dur_s:6.1f} s){marca}")

        fila = {
            "trial":      trial.number,
            "rmse_val":   valor,
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
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    y_train_orig: np.ndarray,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    n_entrada:    int,
    n_trials:     int,
    paciencia:    int,
    max_epochs:   int,
    seed:         int,
    device:       "torch.device",
):
    """
    Optimiza arquitectura e hiperparametros del MLP con TPE sobre el mismo
    split train/val (agrupado por sim_id) que usa el entrenamiento final.

    Cada trial entrena una sola vez (sin k-fold), reporta el RMSE de
    validacion (escala normalizada) epoca a epoca a Optuna, que puede
    podar (MedianPruner) trials poco prometedores antes de llegar a
    max_epochs o a la paciencia de early stopping — evitando que una
    combinacion de hiperparametros mala consuma el presupuesto completo
    de epocas, tal como ya hace el early stopping en los modelos boosting.

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

    y_max = float(y_train_orig.max())

    def objective(trial) -> float:
        # Espacio acotado para busqueda corta (CPU, sin GPU): se excluyen
        # capas de 512 neuronas y batch_size=128, las combinaciones mas
        # lentas por epoca, para mantener el costo por trial bajo control.
        n_capas = trial.suggest_int("n_capas", 2, 3)
        capas   = [
            trial.suggest_categorical(f"neuronas_{i}", [32, 64, 128, 256])
            for i in range(n_capas)
        ]
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout      = trial.suggest_float("dropout", 0.0, 0.5)
        batch_size   = trial.suggest_categorical("batch_size", [256, 512, 1024])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        torch.manual_seed(seed)
        modelo    = MLPDbo(n_entrada=n_entrada, capas=capas, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)

        loader_train = torch.utils.data.DataLoader(
            DboDataset(X_train, y_train, y_train_orig), batch_size=batch_size, shuffle=True
        )
        loader_val = torch.utils.data.DataLoader(
            DboDataset(X_val, y_val), batch_size=batch_size * 4
        )

        mejor_val         = np.inf
        epochs_sin_mejora = 0

        for epoch in range(1, max_epochs + 1):
            modelo.train()
            for xb, yb, yb_orig in loader_train:
                xb, yb, yb_orig = xb.to(device), yb.to(device), yb_orig.to(device)
                w = 1.0 + yb_orig / y_max
                optimizer.zero_grad()
                loss = rmse_loss_weighted(modelo(xb), yb, w)
                loss.backward()
                optimizer.step()

            modelo.eval()
            rmse_val = _evaluar_rmse(modelo, loader_val, device)

            if rmse_val < mejor_val:
                mejor_val         = rmse_val
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1

            trial.report(rmse_val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if epochs_sin_mejora >= paciencia:
                break

        return float(mejor_val)

    sampler  = optuna.samplers.TPESampler(seed=seed)
    # n_startup_trials/n_warmup_steps bajos: con una busqueda corta (pocos
    # trials, pocas epocas) interesa podar trials malos lo antes posible.
    pruner   = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=8)
    study    = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    log_path = OUTPUT_DIR / "optuna_trials_nn.csv"
    if log_path.exists():
        log_path.unlink()  # log limpio en cada corrida

    print(f"\n[red] Busqueda bayesiana Optuna: {n_trials} trials "
          f"(split unico train/val, pruning MedianPruner, max {max_epochs} epochs/trial)...")
    print(f"[red] Trazabilidad por trial    : {log_path}")
    t0 = time.time()
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False,
        callbacks=[_callback_trazabilidad(log_path)],
    )

    print(f"[red] Busqueda completada en {time.time() - t0:.1f} s "
          f"({(time.time() - t0) / n_trials:.1f} s/trial en promedio)")
    print(f"[red] Mejor RMSE val (normalizado): {study.best_value:.4f}  (trial {study.best_trial.number})")
    print("[red] Mejores hiperparametros:")
    for k, v in study.best_params.items():
        print(f"       {k:<22} : {v}")
    print()

    return study


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def _grafica_curvas(hist_train: list, hist_val: list, hist_lr: list, mejor_epoch: int):
    epochs_r = range(1, len(hist_train) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(epochs_r, hist_train, label="Train",      color="#3C5488", linewidth=1.8)
    ax1.plot(epochs_r, hist_val,   label="Validación", color="#E64B35", linewidth=1.8)
    ax1.axvline(mejor_epoch, color="#00A087", linewidth=1.5, linestyle="--",
                label=f"Mejor epoch = {mejor_epoch}")
    ax1.set_xlim(0)
    ax1.set_ylim(0)
    ax1.set_ylabel("RMSE (escala normalizada)", fontweight="bold", fontsize=11)
    ax1.set_title("Curvas de aprendizaje — Train vs Validación", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.minorticks_on()
    ax1.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax1.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)

    ax2.plot(epochs_r, hist_lr, color="#00A087", linewidth=1.8)
    ax2.set_xlabel("Época", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Learning Rate", fontweight="bold", fontsize=11)
    ax2.set_title("Evolución del Learning Rate (ReduceLROnPlateau)", fontweight="bold", fontsize=11)
    ax2.set_yscale("log")
    ax2.minorticks_on()
    ax2.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax2.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)

    plt.tight_layout()
    path = FIGS_DIR / "curvas_aprendizaje.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[red] Figura guardada : {path}")


def _grafica_obs_vs_pred(y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    lim_max = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="#E64B35", linewidth=1.8, linestyle="--", label="1:1")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("DBO simulada QUAL2K (mg/L)",       fontweight="bold", fontsize=11)
    ax.set_ylabel("DBO predicha Red Neuronal (mg/L)", fontweight="bold", fontsize=11)
    ax.set_title("Observado vs Predicho — conjunto de prueba", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "obs_vs_pred.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[red] Figura guardada : {path}")


def _grafica_curvas_cqr(hist_train: list, hist_val: list, hist_lr: list, mejor_epoch: int):
    epochs_r = range(1, len(hist_train) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(epochs_r, hist_train, label="Train",      color="#3C5488", linewidth=1.8)
    ax1.plot(epochs_r, hist_val,   label="Validación", color="#E64B35", linewidth=1.8)
    ax1.axvline(mejor_epoch, color="#00A087", linewidth=1.5, linestyle="--",
                label=f"Mejor epoch = {mejor_epoch}")
    ax1.set_xlim(0)
    ax1.set_ylim(0)
    ax1.set_ylabel("Pinball loss (escala normalizada)", fontweight="bold", fontsize=11)
    ax1.set_title("Curvas de aprendizaje CQR — Train vs Validación", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.minorticks_on()
    ax1.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax1.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)

    ax2.plot(epochs_r, hist_lr, color="#00A087", linewidth=1.8)
    ax2.set_xlabel("Época", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Learning Rate", fontweight="bold", fontsize=11)
    ax2.set_title("Evolución del Learning Rate (ReduceLROnPlateau)", fontweight="bold", fontsize=11)
    ax2.set_yscale("log")
    ax2.minorticks_on()
    ax2.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax2.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)

    plt.tight_layout()
    path = FIGS_CQR_DIR / "curvas_aprendizaje.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[cqr] Figura guardada : {path}")


def _grafica_obs_vs_pred_cqr(y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    lim_max = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="#E64B35", linewidth=1.8, linestyle="--", label="1:1")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("DBO simulada QUAL2K (mg/L)",                fontweight="bold", fontsize=11)
    ax.set_ylabel("DBO predicha Red Neuronal — mediana (mg/L)", fontweight="bold", fontsize=11)
    ax.set_title("Observado vs Predicho (mediana) — conjunto de prueba", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_CQR_DIR / "obs_vs_pred.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[cqr] Figura guardada : {path}")


def _grafica_ancho_intervalo_cqr(y_pred: np.ndarray, ancho_90: np.ndarray, ancho_95: np.ndarray):
    """
    Muestra cómo varía el ancho del intervalo CQR según la magnitud de la
    predicción — evidencia visual de la ventaja frente a intervalo_conformal
    (margen fijo qhat, una línea horizontal en este mismo gráfico).
    """
    orden = np.argsort(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred[orden], ancho_95[orden], alpha=0.35, s=8, color="#E64B35",
               edgecolors="none", label="Ancho IC 95% (CQR)")
    ax.scatter(y_pred[orden], ancho_90[orden], alpha=0.35, s=8, color="#3C5488",
               edgecolors="none", label="Ancho IC 90% (CQR)")
    ax.set_xlabel("DBO predicha — mediana (mg/L)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Ancho del intervalo (mg/L)",    fontweight="bold", fontsize=11)
    ax.set_title(
        "Ancho del intervalo CQR vs magnitud predicha\n"
        "(adaptativo: no es una banda de ancho constante como el conformal simple)",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_CQR_DIR / "ancho_intervalo_vs_prediccion.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[cqr] Figura guardada : {path}")


def _grafica_residuos(y_true: np.ndarray, y_pred: np.ndarray):
    residuos = y_pred - y_true
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuos, alpha=0.4, s=10, color="#3C5488", edgecolors="none")
    ax.axhline(0, color="#E64B35", linewidth=1.8, linestyle="--")
    ax.set_xlim(0)
    ax.set_xlabel("DBO predicha (mg/L)",          fontweight="bold", fontsize=11)
    ax.set_ylabel("Residuo (pred − real)  mg/L",  fontweight="bold", fontsize=11)
    ax.set_title("Residuos — conjunto de prueba", fontweight="bold", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    path = FIGS_DIR / "residuos.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[red] Figura guardada : {path}")


def _grafica_importancia(imp_mean: pd.Series, imp_std: pd.Series):
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
    ax.set_xlabel("ΔR²  =  R²_base − R²_permutado", fontweight="bold", fontsize=11)
    ax.set_title(
        "Importancia de variables — Red Neuronal (Permutation Importance)\n"
        "(mayor ΔR² → más influyente  ·  negativo → feature irrelevante)",
        fontweight="bold", fontsize=11,
    )
    ax.minorticks_on()
    ax.grid(which="major", axis="x", linestyle="-",  color="#CCCCCC", linewidth=0.7, alpha=0.9)
    ax.grid(which="minor", axis="x", linestyle=":",  color="#E5E5E5", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIGS_DIR / "importancia_variables.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[red] Figura guardada : {path}")


def _exportar_excel(
    metricas_train: dict,
    metricas_test:  dict,
    hist_train:     list,
    hist_val:       list,
    hist_lr:        list,
    y_test:         np.ndarray,
    y_pred:         np.ndarray,
    imp_mean:       pd.Series,
    imp_std:        pd.Series,
    conformal:      dict,
    config:         dict,
    study=None,
):
    excel_path = OUTPUT_DIR / "resultados_nn.xlsx"

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

    df_curva = pd.DataFrame({
        "epoca":      range(1, len(hist_train) + 1),
        "RMSE_train": hist_train,
        "RMSE_val":   hist_val,
        "lr":         hist_lr,
    })

    df_imp = pd.DataFrame({
        "variable":         imp_mean.index,
        "importancia_mean": imp_mean.values,
        "importancia_std":  imp_std.values,
    }).sort_values("importancia_mean", ascending=False)

    df_config = pd.DataFrame([
        {"parametro": "capas",            "valor": str(config["capas"])},
        {"parametro": "epochs",           "valor": config["epochs"]},
        {"parametro": "lr",               "valor": config["lr"]},
        {"parametro": "batch_size",       "valor": config["batch_size"]},
        {"parametro": "dropout",          "valor": config["dropout"]},
        {"parametro": "weight_decay",     "valor": config["weight_decay"]},
        {"parametro": "loss",             "valor": "RMSE"},
        {"parametro": "optimizer",        "valor": "Adam"},
        {"parametro": "scheduler",        "valor": "ReduceLROnPlateau (factor=0.5, patience=15)"},
        {"parametro": "features_scaler",  "valor": "StandardScaler"},
        {"parametro": "target_scaler",    "valor": "StandardScaler"},
        {"parametro": "buscar_optuna",    "valor": config["buscar_optuna"]},
        {"parametro": "n_trials",         "valor": config["n_trials"]},
        {"parametro": "metodo_busqueda",  "valor": config["metodo_busqueda"]},
    ])

    df_conformal = pd.DataFrame([
        {"cobertura_nominal": nivel, **valores}
        for nivel, valores in conformal.items()
    ])

    hojas = [
        ("metricas",      df_metricas),
        ("predicciones",  df_pred),
        ("curva_perdida", df_curva),
        ("importancia",   df_imp),
        ("conformal",     df_conformal),
        ("configuracion", df_config),
    ]

    # Trazabilidad completa de la busqueda de hiperparametros (un trial por fila)
    if study is not None:
        df_trials = study.trials_dataframe()
        df_trials = df_trials.rename(columns=lambda c: c.replace("params_", ""))
        hojas.append(("busqueda_optuna", df_trials))

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in hojas:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for col_idx, col_name in enumerate(df.columns, start=1):
                ws.column_dimensions[
                    ws.cell(row=1, column=col_idx).column_letter
                ].width = max(len(str(col_name)) + 2, 14)

    print(f"[red] Excel guardado  : {excel_path}")


def _exportar_excel_cqr(
    metricas_train: dict,
    metricas_test:  dict,
    hist_train:     list,
    hist_val:       list,
    hist_lr:        list,
    y_test:         np.ndarray,
    pred_test:      np.ndarray,
    intervalos:     dict,
    cqr:            dict,
    cobertura:      dict,
    ancho:          dict,
    config:         dict,
):
    excel_path = OUTPUT_DIR / "resultados_nn_cqr.xlsx"

    df_metricas = pd.DataFrame([
        {"conjunto": "Train (mediana)", **metricas_train},
        {"conjunto": "Test (mediana)",  **metricas_test},
    ])[["conjunto", "r2", "rmse", "mae", "bias"]]
    df_metricas.columns = ["Conjunto", "R²", "RMSE (mg/L)", "MAE (mg/L)", "Sesgo (mg/L)"]

    df_pred = pd.DataFrame({
        "dbo_real_mg_L":        y_test,
        "q025_mg_L":            pred_test[:, 0],
        "q05_mg_L":             pred_test[:, 1],
        "mediana_mg_L":         pred_test[:, 2],
        "q95_mg_L":             pred_test[:, 3],
        "q975_mg_L":            pred_test[:, 4],
        "ic90_lo_mg_L":         intervalos["lo_90"],
        "ic90_hi_mg_L":         intervalos["hi_90"],
        "ic95_lo_mg_L":         intervalos["lo_95"],
        "ic95_hi_mg_L":         intervalos["hi_95"],
    })

    df_curva = pd.DataFrame({
        "epoca":         range(1, len(hist_train) + 1),
        "pinball_train": hist_train,
        "pinball_val":   hist_val,
        "lr":            hist_lr,
    })

    df_cqr = pd.DataFrame([
        {"cobertura_nominal": nivel, **valores,
         "cobertura_empirica_test": cobertura[nivel],
         "ancho_promedio_mg_L":     float(np.mean(ancho[nivel])),
         "ancho_min_mg_L":          float(np.min(ancho[nivel])),
         "ancho_max_mg_L":          float(np.max(ancho[nivel]))}
        for nivel, valores in cqr.items()
    ])

    df_config = pd.DataFrame([
        {"parametro": "capas",            "valor": str(config["capas"])},
        {"parametro": "epochs",           "valor": config["epochs"]},
        {"parametro": "lr",               "valor": config["lr"]},
        {"parametro": "batch_size",       "valor": config["batch_size"]},
        {"parametro": "dropout",          "valor": config["dropout"]},
        {"parametro": "weight_decay",     "valor": config["weight_decay"]},
        {"parametro": "quantiles",        "valor": str(config["quantiles"])},
        {"parametro": "loss",             "valor": "Pinball (ponderada)"},
        {"parametro": "optimizer",        "valor": "Adam"},
        {"parametro": "scheduler",        "valor": "ReduceLROnPlateau (factor=0.5, patience=15)"},
        {"parametro": "features_scaler",  "valor": "StandardScaler"},
        {"parametro": "target_scaler",    "valor": "StandardScaler"},
        {"parametro": "metodo_intervalo",  "valor": "Conformalized Quantile Regression (Romano et al. 2019)"},
    ])

    hojas = [
        ("metricas",      df_metricas),
        ("predicciones",  df_pred),
        ("curva_perdida", df_curva),
        ("cqr",           df_cqr),
        ("configuracion", df_config),
    ]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in hojas:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for col_idx, col_name in enumerate(df.columns, start=1):
                ws.column_dimensions[
                    ws.cell(row=1, column=col_idx).column_letter
                ].width = max(len(str(col_name)) + 2, 14)

    print(f"[cqr] Excel guardado  : {excel_path}")


if __name__ == "__main__":
    # Hiperparametros hallados por la busqueda Optuna previa (ver
    # nn_checkpoint.pt, epoch 40/300, mejor_epoch=36) — se fijan aqui para
    # reanudar el entrenamiento desde el checkpoint en vez de buscar de nuevo.
    entrenar(
        capas        = [32, 128],
        epochs       = 300,
        lr           = 0.0026070247583707684,
        batch_size   = 256,
        dropout      = 0.010292247147901223,
        weight_decay = 5.337032762603957e-06,
        test_size    = 0.20,
        val_size     = 0.10,
        paciencia          = 50,
        seed               = 42,
        n_perm             = 30,
        buscar             = False,
    )
