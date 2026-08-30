"""
validar_cqr_honesto.py
========================
Corrige la validacion circular de metamodelo/nn_trainer.py::entrenar_cqr: alli
el q_correction se calibraba y se evaluaba (PICP/MPIW) sobre el MISMO conjunto
de prueba, lo que hace que la cobertura empirica coincida con la nominal por
construccion matematica, no como evidencia de generalizacion.

Este script parte el conjunto de prueba (4000 simulaciones) en dos mitades
independientes por sim_id, con una semilla nueva (no la de train/val/test):
  - calibracion (~2000 sims) -> se usa SOLO para calcular q_correction
  - evaluacion  (~2000 sims) -> nunca vista por la calibracion, se usa SOLO
    para medir PICP (cobertura empirica) y MPIW (ancho promedio) honestos

No reentrena la red (nn_dbo_cqr.pt no cambia): solo recalcula la calibracion
y las metricas. Sobrescribe nn_cqr.json (el que lee app_streamlit.py) con la
correccion calibrada solo en la mitad de calibracion, y guarda el PICP/MPIW
honesto medido en la mitad de evaluacion en nn_cqr_validacion_honesta.json.

Uso:
    python scripts/validar_cqr_honesto.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos      import cargar_datos, FEATURES, TARGET
from metamodelo.metricas   import intervalo_cqr
from metamodelo.nn_trainer import MLPDbo, QUANTILES_CQR

OUTPUT_DIR      = _ROOT / "resultados" / "chicamocha_t1_metamodelo"
MODELO_CQR_PATH = OUTPUT_DIR / "nn_dbo_cqr.pt"
SCALER_X_PATH   = OUTPUT_DIR / "nn_scaler_x_cqr.joblib"
SCALER_Y_PATH   = OUTPUT_DIR / "nn_scaler_y_cqr.joblib"
CQR_PATH        = OUTPUT_DIR / "nn_cqr.json"
HONESTO_PATH    = OUTPUT_DIR / "nn_cqr_validacion_honesta.json"

CAPAS   = [32, 128]
DROPOUT = 0.010292247147901223

# Semillas del split original train/val/test (deben coincidir con entrenar_cqr)
TEST_SIZE = 0.20
VAL_SIZE  = 0.10
SEED_SPLIT_ORIGINAL = 42

# Semilla NUEVA para partir el test en calibracion/evaluacion (independiente
# de la anterior, para no reintroducir ninguna relacion con el split de
# entrenamiento)
SEED_CALIB_EVAL = 123
FRAC_CALIBRACION = 0.5

IDX = {"q025": 0, "q05": 1, "mediana": 2, "q95": 3, "q975": 4}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = cargar_datos()
    sim_ids = df["sim_id"].unique()
    ids_trainval, ids_test = train_test_split(
        sim_ids, test_size=TEST_SIZE, random_state=SEED_SPLIT_ORIGINAL
    )
    val_rel = VAL_SIZE / (1 - TEST_SIZE)
    ids_train, ids_val = train_test_split(
        ids_trainval, test_size=val_rel, random_state=SEED_SPLIT_ORIGINAL
    )
    test_df = df[df["sim_id"].isin(ids_test)]
    print(f"[validar] Test original : {len(test_df):,} filas ({len(ids_test)} sims)")

    # Particion calibracion/evaluacion, independiente y por sim_id
    ids_calib, ids_eval = train_test_split(
        ids_test, test_size=1 - FRAC_CALIBRACION, random_state=SEED_CALIB_EVAL
    )
    calib_df = test_df[test_df["sim_id"].isin(ids_calib)]
    eval_df  = test_df[test_df["sim_id"].isin(ids_eval)]
    print(f"[validar] Calibracion    : {len(calib_df):,} filas ({len(ids_calib)} sims)")
    print(f"[validar] Evaluacion     : {len(eval_df):,} filas ({len(ids_eval)} sims)  <- held-out honesto\n")

    # ── Cargar modelo y scalers ya entrenados (no se reentrena nada) ────────
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    modelo = MLPDbo(n_entrada=len(FEATURES), capas=CAPAS, dropout=DROPOUT,
                     n_salida=len(QUANTILES_CQR)).to(device)
    modelo.load_state_dict(torch.load(MODELO_CQR_PATH, map_location=device))
    modelo.eval()

    def predecir(sub_df):
        X = scaler_x.transform(sub_df[FEATURES].values)
        with torch.no_grad():
            pred_norm = modelo(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
        pred_norm = np.sort(pred_norm, axis=1)  # rearreglo Chernozhukov et al. 2010
        pred = np.column_stack([
            scaler_y.inverse_transform(pred_norm[:, [i]]).ravel()
            for i in range(pred_norm.shape[1])
        ])
        return pred

    pred_calib = predecir(calib_df)
    pred_eval  = predecir(eval_df)
    y_calib = calib_df[TARGET].values
    y_eval  = eval_df[TARGET].values

    resultado_cqr = {}
    resultado_honesto = {}

    for nombre, alpha, i_lo, i_hi, q_lo, q_hi in [
        ("90%", 0.10, IDX["q05"],  IDX["q95"],  0.05,  0.95),
        ("95%", 0.05, IDX["q025"], IDX["q975"], 0.025, 0.975),
    ]:
        # 1) Calibrar SOLO con la mitad de calibracion
        cal = intervalo_cqr(y_calib, pred_calib[:, i_lo], pred_calib[:, i_hi], alpha=alpha)
        resultado_cqr[nombre] = {**cal, "cuantil_lo": q_lo, "cuantil_hi": q_hi}

        # 2) Evaluar PICP/MPIW SOLO en la mitad de evaluacion (held-out real)
        q_correction = cal["q_correction"]
        lo = np.clip(pred_eval[:, i_lo] - q_correction, 0, None)
        hi = pred_eval[:, i_hi] + q_correction
        hi = np.maximum(hi, lo)  # piso de ancho no-negativo (igual que entrenar_cqr)

        picp  = float(np.mean((y_eval >= lo) & (y_eval <= hi)))
        ancho = hi - lo
        resultado_honesto[nombre] = {
            "picp": picp,
            "mpiw_mg_L": float(ancho.mean()),
            "ancho_min_mg_L": float(ancho.min()),
            "ancho_max_mg_L": float(ancho.max()),
            "n_calibracion": len(y_calib),
            "n_evaluacion": len(y_eval),
        }
        print(f"[validar] {nombre}: q_correction (calib) = {q_correction:+.4f} mg/L  |  "
              f"PICP (eval, held-out) = {picp:.4%}  |  MPIW (eval) = {ancho.mean():.2f} mg/L "
              f"(min {ancho.min():.2f} / max {ancho.max():.2f})")

    with open(CQR_PATH, "w", encoding="utf-8") as fh:
        json.dump(resultado_cqr, fh, indent=2)
    print(f"\n[validar] nn_cqr.json recalibrado (solo con la mitad de calibracion): {CQR_PATH}")

    with open(HONESTO_PATH, "w", encoding="utf-8") as fh:
        json.dump(resultado_honesto, fh, indent=2)
    print(f"[validar] PICP/MPIW honesto guardado: {HONESTO_PATH}")


if __name__ == "__main__":
    main()
