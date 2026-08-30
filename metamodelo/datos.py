"""
datos.py
=========
Carga y prepara los datos desde la BD SQLite para entrenar el metamodelo.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------

_ROOT    = Path(__file__).resolve().parent.parent
BD_PATH  = str(_ROOT / "resultados" / "t_rio_jordan_metamodelo" / "simulaciones_Q2K.db")
TABLA    = "simulaciones"

FEATURES = [
    "alpha_1",
    "kaaa",
    "kdc",
    "caudal_bypass",
    "dbo5_bypass",
    "caudal_veolia",
    "dbo5_veolia",
    "caudal_la_vega",
    "dbo5_la_vega",
    "caudal_cabecera",
    "dbo5_cabecera",
    "x_km",
]
TARGET = "dbo_mg_L"


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def cargar_datos(bd_path: str = BD_PATH, tabla: str = TABLA) -> pd.DataFrame:
    """
    Carga toda la tabla de simulaciones desde SQLite.

    Returns:
        DataFrame con columnas: sim_id | FEATURES | TARGET
    """
    conn = sqlite3.connect(bd_path)
    df   = pd.read_sql(f"SELECT * FROM {tabla}", conn)
    conn.close()

    print(f"[datos] Filas cargadas : {len(df):,}")
    print(f"[datos] Simulaciones   : {df['sim_id'].nunique():,}")
    print(f"[datos] Puntos/sim     : {len(df) // df['sim_id'].nunique()}")
    return df


def preparar_sets(
    df:          pd.DataFrame,
    test_size:   float = 0.20,
    seed:        int   = 42,
    por_sim_id:  bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide en train/test y retorna X_train, X_test, y_train, y_test.

    Args:
        df          : DataFrame completo (salida de cargar_datos).
        test_size   : Fracción reservada para test (default 20 %).
        seed        : Semilla para reproducibilidad.
        por_sim_id  : Si True, la división se hace por sim_id (no por fila),
                      evitando que puntos de la misma simulación queden en
                      train y test a la vez — split más honesto.
    """
    if por_sim_id:
        sim_ids = df["sim_id"].unique()
        ids_train, ids_test = train_test_split(
            sim_ids, test_size=test_size, random_state=seed
        )
        train = df[df["sim_id"].isin(ids_train)]
        test  = df[df["sim_id"].isin(ids_test)]
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=seed)

    X_train = train[FEATURES]
    X_test  = test[FEATURES]
    y_train = train[TARGET]
    y_test  = test[TARGET]

    print(f"[datos] Train : {len(train):,} filas  ({train['sim_id'].nunique()} sims)")
    print(f"[datos] Test  : {len(test):,} filas  ({test['sim_id'].nunique()} sims)")

    return X_train, X_test, y_train, y_test
