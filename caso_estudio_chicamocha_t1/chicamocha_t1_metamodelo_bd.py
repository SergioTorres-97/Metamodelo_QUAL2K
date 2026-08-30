"""
chicamocha_t1_metamodelo_bd.py
================================
Genera la base de datos de entrenamiento para el metamodelo de DBO.

Flujo por simulación:
  1. LHS sampling sobre 11 variables sensibles identificadas
     (seleccionadas a partir del SRCC de chicamocha_t1_sensibilidad.py
     sobre carbonaceous_bod_fast, corrida con la topología CABECERA -> PLAYA ABAJO)
  2. Para cada muestra → modifica el JSON base y corre QUAL2K
  3. Extrae DBO (carbonaceous_bod_fast) en los 52 elementos computacionales
  4. Acumula en SQLite (simulaciones_Q2K.db), tabla: simulaciones

Esquema de la tabla 'simulaciones':
  sim_id          INTEGER
  alpha_1         REAL
  kaaa            REAL
  kdc             REAL
  caudal_bypass   REAL
  dbo5_bypass     REAL
  caudal_veolia   REAL
  dbo5_veolia     REAL
  caudal_la_vega  REAL
  dbo5_la_vega    REAL
  caudal_cabecera REAL
  dbo5_cabecera   REAL
  x_km            REAL
  dbo_mg_L        REAL

Variables sensibles (rangos tomados de chicamocha_t1_sensibilidad.py):
  - alpha_1         : Parámetro hidráulico α₁       [0.03  – 0.40]  absoluto
  - kaaa            : Reaireación                   [0.5   – 3.0]   ×cal  relativo
  - kdc             : Oxidación DBO rápida          [0.3   – 3.0]   ×cal  relativo
  - caudal_bypass   : Caudal By-Pass Veolia         [0.05  – 0.50]  m³/s  absoluto
  - dbo5_bypass     : DBO5  By-Pass Veolia          [5     – 600]   mg/L  absoluto
  - caudal_veolia   : Caudal VEOLIA (tratada)       [0.05  – 0.50]  m³/s  absoluto
  - dbo5_veolia     : DBO5  VEOLIA (tratada)        [5     – 600]   mg/L  absoluto
  - caudal_la_vega  : Caudal R. La Vega             [0.001 – 1.0]   m³/s  absoluto
  - dbo5_la_vega    : DBO5  R. La Vega              [1.0   – 80]    mg/L  absoluto
  - caudal_cabecera : Caudal cabecera               [0.005 – 0.300] m³/s  absoluto
  - dbo5_cabecera   : DBO5  cabecera                [0.5   – 50.0]  mg/L  absoluto

Variables no sensibles → fijas en su valor calibrado (JSON base).

Uso:
    python caso_estudio_chicamocha_t1/chicamocha_t1_metamodelo_bd.py
    python caso_estudio_chicamocha_t1/chicamocha_t1_metamodelo_bd.py --n 500 --seed 99
    python caso_estudio_chicamocha_t1/chicamocha_t1_metamodelo_bd.py --n 100 --continuar
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

JSON_BASE  = str(_ROOT / "caso_estudio_chicamocha_t1" / "chicamocha_t1_simulacion.json")
OUTPUT_DIR = str(_ROOT / "resultados" / "chicamocha_t1_metamodelo")
BD_PATH    = str(Path(OUTPUT_DIR) / "simulaciones_Q2K.db")
TABLA      = "simulaciones"
RUNS_DIR   = str(Path(OUTPUT_DIR) / "_runs")

# ---------------------------------------------------------------------------
# Importar utilidades de sensibilidad.py (modificadores de config)
# ---------------------------------------------------------------------------

from scripts.sensibilidad import (
    ParametroSensibilidad,
    _muestrear_lhs,
    _modificar_config,
)
from qual2k.core.model import Q2KModel
from qual2k.processing.json_loader import Q2KJsonLoader

# ---------------------------------------------------------------------------
# Variables sensibles y sus rangos
# ---------------------------------------------------------------------------

PARAMETROS: list[ParametroSensibilidad] = [

    # ── Parámetros hidráulicos ────────────────────────────────────────────────
    ParametroSensibilidad(
        nombre    = "alpha_1",
        categoria = "reach",
        campo     = "alpha_1",
        minimo    = 0.03,
        maximo    = 0.40,
        tipo      = "absoluto",
    ),

    # ── Tasas cinéticas ───────────────────────────────────────────────────────
    ParametroSensibilidad(
        nombre    = "kaaa",
        categoria = "reach_rates",
        campo     = "kaaa",
        minimo    = 0.5,
        maximo    = 3.0,
        tipo      = "relativo",    # factor × 1.82 calibrado
    ),

    ParametroSensibilidad(
        nombre    = "kdc",
        categoria = "reach_rates",
        campo     = "kdc",
        minimo    = 0.3,
        maximo    = 3.0,
        tipo      = "relativo",    # factor × 0.565 calibrado
    ),

    # ── BY-PASS-VEOLIA ────────────────────────────────────────────────────────
    ParametroSensibilidad(
        nombre        = "caudal_bypass",
        categoria     = "fuente",
        campo         = "caudal",
        nombre_fuente = "BY-PASS-VEOLIA AGUAS DE TUNJA S.A. E.S.P.",
        minimo        = 0.050,
        maximo        = 0.500,
        tipo          = "absoluto",
    ),

    ParametroSensibilidad(
        nombre        = "dbo5_bypass",
        categoria     = "fuente",
        campo         = "dbo5",
        nombre_fuente = "BY-PASS-VEOLIA AGUAS DE TUNJA S.A. E.S.P.",
        minimo        = 5.0,
        maximo        = 600.0,
        tipo          = "absoluto",
    ),

    # ── VEOLIA ────────────────────────────────────────────────────────────────
    ParametroSensibilidad(
        nombre        = "caudal_veolia",
        categoria     = "fuente",
        campo         = "caudal",
        nombre_fuente = "VEOLIA AGUAS DE TUNJA S.A. E.S.P.",
        minimo        = 0.050,
        maximo        = 0.500,
        tipo          = "absoluto",
    ),

    ParametroSensibilidad(
        nombre        = "dbo5_veolia",
        categoria     = "fuente",
        campo         = "dbo5",
        nombre_fuente = "VEOLIA AGUAS DE TUNJA S.A. E.S.P.",
        minimo        = 5.0,
        maximo        = 600.0,
        tipo          = "absoluto",
    ),

    # ── R. LA VEGA ────────────────────────────────────────────────────────────
    ParametroSensibilidad(
        nombre        = "caudal_la_vega",
        categoria     = "fuente",
        campo         = "caudal",
        nombre_fuente = "R. LA VEGA ",     # espacio al final — igual que en el JSON
        minimo        = 0.001,
        maximo        = 1.000,
        tipo          = "absoluto",
    ),

    ParametroSensibilidad(
        nombre        = "dbo5_la_vega",
        categoria     = "fuente",
        campo         = "dbo5",
        nombre_fuente = "R. LA VEGA ",     # espacio al final — igual que en el JSON
        minimo        = 1.0,
        maximo        = 80.0,
        tipo          = "absoluto",
    ),

    # ── CABECERA ──────────────────────────────────────────────────────────────
    ParametroSensibilidad(
        nombre          = "caudal_cabecera",
        categoria       = "cabecera",
        campo           = "caudal",
        nombre_estacion = "CABECERA",
        minimo          = 0.005,
        maximo          = 0.300,
        tipo            = "absoluto",
    ),

    ParametroSensibilidad(
        nombre          = "dbo5_cabecera",
        categoria       = "cabecera",
        campo           = "dbo5",
        nombre_estacion = "CABECERA",
        minimo          = 0.5,
        maximo          = 50.0,
        tipo            = "absoluto",
    ),
]

_NOMBRES_VARS = [p.nombre for p in PARAMETROS]

# ---------------------------------------------------------------------------
# SQLite — helpers
# ---------------------------------------------------------------------------

def _crear_tabla(conn: sqlite3.Connection, nueva: bool = False):
    """
    Crea la tabla 'simulaciones'.

    Args:
        nueva: Si True, elimina la tabla existente antes de crearla (modo fresh).
    """
    if nueva:
        conn.execute(f"DROP TABLE IF EXISTS {TABLA}")

    cols_sql = "\n    ".join(
        f"{nombre}  REAL  NOT NULL," for nombre in _NOMBRES_VARS
    )
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLA} (
            sim_id        INTEGER  NOT NULL,
            {cols_sql}
            x_km          REAL     NOT NULL,
            dbo_mg_L      REAL     NOT NULL
        )
    """)
    # Índice sobre sim_id para acelerar el --continuar y futuras queries
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_sim_id ON {TABLA} (sim_id)
    """)
    conn.commit()


def _ultimo_sim_id(conn: sqlite3.Connection) -> int:
    """Retorna el último sim_id guardado, o -1 si la tabla está vacía."""
    cur = conn.execute(f"SELECT MAX(sim_id) FROM {TABLA}")
    val = cur.fetchone()[0]
    return -1 if val is None else int(val)


def _insertar_resultados(
    conn:       sqlite3.Connection,
    sim_id:     int,
    valores_sim: dict[str, float],
    df_result:  pd.DataFrame,
):
    """Inserta todas las filas de una simulación en una sola transacción."""
    cols   = ["sim_id"] + _NOMBRES_VARS + ["x_km", "dbo_mg_L"]
    placeh = ", ".join(["?"] * len(cols))

    filas = []
    for _, row in df_result.iterrows():
        fila = (
            [sim_id]
            + [valores_sim[n] for n in _NOMBRES_VARS]
            + [row["x_km"], row["dbo_mg_L"]]
        )
        filas.append(fila)

    conn.executemany(
        f"INSERT INTO {TABLA} ({', '.join(cols)}) VALUES ({placeh})",
        filas,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Worker  (nivel de módulo → serializable con pickle en Windows/spawn)
# ---------------------------------------------------------------------------

def _worker_simulacion(args: tuple) -> dict:
    """
    Ejecuta una simulación QUAL2K en un proceso worker.

    Recibe  : (sim_id, config_dict, run_dir)
    Retorna : dict con sim_id, exito y (si exito) x_km/dbo_mg_L como listas
              (los DataFrame de pandas no viajan bien entre procesos, listas sí).

    Limpia run_dir antes de retornar, sin importar el resultado.
    """
    sim_id, config_dict, run_dir = args

    import matplotlib
    matplotlib.use("Agg")
    import warnings
    warnings.filterwarnings("ignore")

    try:
        os.makedirs(run_dir, exist_ok=True)
        config_dict["header"]["filedir"] = run_dir

        json_path = os.path.join(run_dir, "config.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(config_dict, fh, indent=2, ensure_ascii=False)

        loader = Q2KJsonLoader(json_path).cargar()

        model = Q2KModel(
            filepath    = loader.header_dict["filedir"],
            header_dict = loader.header_dict,
        )
        model.data_reaches = loader.data_reaches
        model.data_sources = loader.data_sources
        model.data_wq      = loader.data_wq

        if loader.rates_override:
            model.config.actualizar_rates(**loader.rates_override)
        if loader.light_override:
            model.config.actualizar_light(**loader.light_override)

        model.configurar_modelo(
            numelem_default    = loader.numelem_default,
            q_cabecera         = loader.q_cabecera,
            estacion_cabecera  = loader.estacion_cabecera,
            reach_rates_custom = loader.reach_rates_custom,
        )

        model.generar_archivo_q2k()
        model.ejecutar_simulacion()
        model.analizar_resultados(generar_graficas=False)

        wq = model.wq_data_model
        if wq is None or wq.empty:
            return {"sim_id": sim_id, "exito": False, "error": "wq_data_model vacío"}

        df = wq[["Distancia Longitudinal (km)", "carbonaceous_bod_fast"]]
        return {
            "sim_id":   sim_id,
            "exito":    True,
            "x_km":     df["Distancia Longitudinal (km)"].tolist(),
            "dbo_mg_L": df["carbonaceous_bod_fast"].tolist(),
        }

    except Exception as exc:
        return {"sim_id": sim_id, "exito": False, "error": str(exc)}

    finally:
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Generación de la base de datos
# ---------------------------------------------------------------------------

def generar_bd(n: int = 500, seed: int = 42, continuar: bool = False, n_workers: int = 4):
    """
    Genera (o continúa) la base de datos ejecutando N simulaciones LHS.

    Las simulaciones corren en paralelo con multiprocessing.Pool (mismo patrón
    que scripts/sensibilidad.py). La escritura en SQLite se hace siempre desde
    el proceso principal a medida que llegan los resultados.

    Args:
        n         : Número total de simulaciones a generar.
        seed      : Semilla LHS para reproducibilidad.
        continuar : Si True, retoma desde el último sim_id guardado en la BD.
        n_workers : Número de procesos en paralelo.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR,   exist_ok=True)

    # Leer JSON base
    with open(JSON_BASE, encoding="utf-8") as fh:
        config_base = json.load(fh)

    # LHS sobre el espacio completo de N muestras (semilla fija → reproducible)
    muestras = _muestrear_lhs(PARAMETROS, n=n, seed=seed)

    # Abrir / crear la BD SQLite
    conn = sqlite3.connect(BD_PATH)

    # Determinar punto de inicio y modo de tabla
    sim_inicio = 0
    if continuar:
        _crear_tabla(conn, nueva=False)   # conserva datos existentes
        ultimo = _ultimo_sim_id(conn)
        if ultimo >= 0:
            sim_inicio = ultimo + 1
            print(f"[BD] Continuando desde sim_id={sim_inicio} "
                  f"({sim_inicio} simulaciones ya en disco).")
    else:
        _crear_tabla(conn, nueva=True)    # borra y recrea la tabla
        print("[BD] BD nueva — tabla reiniciada.")

    # ── Encabezado ────────────────────────────────────────────────────────────
    print("=" * 65)
    print("GENERACIÓN DE BD — METAMODELO DBO — CHICAMOCHA T1")
    print(f"  BD SQLite             : {BD_PATH}")
    print(f"  Tabla                 : {TABLA}")
    print(f"  Simulaciones a correr : {n - sim_inicio}")
    print(f"  Elementos por tramo   : {config_base['simulacion']['numelem_default']}")
    print(f"  Variables sensibles   : {len(PARAMETROS)}  {[p.nombre for p in PARAMETROS]}")
    print(f"  Workers en paralelo   : {n_workers}")
    print("=" * 65)

    # ── Preparar argumentos por simulación ──────────────────────────────────
    valores_por_sim: dict[int, dict[str, float]] = {}
    args_list = []
    for sim_id in range(sim_inicio, n):
        valores_sim = {p.nombre: float(muestras[p.nombre][sim_id]) for p in PARAMETROS}
        valores_por_sim[sim_id] = valores_sim

        cfg = copy.deepcopy(config_base)
        for param in PARAMETROS:
            _modificar_config(cfg, param, valores_sim[param.nombre])

        run_dir = os.path.join(RUNS_DIR, f"run_{sim_id:04d}")
        args_list.append((sim_id, cfg, run_dir))

    total_a_correr = len(args_list)
    exitosas       = 0
    fallidas       = 0
    t_inicio_bd    = time.time()

    # ── Ejecución en paralelo ────────────────────────────────────────────────
    pool = mp.Pool(processes=n_workers)
    try:
        futures = [(a[0], pool.apply_async(_worker_simulacion, (a,))) for a in args_list]

        for idx, (sim_id, fut) in enumerate(futures):
            try:
                res = fut.get()
            except Exception as exc:
                res = {"sim_id": sim_id, "exito": False, "error": str(exc)}

            completadas = idx + 1
            transcurrido_seg = time.time() - t_inicio_bd
            prom = transcurrido_seg / completadas
            eta_seg = prom * (total_a_correr - completadas)
            eta_str = str(timedelta(seconds=int(eta_seg)))
            transcurrido_str = str(timedelta(seconds=int(transcurrido_seg)))

            print(f"  sim {completadas:>5d}/{total_a_correr}  (id={sim_id:04d})  ",
                  end="", flush=True)

            if res["exito"]:
                df_result = pd.DataFrame({
                    "x_km":     res["x_km"],
                    "dbo_mg_L": res["dbo_mg_L"],
                })
                _insertar_resultados(conn, sim_id, valores_por_sim[sim_id], df_result)
                exitosas += 1

                print(f"OK  ({len(df_result)} pts) | prom {prom:5.2f}s/sim (paralelo) | "
                      f"ETA {eta_str} | transcurrido {transcurrido_str}")
            else:
                fallidas += 1
                print(f"FALLÓ  [{res.get('error', '')}]")
    finally:
        pool.close()
        pool.join()

    # ── Resumen final ─────────────────────────────────────────────────────────
    total_filas = conn.execute(f"SELECT COUNT(*) FROM {TABLA}").fetchone()[0]
    conn.close()

    t_total = time.time() - t_inicio_bd
    prom_final = t_total / total_a_correr if total_a_correr else 0.0

    print("=" * 65)
    print(f"COMPLETADO: {exitosas} exitosas | {fallidas} fallidas")
    print(f"Filas en BD       : {total_filas:,}  "
          f"(≈ {exitosas} sims × {total_filas // max(exitosas, 1)} pts/sim)")
    print(f"Tiempo total      : {str(timedelta(seconds=int(t_total)))}")
    print(f"Tiempo promedio   : {prom_final:.2f} s/sim (paralelo, {n_workers} workers)")
    print(f"Archivo           : {BD_PATH}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chicamocha_t1_metamodelo_bd",
        description="Genera la BD SQLite de entrenamiento para el metamodelo DBO.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=500,
        help="Número de simulaciones LHS (default: 500).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad del LHS (default: 42).",
    )
    p.add_argument(
        "--continuar",
        action="store_true",
        default=False,
        help="Retomar desde el último sim_id guardado en la BD.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Número de procesos en paralelo (default: 4).",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    generar_bd(n=args.n, seed=args.seed, continuar=args.continuar, n_workers=args.n_workers)


