"""
sensibilidad.py  –  Análisis de sensibilidad LHS / SRCC para QUAL2K
====================================================================

Muestrea N combinaciones de parámetros usando Latin Hypercube Sampling (LHS),
corre QUAL2K en paralelo y calcula el SRCC (Spearman Rank Correlation
Coefficient) entre cada parámetro y la media espacial de cada variable simulada.

¿Por qué LHS?
-------------
Con Monte Carlo puro los valores se agrupan aleatoriamente, dejando zonas del
espacio de parámetros sin cubrir. LHS divide el rango de cada parámetro en N
intervalos equiprobables y toma exactamente una muestra por intervalo, rotando
aleatoriamente entre parámetros. El resultado:
  - Cobertura uniforme garantizada con N mucho menor que Monte Carlo puro.
  - SRCC más estable con 50–100 corridas en vez de 200–500.

Distribución asumida: uniforme [minimo, maximo]
-----------------------------------------------
Cuando no se conoce la distribución real de un parámetro, la distribución
uniforme es la hipótesis más honesta: solo se asume que el parámetro está
dentro de un rango plausible, sin dar más peso a ningún valor interior.

Parámetros soportados
---------------------
  categoria="reach"    → alpha_1, beta_1, alpha_2, beta_2
  categoria="fuente"   → dbo5, oxigeno_disuelto, temperatura, pH, caudal
                         (de un vertimiento identificado por nombre_fuente)
  categoria="cabecera" → caudal, dbo5, oxigeno_disuelto, temperatura, pH
                         (en la estación de cabecera de wq_data)

Uso básico
----------
    from scripts.sensibilidad import ParametroSensibilidad, analisis_sensibilidad

    parametros = [
        ParametroSensibilidad(
            nombre    = "alpha_1",
            categoria = "reach",
            campo     = "alpha_1",
            minimo    = 0.10,
            maximo    = 0.80,
            tipo      = "absoluto",
        ),
        ParametroSensibilidad(
            nombre        = "dbo5_bypass",
            categoria     = "fuente",
            campo         = "dbo5",
            nombre_fuente = "By-Pass PTAR Rio de Oro",
            minimo        = 50.0,
            maximo        = 400.0,
            tipo          = "absoluto",
        ),
        ParametroSensibilidad(
            nombre          = "caudal_cabecera",
            categoria       = "cabecera",
            campo           = "caudal",
            nombre_estacion = "CABECERA",
            minimo          = 0.5,
            maximo          = 5.0,
            tipo            = "absoluto",
        ),
    ]

    df_srcc = analisis_sensibilidad(
        json_base    = "examples/chicamocha_simulacion.json",
        parametros   = parametros,
        n            = 100,
        output_dir   = "resultados/sensibilidad",
        n_workers    = 4,
        limpiar_runs = True,
    )
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from scipy.stats.qmc import LatinHypercube

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Campos permitidos por categoría
# ---------------------------------------------------------------------------

_CAMPOS_REACH    = {"alpha_1", "beta_1", "alpha_2", "beta_2"}
_CAMPOS_FUENTE   = {"dbo5", "oxigeno_disuelto", "temperatura", "pH", "caudal"}
_CAMPOS_CABECERA = {"caudal", "dbo5", "oxigeno_disuelto", "temperatura", "pH"}
# Nombres cortos aceptados en reach_rates (aliases de json_loader + kaaa)
_CAMPOS_REACH_RATES = {
    "kaaa",
    "vss", "khc", "kdcs", "kdc",
    "khn", "von", "kn",
    "ki", "vdi",
    "khp", "vop", "vip",
    "kga", "krea", "kexa", "kdea", "va",
    "kgaF", "kreaF", "kexaF", "kdeaF",
    "kdt", "vdt", "ffast",
}

# ---------------------------------------------------------------------------
# Variables de salida de QUAL2K incluidas en el análisis
# ---------------------------------------------------------------------------

_VARIABLES_SALIDA: Dict[str, str] = {
    "dissolved_oxygen":      "Oxígeno Disuelto (mg/L)",
    "carbonaceous_bod_fast": "DBO Rápida (mg/L)",
    "water_temp_c":          "Temperatura (°C)",
    "ammonium":              "Amonio (µg/L)",
    "nitrate":               "Nitrato (µg/L)",
}


# ===========================================================================
# ParametroSensibilidad
# ===========================================================================

@dataclass
class ParametroSensibilidad:
    """
    Define un parámetro a perturbar en el análisis de sensibilidad.

    El muestreo es uniforme en [minimo, maximo] con LHS.
    Solo necesitás indicar el rango plausible del parámetro.

    Atributos comunes
    -----------------
    nombre          : Identificador único (columna en la tabla de resultados).
    categoria       : "reach" | "fuente" | "cabecera" | "reach_rates"
    campo           : Campo JSON a modificar.
                        reach       → "alpha_1", "beta_1", "alpha_2", "beta_2"
                        fuente      → "dbo5", "oxigeno_disuelto", "temperatura",
                                      "pH", "caudal"
                        cabecera    → "caudal", "dbo5", "oxigeno_disuelto",
                                      "temperatura", "pH"
                        reach_rates → "kaaa", "kdc", "kn", "khp", "kdt",
                                      "kdcs", "vss", "khc", …  (nombres cortos
                                      de la sección reach_rates del JSON)
    minimo          : Límite inferior del rango uniforme.
    maximo          : Límite superior del rango uniforme.
    tipo            : "absoluto" → el valor muestreado REEMPLAZA al base.
                      "relativo" → el valor muestreado MULTIPLICA al base.
                                   (minimo=0.5, maximo=1.5 → ±50% del calibrado)

    Atributos por categoría
    -----------------------
    nombre_fuente   : (fuente)   nombre exacto del vertimiento en el JSON.
    tramos          : (reach)    índices de tramos a modificar — None = todos.
    nombre_estacion : (cabecera) nombre de la estación en wq_data — None = primera.
    """
    nombre:    str
    categoria: str
    campo:     str
    minimo:    float
    maximo:    float
    tipo:      str            = "absoluto"
    # Por categoría
    nombre_fuente:   Optional[str]       = None
    tramos:          Optional[List[int]] = None
    nombre_estacion: Optional[str]       = None

    def validar(self):
        """Lanza ValueError si la configuración es inconsistente."""
        if self.minimo >= self.maximo:
            raise ValueError(
                f"[{self.nombre}] minimo ({self.minimo}) debe ser menor que "
                f"maximo ({self.maximo})."
            )
        if self.categoria == "reach" and self.campo not in _CAMPOS_REACH:
            raise ValueError(
                f"[{self.nombre}] campo='{self.campo}' no válido para 'reach'. "
                f"Permitidos: {sorted(_CAMPOS_REACH)}"
            )
        if self.categoria == "fuente":
            if self.campo not in _CAMPOS_FUENTE:
                raise ValueError(
                    f"[{self.nombre}] campo='{self.campo}' no válido para 'fuente'. "
                    f"Permitidos: {sorted(_CAMPOS_FUENTE)}"
                )
            if self.nombre_fuente is None:
                raise ValueError(
                    f"[{self.nombre}] Se requiere 'nombre_fuente' para "
                    f"categoria='fuente'."
                )
        if self.categoria == "cabecera" and self.campo not in _CAMPOS_CABECERA:
            raise ValueError(
                f"[{self.nombre}] campo='{self.campo}' no válido para 'cabecera'. "
                f"Permitidos: {sorted(_CAMPOS_CABECERA)}"
            )
        if self.categoria == "reach_rates" and self.campo not in _CAMPOS_REACH_RATES:
            raise ValueError(
                f"[{self.nombre}] campo='{self.campo}' no válido para 'reach_rates'. "
                f"Permitidos: {sorted(_CAMPOS_REACH_RATES)}"
            )
        if self.categoria not in {"reach", "fuente", "cabecera", "reach_rates"}:
            raise ValueError(
                f"[{self.nombre}] categoria='{self.categoria}' no reconocida. "
                f"Usar 'reach', 'fuente', 'cabecera' o 'reach_rates'."
            )
        if self.tipo not in {"absoluto", "relativo"}:
            raise ValueError(
                f"[{self.nombre}] tipo='{self.tipo}' no reconocido. "
                f"Usar 'absoluto' o 'relativo'."
            )


# ===========================================================================
# LHS — muestreo
# ===========================================================================

def _muestrear_lhs(
    parametros: List[ParametroSensibilidad],
    n:          int,
    seed:       int,
) -> Dict[str, np.ndarray]:
    """
    Genera N muestras con Latin Hypercube Sampling para todos los parámetros.

    Retorna un dict {nombre_param: array de N valores escalados a [min, max]}.
    """
    d = len(parametros)
    sampler   = LatinHypercube(d=d, seed=seed)
    lhs_unit  = sampler.random(n=n)          # shape (n, d) — valores en [0, 1]

    muestras: Dict[str, np.ndarray] = {}
    for j, p in enumerate(parametros):
        # Escalar de [0, 1] a [minimo, maximo]
        muestras[p.nombre] = p.minimo + lhs_unit[:, j] * (p.maximo - p.minimo)

    return muestras


# ===========================================================================
# Modificadores del config JSON
# ===========================================================================

def _aplicar_valor(base: float, valor: float, tipo: str) -> float:
    if tipo == "relativo":
        return base * valor
    return valor   # absoluto


def _mod_reach(config: dict, param: ParametroSensibilidad, valor: float):
    reaches = config.get("reaches", [])
    if not reaches:
        raise ValueError("El JSON no tiene la sección 'reaches'.")
    indices = param.tramos if param.tramos is not None else list(range(len(reaches)))
    for i in indices:
        if i >= len(reaches):
            raise IndexError(
                f"[{param.nombre}] Índice de tramo {i} fuera de rango "
                f"(total={len(reaches)})."
            )
        base = float(reaches[i].get(param.campo, 1.0) or 1.0)
        reaches[i][param.campo] = _aplicar_valor(base, valor, param.tipo)


def _mod_fuente(config: dict, param: ParametroSensibilidad, valor: float):
    sources    = config.get("sources", [])
    encontrado = False
    for src in sources:
        if src.get("nombre") == param.nombre_fuente:
            base = float(src.get(param.campo, 0.0) or 0.0)
            src[param.campo] = _aplicar_valor(base, valor, param.tipo)
            encontrado = True
    if not encontrado:
        raise ValueError(
            f"[{param.nombre}] No se encontró la fuente '{param.nombre_fuente}' "
            f"en 'sources'."
        )


def _mod_cabecera(config: dict, param: ParametroSensibilidad, valor: float):
    wq_data = config.get("wq_data", [])
    if not wq_data:
        raise ValueError("El JSON no tiene la sección 'wq_data'.")
    if param.nombre_estacion:
        estacion = next(
            (s for s in wq_data if s.get("nombre_estacion") == param.nombre_estacion),
            None,
        )
        if estacion is None:
            raise ValueError(
                f"[{param.nombre}] Estación '{param.nombre_estacion}' "
                f"no encontrada en 'wq_data'."
            )
    else:
        estacion = wq_data[0]

    base   = float(estacion.get(param.campo, 0.0) or 0.0)
    nuevo  = _aplicar_valor(base, valor, param.tipo)
    estacion[param.campo] = nuevo

    if param.campo == "caudal":
        config.setdefault("simulacion", {})["q_cabecera"] = nuevo


def _mod_reach_rates(config: dict, param: ParametroSensibilidad, valor: float):
    rr = config.get("reach_rates")
    if rr is None:
        raise ValueError(
            f"[{param.nombre}] El JSON no tiene la sección 'reach_rates'. "
            f"Agregá la sección con los valores calibrados por tramo."
        )
    campo = param.campo
    if campo not in rr:
        raise KeyError(
            f"[{param.nombre}] La tasa '{campo}' no existe en 'reach_rates'. "
            f"Claves disponibles: {sorted(k for k in rr if not k.startswith('_'))}"
        )
    valores = rr[campo]
    # Scalar en el JSON → convertir a lista
    n = len(config.get("reaches", []))
    if not isinstance(valores, list):
        valores = [float(valores)] * n
        rr[campo] = valores

    indices = param.tramos if param.tramos is not None else list(range(len(valores)))
    for i in indices:
        if i >= len(valores):
            raise IndexError(
                f"[{param.nombre}] Índice de tramo {i} fuera de rango "
                f"(total={len(valores)})."
            )
        base = float(valores[i]) if valores[i] is not None else 0.0
        valores[i] = _aplicar_valor(base, valor, param.tipo)


def _modificar_config(config: dict, param: ParametroSensibilidad, valor: float):
    if param.categoria == "reach":
        _mod_reach(config, param, valor)
    elif param.categoria == "fuente":
        _mod_fuente(config, param, valor)
    elif param.categoria == "cabecera":
        _mod_cabecera(config, param, valor)
    elif param.categoria == "reach_rates":
        _mod_reach_rates(config, param, valor)


# ===========================================================================
# Worker  (nivel de módulo → serializable con pickle en Windows)
# ===========================================================================

def _worker_corrida(args: tuple):
    """
    Ejecuta una corrida de QUAL2K con la configuración pre-modificada.

    Recibe  : (run_id, config_dict, run_dir)
    Retorna : dict con run_id, exito y medias espaciales de variables de salida.

    Usa Q2KJsonLoader + Q2KModel directamente (sin depender de run_from_json).
    """
    run_id, config_dict, run_dir = args

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

        from qual2k.processing.json_loader import Q2KJsonLoader
        from qual2k.core.model import Q2KModel

        loader = Q2KJsonLoader(json_path).cargar()

        model = Q2KModel(
            filepath=loader.header_dict["filedir"],
            header_dict=loader.header_dict,
        )
        model.data_reaches = loader.data_reaches
        model.data_sources = loader.data_sources
        model.data_wq      = loader.data_wq

        if loader.rates_override:
            model.config.actualizar_rates(**loader.rates_override)
        if loader.light_override:
            model.config.actualizar_light(**loader.light_override)

        model.configurar_modelo(
            numelem_default=loader.numelem_default,
            q_cabecera=loader.q_cabecera,
            estacion_cabecera=loader.estacion_cabecera,
            reach_rates_custom=loader.reach_rates_custom,
        )

        model.generar_archivo_q2k()
        model.ejecutar_simulacion()

        data_exp = model.analizar_resultados(generar_graficas=False)

        if data_exp is None or data_exp.empty:
            return {"run_id": run_id, "exito": False, "medias": {}}

        cols      = [c for c in _VARIABLES_SALIDA if c in data_exp.columns]
        medias    = {c: float(data_exp[c].mean()) for c in cols}
        dist_col  = "Distancia Longitudinal (km)"
        perfil    = {c: data_exp[c].tolist() for c in cols}
        distancias = (
            data_exp[dist_col].tolist()
            if dist_col in data_exp.columns
            else list(range(len(data_exp)))
        )
        return {
            "run_id":    run_id,
            "exito":     True,
            "medias":    medias,
            "perfil":    perfil,
            "distancias": distancias,
        }

    except Exception as exc:
        return {"run_id": run_id, "exito": False, "medias": {}, "error": str(exc)}


def _limpiar_run(run_dir: str):
    """
    Limpia el directorio de una corrida tras extraer sus resultados.
    Conserva únicamente config.json (útil para debugging).
    Elimina todo lo demás: ejecutable, archivos FORTRAN y subdirectorios.
    """
    import shutil
    run_path = Path(run_dir)
    if not run_path.exists():
        return
    for entry in run_path.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)   # ej. resultados/
            elif entry.name != "config.json":
                entry.unlink()         # .q2k, .out, .DAT, .exe, etc.
        except Exception:
            pass


# ===========================================================================
# Función principal
# ===========================================================================

def analisis_sensibilidad(
    json_base:    str,
    parametros:   List[ParametroSensibilidad],
    n:            int  = 100,
    output_dir:   str  = "resultados/sensibilidad",
    seed:         int  = 42,
    n_workers:    int  = 4,
    limpiar_runs: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta N corridas LHS de QUAL2K y calcula el SRCC de sensibilidad.

    Parámetros
    ----------
    json_base    : Ruta al JSON de simulación base.
    parametros   : Lista de ParametroSensibilidad (solo necesitan minimo/maximo).
    n            : Número de corridas (con LHS, 50–100 suele ser suficiente).
    output_dir   : Directorio de salida para resultados y gráficas.
    seed         : Semilla aleatoria para reproducibilidad.
    n_workers    : Número de procesos en paralelo.
    limpiar_runs : Si True, borra .q2k/.out/etc. tras cada corrida.

    Retorna
    -------
    DataFrame SRCC con índice = parámetros y columnas = variables de salida.
    """
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore")

    # ── Validar ──────────────────────────────────────────────────────────
    print("Validando parámetros...")
    for p in parametros:
        p.validar()
    print(f"  OK — {len(parametros)} parámetros, {n} corridas LHS, "
          f"{n_workers} worker(s).\n")

    # ── Cargar JSON base ─────────────────────────────────────────────────
    with open(json_base, encoding="utf-8") as fh:
        config_base = json.load(fh)

    # ── LHS ──────────────────────────────────────────────────────────────
    muestras = _muestrear_lhs(parametros, n, seed)

    print("Rangos muestreados (LHS uniforme):")
    for p in parametros:
        v = muestras[p.nombre]
        print(f"  {p.nombre:<35}  [{p.minimo:.4g}, {p.maximo:.4g}]  "
              f"media_obs={v.mean():.4g}")
    print()

    _graficar_lhs(parametros, muestras, output_dir)

    # ── Preparar argumentos por corrida ──────────────────────────────────
    args_list = []
    for i in range(n):
        cfg = copy.deepcopy(config_base)
        for p in parametros:
            _modificar_config(cfg, p, float(muestras[p.nombre][i]))
        run_dir = os.path.join(output_dir, "runs", f"run_{i:03d}")
        args_list.append((i, cfg, run_dir))

    # ── Ejecución en paralelo ─────────────────────────────────────────────
    print(f"Lanzando {n} corridas con {n_workers} worker(s)...")
    print("=" * 70)

    resultados_raw = []
    pool = mp.Pool(processes=n_workers)
    try:
        futures = [pool.apply_async(_worker_corrida, (a,)) for a in args_list]
        for i, fut in enumerate(futures):
            try:
                res = fut.get(timeout=600)
                resultados_raw.append(res)
                estado  = "✓" if res["exito"] else "✗"
                detalle = f"  [{res.get('error', '')}]" if not res["exito"] else ""
                print(f"  {estado} Corrida {i:03d}{detalle}")
            except mp.TimeoutError:
                print(f"  ✗ Corrida {i:03d} — timeout")
                resultados_raw.append({"run_id": i, "exito": False, "medias": {}})
            finally:
                if limpiar_runs:
                    _limpiar_run(os.path.join(output_dir, "runs", f"run_{i:03d}"))
    finally:
        pool.close()
        pool.join()

    # ── Construir tabla de resultados ─────────────────────────────────────
    exitosas = sum(1 for r in resultados_raw if r["exito"])
    print(f"\nCorridas completadas: {exitosas}/{n}")

    filas = []
    for res in resultados_raw:
        if not res["exito"]:
            continue
        i    = res["run_id"]
        fila = {"run_id": i}
        for p in parametros:
            fila[p.nombre] = float(muestras[p.nombre][i])
        fila.update(res["medias"])
        filas.append(fila)

    if not filas:
        print("[Error] Ninguna corrida produjo resultados.")
        return pd.DataFrame()

    df = pd.DataFrame(filas)
    csv_path = os.path.join(output_dir, "resultados_por_corrida.csv")
    df.to_csv(csv_path, index=False)
    print(f"Resultados por corrida: {csv_path}\n")

    # ── SRCC global (media espacial) ──────────────────────────────────────
    nombres = [p.nombre for p in parametros]
    df_srcc = _calcular_srcc(df, nombres, output_dir)

    # ── SRCC espacial (por sección) ───────────────────────────────────────
    distancias_ref: Optional[List[float]] = None
    perfiles_por_var: Dict[str, List[list]] = {c: [] for c in _VARIABLES_SALIDA}
    run_ids_exitosos: List[int] = []

    for res in resultados_raw:
        if not res["exito"] or "perfil" not in res:
            continue
        rid = res["run_id"]
        if distancias_ref is None and "distancias" in res:
            distancias_ref = res["distancias"]
        primer_var = next(iter(res["perfil"]), None)
        if primer_var is None or len(res["perfil"][primer_var]) != len(distancias_ref):
            continue
        run_ids_exitosos.append(rid)
        for c in _VARIABLES_SALIDA:
            if c in res["perfil"]:
                perfiles_por_var[c].append(res["perfil"][c])

    if distancias_ref and run_ids_exitosos:
        params_vals = {
            p.nombre: [float(muestras[p.nombre][i]) for i in run_ids_exitosos]
            for p in parametros
        }
        _calcular_srcc_espacial(
            perfiles_por_var,
            params_vals,
            [p.nombre for p in parametros],
            distancias_ref,
            output_dir,
        )

    return df_srcc


# ===========================================================================
# SRCC
# ===========================================================================

def _calcular_srcc(
    df:             pd.DataFrame,
    nombres_params: List[str],
    output_dir:     str,
) -> pd.DataFrame:
    """
    Calcula el SRCC entre cada parámetro y la media espacial de cada variable.

    |SRCC| ≈ 1  → el parámetro domina esa variable
    |SRCC| ≈ 0  → el parámetro no influye
    SRCC  > 0   → relación directa  (↑ parámetro → ↑ variable)
    SRCC  < 0   → relación inversa  (↑ parámetro → ↓ variable)
    """
    cols_sal = [c for c in _VARIABLES_SALIDA if c in df.columns]
    params   = [p for p in nombres_params    if p in df.columns]

    if not params or not cols_sal:
        print("[SRCC] No hay datos suficientes.")
        return pd.DataFrame()

    registros = []
    for param in params:
        fila = {"parametro": param}
        for col in cols_sal:
            datos = df[[param, col]].dropna()
            if len(datos) < 4:
                fila[col] = float("nan")
            else:
                rho, _ = spearmanr(datos[param], datos[col])
                fila[col] = round(rho, 4)
        registros.append(fila)

    df_srcc = pd.DataFrame(registros).set_index("parametro")

    csv_path = os.path.join(output_dir, "srcc_sensibilidad.csv")
    df_srcc.to_csv(csv_path)
    print(f"SRCC guardado: {csv_path}")

    etiquetas = {c: lbl for c, lbl in _VARIABLES_SALIDA.items() if c in cols_sal}
    _graficar_dispersiones(df, params, etiquetas, output_dir)
    _graficar_tornado(df_srcc, etiquetas, output_dir)
    _graficar_heatmap(df_srcc, etiquetas, output_dir)

    return df_srcc


# ===========================================================================
# Gráficas
# ===========================================================================

def _graficar_lhs(
    parametros: List[ParametroSensibilidad],
    muestras:   Dict[str, np.ndarray],
    output_dir: str,
):
    """
    Visualiza la cobertura del LHS:
      - Izquierda: histograma de cada parámetro (debe ser uniforme).
      - Derecha:   scatter matrix de los primeros 4 parámetros para verificar
                   que no hay correlación entre ellos (independencia).
    """
    n_params = len(parametros)
    if n_params == 0:
        return

    # ── Histogramas ───────────────────────────────────────────────────────
    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)

    for k, p in enumerate(parametros):
        ax   = axes[k // ncols][k % ncols]
        vals = muestras[p.nombre]
        ax.hist(vals, bins=max(8, len(vals) // 5), color="#3C5488",
                edgecolor="white", alpha=0.85, density=False)
        ax.axvline(vals.min(),  color="#8491B4", lw=1,   ls="--", label=f"min={p.minimo:.4g}")
        ax.axvline(vals.max(),  color="#8491B4", lw=1,   ls="--", label=f"max={p.maximo:.4g}")
        ax.axvline(vals.mean(), color="#E64B35", lw=1.5, ls="-",  label=f"μ={vals.mean():.4g}")
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_title(p.nombre, fontsize=9, fontweight="bold")
        ax.set_xlabel("Valor", fontsize=8, fontweight="bold")
        ax.set_ylabel("Frecuencia", fontsize=8, fontweight="bold")
        ax.legend(fontsize=6, framealpha=0.9)
        ax.minorticks_on()
        ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.6, alpha=0.9)
        ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.3, alpha=0.7)

    for k in range(n_params, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle("Cobertura LHS — distribución uniforme por parámetro",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "lhs_histogramas.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Histogramas LHS: {path}")

    # ── Scatter matrix (máximo 5 parámetros para legibilidad) ─────────────
    params_scatter = parametros[:5]
    m = len(params_scatter)
    if m >= 2:
        fig2, axes2 = plt.subplots(m, m, figsize=(2.5 * m, 2.5 * m))
        for r in range(m):
            for c in range(m):
                ax = axes2[r][c]
                xv = muestras[params_scatter[c].nombre]
                yv = muestras[params_scatter[r].nombre]
                if r == c:
                    ax.hist(xv, bins=10, color="#3C5488", edgecolor="white", alpha=0.85)
                else:
                    ax.scatter(xv, yv, s=6, alpha=0.5, color="#3C5488", edgecolors="none")
                ax.set_xlim(0)
                ax.set_ylim(0)
                ax.minorticks_on()
                ax.grid(which="major", linestyle="-", color="#CCCCCC", linewidth=0.5, alpha=0.8)
                ax.grid(which="minor", linestyle=":", color="#E5E5E5", linewidth=0.3, alpha=0.6)
                if c == 0:
                    ax.set_ylabel(params_scatter[r].nombre, fontsize=6, fontweight="bold")
                if r == m - 1:
                    ax.set_xlabel(params_scatter[c].nombre, fontsize=6, fontweight="bold")
                ax.tick_params(labelsize=5)

        fig2.suptitle("Independencia entre parámetros (LHS)",
                      fontsize=10, fontweight="bold", y=1.01)
        fig2.tight_layout()
        path2 = os.path.join(output_dir, "lhs_scatter_matrix.png")
        fig2.savefig(path2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Scatter matrix LHS: {path2}")


def _graficar_dispersiones(
    df:         pd.DataFrame,
    params:     List[str],
    etiquetas:  Dict[str, str],
    output_dir: str,
):
    """
    Scatter: valor del parámetro (eje X) vs media espacial de la variable
    simulada (eje Y). Una figura por parámetro, subplots por variable.
    """
    cols_sal = [c for c in etiquetas if c in df.columns]
    if not cols_sal:
        return
    ncols = min(3, len(cols_sal))
    nrows = (len(cols_sal) + ncols - 1) // ncols

    for param in params:
        if param not in df.columns:
            continue
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
        for j, col in enumerate(cols_sal):
            ax    = axes[j // ncols][j % ncols]
            datos = df[[param, col]].dropna()
            x, y  = datos[param].values, datos[col].values
            ax.scatter(x, y, alpha=0.5, s=20, color="#3C5488", edgecolors="none")

            if len(x) >= 4:
                slope, intercept, *_ = linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, intercept + slope * x_line,
                        color="#E64B35", lw=1.5, linestyle="--")
                rho, _ = spearmanr(x, y)
                ax.annotate(
                    f"SRCC = {rho:+.3f}",
                    xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=8, color="#E64B35",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                )

            ax.set_xlim(0)
            ax.set_ylim(0)
            ax.set_xlabel(param, fontsize=8, fontweight="bold")
            ax.set_ylabel(etiquetas[col], fontsize=8, fontweight="bold")
            ax.set_title(etiquetas[col], fontsize=9, fontweight="bold")
            ax.minorticks_on()
            ax.grid(which="major", linestyle="-",  color="#CCCCCC", linewidth=0.6, alpha=0.9)
            ax.grid(which="minor", linestyle=":",  color="#E5E5E5", linewidth=0.3, alpha=0.7)

        for j in range(len(cols_sal), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.suptitle(f"Sensibilidad: {param}", fontsize=11, fontweight="bold", y=1.01)
        fig.tight_layout()
        fname = param.replace(" ", "_").replace("/", "-")
        path  = os.path.join(output_dir, f"scatter_{fname}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Scatter plots guardados en: {output_dir}")


def _graficar_tornado(
    df_srcc:    pd.DataFrame,
    etiquetas:  Dict[str, str],
    output_dir: str,
):
    """Tornado plot por variable. Barras ordenadas por |SRCC| descendente."""
    cols_sal = [c for c in etiquetas if c in df_srcc.columns]

    for col in cols_sal:
        serie = df_srcc[col].dropna().sort_values(key=abs)
        if serie.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(serie))))
        colores = ["#3C5488" if v < 0 else "#E64B35" for v in serie.values]
        ax.barh(serie.index, serie.values, color=colores, edgecolor="white", height=0.65)
        ax.axvline(0, color="#333333", lw=1.0)
        ax.set_xlabel("SRCC  (Spearman Rank Correlation Coefficient)",
                      fontsize=10, fontweight="bold")
        ax.set_title(f"Diagrama tornado — {etiquetas[col]}",
                     fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlim(-1, 1)
        ax.minorticks_on()
        ax.grid(which="major", axis="x", linestyle="-",  color="#CCCCCC", linewidth=0.6, alpha=0.9)
        ax.grid(which="minor", axis="x", linestyle=":",  color="#E5E5E5", linewidth=0.3, alpha=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fname = col.replace(" ", "_")
        path  = os.path.join(output_dir, f"tornado_{fname}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Tornado plots guardados en: {output_dir}")


def _graficar_heatmap(
    df_srcc:    pd.DataFrame,
    etiquetas:  Dict[str, str],
    output_dir: str,
):
    """Heatmap parámetros × variables con SRCC anotados."""
    cols_sal = [c for c in etiquetas if c in df_srcc.columns]
    if not cols_sal:
        return

    df_plot = df_srcc[cols_sal].copy()
    orden   = df_plot.abs().mean(axis=1).sort_values(ascending=False).index
    df_plot = df_plot.loc[orden]

    fig, ax = plt.subplots(
        figsize=(max(5, 1.5 * len(cols_sal)), max(3, 0.5 * len(df_plot)))
    )
    im = ax.imshow(df_plot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="SRCC")

    ax.set_xticks(range(len(cols_sal)))
    ax.set_xticklabels(
        [etiquetas[c] for c in cols_sal], rotation=30, ha="right",
        fontsize=9, fontweight="bold"
    )
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot.index, fontsize=9, fontweight="bold")
    ax.set_title(
        "Índices de sensibilidad (SRCC)\nrojo = directa  ·  azul = inversa",
        fontsize=11, fontweight="bold",
    )

    for i in range(len(df_plot)):
        for j in range(len(cols_sal)):
            v = df_plot.iloc[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) >= 0.6 else "#333333"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(output_dir, "heatmap_srcc.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap guardado: {path}")


# ===========================================================================
# SRCC espacial
# ===========================================================================

def _calcular_srcc_espacial(
    perfiles:       Dict[str, List[list]],
    params_vals:    Dict[str, List[float]],
    nombres_params: List[str],
    distancias:     List[float],
    output_dir:     str,
):
    """
    Calcula el SRCC entre cada parámetro y la variable simulada en cada
    sección espacial. Genera un CSV y un heatmap por variable de salida.

    El resultado revela dónde a lo largo del río cada parámetro tiene más
    (o menos) influencia, evitando el sesgo de dominancia que produce la
    media espacial global.
    """
    dist_arr = np.array(distancias)

    print("Calculando SRCC espacial (por sección)...")
    for var, runs_data in perfiles.items():
        if not runs_data or var not in _VARIABLES_SALIDA:
            continue

        mat = np.array(runs_data)          # (n_runs, n_secciones)
        if mat.ndim != 2 or mat.shape[1] != len(dist_arr):
            continue

        srcc_por_param: Dict[str, List[float]] = {}
        for param in nombres_params:
            x    = np.array(params_vals[param])
            rhos = []
            for s in range(mat.shape[1]):
                y    = mat[:, s]
                mask = ~np.isnan(y)
                if mask.sum() >= 4:
                    rho, _ = spearmanr(x[mask], y[mask])
                    rhos.append(round(float(rho), 4))
                else:
                    rhos.append(float("nan"))
            srcc_por_param[param] = rhos

        # DataFrame: filas = parámetros, columnas = km
        df_esp = pd.DataFrame(srcc_por_param, index=dist_arr).T
        df_esp.index.name   = "parametro"
        df_esp.columns.name = "km"

        csv_path = os.path.join(output_dir, f"srcc_espacial_{var}.csv")
        df_esp.to_csv(csv_path)
        print(f"  SRCC espacial ({_VARIABLES_SALIDA[var]}): {csv_path}")

        _graficar_perfil_sensibilidad(df_esp, _VARIABLES_SALIDA[var], dist_arr, output_dir, var)

    print()


def _graficar_perfil_sensibilidad(
    df_esp:    "pd.DataFrame",
    etiqueta:  str,
    distancias: np.ndarray,
    output_dir: str,
    var:        str,
):
    """
    Heatmap: eje X = distancia longitudinal (km), eje Y = parámetros,
    color = SRCC. Muestra cómo varía la influencia de cada parámetro
    a lo largo del río.

    Útil para identificar parámetros que dominan localmente en una zona
    (ej. cabecera aguas arriba) aunque su media global sea baja.
    """
    n_params = len(df_esp)
    n_secs   = len(distancias)

    fig, ax = plt.subplots(
        figsize=(max(10, n_secs * 0.12), max(3, n_params * 0.55))
    )

    im = ax.imshow(
        df_esp.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="SRCC", fraction=0.03, pad=0.02)

    ax.set_yticks(range(n_params))
    ax.set_yticklabels(df_esp.index, fontsize=9, fontweight="bold")

    n_xticks = min(12, n_secs)
    step     = max(1, n_secs // n_xticks)
    ax.set_xticks(range(0, n_secs, step))
    ax.set_xticklabels(
        [f"{distancias[i]:.1f}" for i in range(0, n_secs, step)],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_xlabel(
        "Distancia Longitudinal (km)  —  aguas arriba (km alto) → aguas abajo (km 0)",
        fontsize=9, fontweight="bold",
    )
    ax.set_title(
        f"Sensibilidad espacial — {etiqueta}\n"
        "rojo = directa  ·  azul = inversa  ·  blanco = sin influencia",
        fontsize=11, fontweight="bold",
    )

    fig.tight_layout()
    fname = var.replace(" ", "_")
    path  = os.path.join(output_dir, f"sensibilidad_espacial_{fname}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap espacial guardado: {path}")
