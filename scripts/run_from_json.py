"""
run_from_json.py
================
Ejecuta una simulación QUAL2K completa a partir de un archivo JSON de configuración.
No modifica las clases originales: usa Q2KJsonLoader como puente entre el JSON
y Q2KModel.

Uso:
    python scripts/run_from_json.py simulacion.json
    python scripts/run_from_json.py simulacion.json --sin-graficas
    python scripts/run_from_json.py simulacion.json --sin-metricas

Para análisis de sensibilidad / corridas iterativas, importar run_simulacion():
    from scripts.run_from_json import run_simulacion
    resultado = run_simulacion("escenario_01.json")
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# Asegurar que el paquete qual2k sea encontrado desde cualquier directorio
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

warnings.filterwarnings("ignore")

from qual2k.core.model import Q2KModel
from qual2k.processing.json_loader import Q2KJsonLoader


# ---------------------------------------------------------------------------
# Función principal reutilizable
# ---------------------------------------------------------------------------

def run_simulacion(
    json_path: str,
    generar_graficas: bool = True,
    calcular_metricas: bool = True,
    verbose: bool = True,
) -> Optional[object]:
    """
    Ejecuta el flujo completo de simulación QUAL2K desde un JSON.

    Args:
        json_path        : Ruta al archivo JSON de configuración.
        generar_graficas : Si se generan las gráficas de resultados.
        calcular_metricas: Si se calculan las métricas KGE al final.
        verbose          : Si se imprime el progreso en consola.

    Returns:
        data_exp (DataFrame con resultados modelados + observados),
        o None si la simulación falla.
    """
    if verbose:
        print("=" * 70)
        print(f"SIMULACIÓN QUAL2K DESDE JSON")
        print(f"Archivo: {json_path}")
        print("=" * 70)

    # 1. Cargar el JSON
    loader = Q2KJsonLoader(json_path).cargar()

    # 2. Crear el modelo con el header del JSON
    model = Q2KModel(
        filepath=loader.header_dict["filedir"],
        header_dict=loader.header_dict,
    )

    # 3. Inyectar los DataFrames (equivalente a cargar_plantillas() pero desde JSON)
    model.data_reaches = loader.data_reaches
    model.data_sources = loader.data_sources
    model.data_wq      = loader.data_wq

    # 4. Aplicar overrides de rates y light (sobre los defaults de Q2KConfig)
    if loader.rates_override:
        model.config.actualizar_rates(**loader.rates_override)

    if loader.light_override:
        model.config.actualizar_light(**loader.light_override)

    # 5. Configurar el modelo (procesa los DataFrames → q2k_data)
    model.configurar_modelo(
        numelem_default=loader.numelem_default,
        q_cabecera=loader.q_cabecera,
        estacion_cabecera=loader.estacion_cabecera,
        reach_rates_custom=loader.reach_rates_custom,
    )

    # 6. Generar archivo .q2k y ejecutar FORTRAN
    model.generar_archivo_q2k()
    model.ejecutar_simulacion()

    # 7. Analizar resultados
    data_exp = model.analizar_resultados(generar_graficas=generar_graficas)

    # 8. Métricas de calibración (opcional)
    if calcular_metricas:
        try:
            _, kge = model.calcular_metricas_calibracion()
            if verbose:
                print(f"\nKGE global: {kge:.4f}")
        except Exception as e:
            if verbose:
                print(f"\n[Advertencia] No se pudieron calcular métricas: {e}")

    if verbose:
        print("=" * 70)
        print("SIMULACIÓN COMPLETADA")
        print("=" * 70)

    return data_exp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_from_json",
        description="Ejecuta QUAL2K desde un archivo JSON de configuración.",
    )
    p.add_argument(
        "json",
        metavar="ARCHIVO.json",
        help="Ruta al JSON de simulación.",
    )
    p.add_argument(
        "--sin-graficas",
        action="store_true",
        default=False,
        help="Omitir generación de gráficas.",
    )
    p.add_argument(
        "--sin-metricas",
        action="store_true",
        default=False,
        help="Omitir cálculo de métricas KGE.",
    )
    p.add_argument(
        "--silencioso",
        action="store_true",
        default=False,
        help="Suprimir salida en consola.",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    run_simulacion(
        json_path=args.json,
        generar_graficas=not args.sin_graficas,
        calcular_metricas=not args.sin_metricas,
        verbose=not args.silencioso,
    )
