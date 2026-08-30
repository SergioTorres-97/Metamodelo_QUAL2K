from pathlib import Path
import multiprocessing as mp
from qual2k.core.calibrator_global import CalibracionGlobal

# ============================================================================
# CONFIGURACIÓN DE PATHS
# ============================================================================
base = Path(__file__).parent.parent

# ============================================================================
# CONFIGURACIONES DE LOS MODELOS
# Rutas y parámetros tal como en pipeline_modelo_calidad.py
# ============================================================================

HEADER_BASE = {
    "version":      "v2.12",
    "xmon":         6,
    "xday":         27,
    "xyear":        2012,
    "timezonehour": -6,
    "pco2":         0.000347,
    "dtuser":       4.16666666666667E-03,
    "tf":           5,
    "IMeth":        "Euler",
    "IMethpH":      "Brent",
}

config_vargas = {
    'filepath':    str(base / 'data/templates/Canal_vargas'),
    'header_dict': {**HEADER_BASE,
                    "rivname":  "Canal_vargas",
                    "filename": "Canal_vargas",
                    "filedir":  str(base / 'data/templates/Canal_vargas'),
                    "applabel": "Canal_vargas (6/27/2012)"},
    'q_cabecera':  2.3148E-06,
}

config_tramo3s = {
    'filepath':    str(base / 'data/templates/Tramo_3s'),
    'header_dict': {**HEADER_BASE,
                    "rivname":  "Tramo_3s",
                    "filename": "Tramo_3s",
                    "filedir":  str(base / 'data/templates/Tramo_3s'),
                    "applabel": "Tramo_3s (6/27/2012)"},
    'q_cabecera':  4.1666E-06,
}

config_chicamocha = {
    'filepath':    str(base / 'data/templates/Chicamocha'),
    'header_dict': {**HEADER_BASE,
                    "rivname":  "Chicamocha",
                    "filename": "Chicamocha",
                    "filedir":  str(base / 'data/templates/Chicamocha'),
                    "applabel": "Chicamocha (6/27/2012)"},
    'q_cabecera':  1.053E-06,
}

# ============================================================================
# PARÁMETROS A CALIBRAR
# (min, max, es_global)  →  es_global=False = un gen por reach
# ============================================================================

parametros_vargas = {
    'kaaa': (0.5, 4.5, False),
    'kdc':  (0.01, 2.0, False),
    'kn':   (0.0001, 0.05, False),
    'khp':  (0.1, 2.0, False),
    'kdt':  (0.01, 2.0, False),
}

parametros_tramo3s = {
    'kaaa': (0.5, 4.5, False),
    'kdc':  (0.01, 2.0, False),
    'kn':   (0.0001, 0.05, False),
    'khp':  (0.1, 2.0, False),
    'kdt':  (0.01, 2.0, False),
}

parametros_chicamocha = {
    'kaaa': (0.5, 4.5, False),
    'kdc':  (0.01, 2.0, False),
    'kn':   (0.0001, 0.05, False),
    'khp':  (0.1, 2.0, False),
    'kdt':  (0.01, 2.0, False),
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║     CALIBRACIÓN GLOBAL — Canal Vargas → Tramo 3S → Chicamocha             ║
    ║     Objetivo: maximizar KGE de Chicamocha                                 ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    calibrador = CalibracionGlobal(
        config_vargas=config_vargas,
        config_tramo3s=config_tramo3s,
        config_chicamocha=config_chicamocha,

        parametros_vargas=parametros_vargas,
        parametros_tramo3s=parametros_tramo3s,
        parametros_chicamocha=parametros_chicamocha,

        # ===== PARÁMETROS DEL GA =====
        num_generations=150,
        population_size=50,
        num_parents_mating=25,

        parent_selection_type="tournament",
        k_tournament=3,

        crossover_type="uniform",
        crossover_probability=0.85,

        mutation_type="adaptive",
        mutation_probability=[0.35, 0.08],
        mutation_percent_genes=20,

        keep_elitism=5,

        stop_criteria="saturate_20",

        random_seed=42,
        usar_paralelo=True,
        num_workers=max(1, mp.cpu_count() - 1),
    )

    try:
        solucion, kge = calibrador.ejecutar(
            txt_log_path=str(base / 'data/resultados_calibracion/log_generaciones.txt')
        )

        calibrador.imprimir_parametros_calibrados()

        output_dir = str(base / 'data/resultados_calibracion')
        calibrador.plotear_evolucion_fitness(
            filename=str(base / 'data/resultados_calibracion/evolucion_fitness_global.png')
        )
        calibrador.exportar_historial_csv(
            filename=str(base / 'data/resultados_calibracion/historial_calibracion_global.csv')
        )
        calibrador.correr_mejor_solucion(output_dir=output_dir)

        print("\n" + "=" * 80)
        print(f"✅ CALIBRACIÓN COMPLETADA  |  KGE Chicamocha: {kge:.6f}")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️  Calibración interrumpida por el usuario.")

    except Exception as e:
        import traceback
        print(f"\n❌ Error durante la calibración: {e}")
        traceback.print_exc()


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == '__main__':
    mp.freeze_support()
    main()
