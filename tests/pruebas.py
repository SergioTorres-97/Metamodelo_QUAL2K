from pathlib import Path
import multiprocessing as mp
from qual2k.core.calibrator import Calibracion

if __name__ == '__main__':
    mp.freeze_support()

    # Configuración base
    base = Path(__file__).parent.parent
    filepath = f'{base}/data/templates/Tramo_3s'

    header_dict = {
        "version": "v2.12",
        "rivname": "Tramo_3s",
        "filename": "Tramo_3s",
        "filedir": filepath,
        "applabel": "Tramo_3s (6/27/2012)",
        "xmon": 6,
        "xday": 27,
        "xyear": 2012,
        "timezonehour": -6,
        "pco2": 0.000347,
        "dtuser": 4.16666666666667E-03,
        "tf": 5,
        "IMeth": "Euler",
        "IMethpH": "Brent"
    }

    # Parámetros a calibrar
    parametros = {
        'kaaa': (0.1, 3, False),
        'kn': (0.0005, 0.05, False),
        'kdc': (0.05, 1.5, False),
        'kdt': (0.05, 2.5, False),
        'khp': (0.05, 2.5, False),

    }

    # Crear instancia de calibración
    calibracion = Calibracion(
            filepath=filepath,
            header_dict=header_dict,
            parametros=parametros,
            # Básicos
            num_generations=100,
            population_size=50,
            num_parents_mating=20,
            # Selección
            parent_selection_type="tournament",
            k_tournament=3,
            # Cruce
            crossover_type="single_point",
            crossover_probability=0.9,
            # Mutación
            mutation_type="random",
            mutation_probability=0.2,
            mutation_percent_genes=20,
            # Elitismo
            keep_elitism=5,
            # Reproducibilidad
            random_seed=42,
            # Paralelismo
            num_workers=4,
            usar_paralelo=True)

    # ========================================================================
    # OPCIÓN 2: Usar preset
    # ========================================================================
    # preset_config = CalibracionPresets.balanceado()
    # calibracion = Calibracion(
    #     filepath=filepath,
    #     header_dict=header_dict,
    #     parametros=parametros,
    #     **preset_config
    # )

    # ========================================================================
    # OPCIÓN 3: Configuración avanzada con criterios de parada
    # ========================================================================
    # calibracion = Calibracion(
    #     filepath=filepath,
    #     header_dict=header_dict,
    #     parametros=parametros,
    #     num_generations=200,
    #     population_size=50,
    #     num_parents_mating=20,
    #     parent_selection_type="rank",
    #     crossover_type="two_points",
    #     mutation_type="adaptive",
    #     stop_criteria=["reach_0.95", "saturate_50"],  # Para si alcanza KGE=0.95 o se estanca 50 gens
    #     random_seed=42
    # )

    # Exportar configuración
    calibracion.exportar_configuracion()

    # Ejecutar calibración (con gráficas automáticas)
    resultado = calibracion.ejecutar(generar_graficas=True)

    # También puedes generar gráficas manualmente después
    if resultado is not None:
        # Gráfica completa en alta resolución
        calibracion.plotear_evolucion_fitness(
            filename='analisis_completo.png',
            dpi=300,
            formato='png'
        )

        # Gráfica simple para presentación
        calibracion.plotear_fitness_simple(
            filename='presentacion.png',
            dpi=150
        )

        # Versión PDF para publicación
        calibracion.plotear_evolucion_fitness(
            filename='publicacion.pdf',
            dpi=500,
            formato='pdf'
        )
