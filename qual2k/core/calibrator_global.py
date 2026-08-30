"""
calibrator_global.py
====================
Calibración global del pipeline Canal Vargas → Tramo 3S → Chicamocha.

Optimiza los parámetros de los tres subsistemas simultáneamente con un
único algoritmo genético, maximizando el KGE de Chicamocha (modelo final).

Cada evaluación del GA corre el pipeline completo con propagación de
resultados entre subsistemas, replicando el flujo de pipeline_modelo_calidad.py.
"""

import os
import glob
import shutil
import tempfile
import threading
import multiprocessing as mp
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import pygad
import matplotlib.pyplot as plt

from qual2k.core.model import Q2KModel

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

# ─── CONSTANTES ───────────────────────────────────────────────────────────────

PARAM_NAMES = ['kaaa', 'khc', 'kdcs', 'kdc', 'khn', 'kn', 'ki', 'khp', 'kdt']

MAPEO_COLUMNAS = {
    'flow':                    'CAUDAL',
    'water_temp_c':            'TEMPERATURA',
    'conductivity':            'CONDUCTIVIDAD',
    'total_suspended_solids':  'SST',
    'dissolved_oxygen':        'OXIGENO_DISUELTO',
    'carbonaceous_bod_fast':   'DBO5',
    'total_kjeldahl_nitrogen': 'NTK',
    'ammonium':                'NITROGENO_AMONIACAL',
    'nitrite':                 'NITRITOS',
    'nitrate':                 'NITRATOS',
    'total_phosphorus':        'FOSFORO_TOTAL',
    'inorganic_phosphorus':    'ORTOFOSFATOS',
    'pathogen':                'E_COLI',
    'alkalinity':              'ALCALINIDAD',
    'pH':                      'pH',
}

COLUMNAS_UG_A_MG = {
    'total_kjeldahl_nitrogen', 'ammonium', 'nitrite',
    'nitrate', 'total_phosphorus', 'inorganic_phosphorus',
}

PESOS_KGE_DEFAULT = {
    "water_temp_c":            0.05,
    "conductivity":            0.05,
    "nitrate":                 0.05,
    "pathogen":                0.10,
    "pH":                      0.05,
    "total_suspended_solids":  0.05,
    "dissolved_oxygen":        0.30,
    "carbonaceous_bod_fast":   0.30,
    "total_kjeldahl_nitrogen": 0.02,
    "ammonium":                0.02,
    "total_phosphorus":        0.01,
}

# ─── FUNCIONES AUXILIARES (deben ser importables en workers) ──────────────────

def _copiar_directorio(src: str, dst: str) -> None:
    """Copia las plantillas de src a dst, excluyendo archivos de salida."""
    for archivo in glob.glob(os.path.join(src, '*')):
        if os.path.isfile(archivo):
            nombre = os.path.basename(archivo)
            if not any(nombre.endswith(ext) for ext in ['.out', '.txt', '.dat', '.q2k']):
                try:
                    shutil.copy2(archivo, os.path.join(dst, nombre))
                except Exception:
                    pass


def _actualizar_vertimiento(
        filepath_resultados: str,
        filepath_excel: str,
        nombre_vertimiento: str,
        fila_resultado: int = 0,
) -> None:
    """
    Actualiza un vertimiento en el Excel con la fila indicada del CSV de
    resultados del modelo anterior. Replica el comportamiento de
    actualizar_vertimiento_desde_resultados() en pipeline_modelo_calidad.py.
    """
    df = pd.read_csv(filepath_resultados, header=0, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(',', '.'),
            errors='coerce',
        )
    resultados = df.iloc[fila_resultado]

    todas_las_hojas = pd.read_excel(filepath_excel, sheet_name=None)
    sources = todas_las_hojas['SOURCES'].copy()

    idx = sources[sources['NOMBRE_VERTIMIENTO'] == nombre_vertimiento].index
    if len(idx) > 0:
        for col_csv, col_excel in MAPEO_COLUMNAS.items():
            if col_csv in resultados.index and col_excel in sources.columns:
                valor = resultados[col_csv] if pd.notna(resultados[col_csv]) else None
                if valor is not None and col_csv in COLUMNAS_UG_A_MG:
                    valor = valor / 1000.0
                sources.at[idx[0], col_excel] = valor
        for col in MAPEO_COLUMNAS.values():
            if col in sources.columns:
                sources[col] = sources[col].astype(float)

    todas_las_hojas['SOURCES'] = sources
    with pd.ExcelWriter(filepath_excel, engine='openpyxl') as writer:
        for nombre_hoja, df_hoja in todas_las_hojas.items():
            df_hoja.to_excel(writer, sheet_name=nombre_hoja, index=False)


def _construir_reach_rates(model: Q2KModel, n: int, params: Dict) -> Any:
    """Construye reach_rates_custom a partir del dict de parámetros decodificado."""
    return model.config.generar_reach_rates_custom(
        n=n,
        kaaa_list=params['kaaa'],
        khc_list=params['khc'],
        kdcs_list=params['kdcs'],
        kdc_list=params['kdc'],
        khn_list=params['khn'],
        kn_list=params['kn'],
        ki_list=params['ki'],
        khp_list=params['khp'],
        kdt_list=params['kdt'],
    )


# ─── WORKER ESTÁTICO ──────────────────────────────────────────────────────────

def _evaluar_pipeline_worker(args: Tuple) -> Tuple[int, float]:
    """
    Evalúa un cromosoma ejecutando el pipeline completo:
        Canal Vargas → (propaga) → Tramo 3S → (propaga) → Chicamocha

    Retorna (eval_id, kge_chicamocha).
    Debe ser una función de módulo (no método) para ser serializable por multiprocessing.
    """
    (solution, eval_id,
     filepath_vargas, header_vargas, q_cabecera_vargas,
     filepath_tramo3s, header_tramo3s, q_cabecera_tramo3s,
     filepath_chicamocha, header_chicamocha, q_cabecera_chicamocha,
     param_map, n_vargas, n_tramo3s, n_chicamocha,
     pesos_kge) = args

    temp_vargas     = tempfile.mkdtemp(prefix=f'q2k_v{eval_id}_')
    temp_tramo3s    = tempfile.mkdtemp(prefix=f'q2k_t{eval_id}_')
    temp_chicamocha = tempfile.mkdtemp(prefix=f'q2k_c{eval_id}_')

    try:
        # Copiar plantillas a directorios temporales independientes
        _copiar_directorio(filepath_vargas,     temp_vargas)
        _copiar_directorio(filepath_tramo3s,    temp_tramo3s)
        _copiar_directorio(filepath_chicamocha, temp_chicamocha)

        # ── Decodificar cromosoma ────────────────────────────────────────────
        params = {
            'vargas':     {p: [None] * n_vargas     for p in PARAM_NAMES},
            'tramo3s':    {p: [None] * n_tramo3s    for p in PARAM_NAMES},
            'chicamocha': {p: [None] * n_chicamocha for p in PARAM_NAMES},
        }
        n_map = {'vargas': n_vargas, 'tramo3s': n_tramo3s, 'chicamocha': n_chicamocha}

        for gene_idx, (subsistema, param_name, reach_idx) in enumerate(param_map):
            valor = solution[gene_idx]
            if reach_idx is None:
                params[subsistema][param_name] = [valor] * n_map[subsistema]
            else:
                params[subsistema][param_name][reach_idx] = valor

        # ── 1. Canal Vargas ──────────────────────────────────────────────────
        header_v = {**header_vargas, 'filedir': temp_vargas}
        model_v = Q2KModel(temp_vargas, header_v)
        model_v.cargar_plantillas()
        model_v.configurar_modelo(
            reach_rates_custom=_construir_reach_rates(model_v, n_vargas, params['vargas']),
            q_cabecera=q_cabecera_vargas,
        )
        model_v.generar_archivo_q2k()
        model_v.ejecutar_simulacion()
        model_v.analizar_resultados(generar_graficas=False)

        csv_vargas = os.path.join(
            temp_vargas, 'resultados', f"{header_vargas['filename']}.csv"
        )

        # ── 2. Tramo 3S ──────────────────────────────────────────────────────
        _actualizar_vertimiento(
            filepath_resultados=csv_vargas,
            filepath_excel=os.path.join(temp_tramo3s, 'PlantillaBaseQ2K.xlsx'),
            nombre_vertimiento='CANAL VARGAS',
        )
        header_t = {**header_tramo3s, 'filedir': temp_tramo3s}
        model_t = Q2KModel(temp_tramo3s, header_t)
        model_t.cargar_plantillas()
        model_t.configurar_modelo(
            reach_rates_custom=_construir_reach_rates(model_t, n_tramo3s, params['tramo3s']),
            q_cabecera=q_cabecera_tramo3s,
        )
        model_t.generar_archivo_q2k()
        model_t.ejecutar_simulacion()
        model_t.analizar_resultados(generar_graficas=False)

        csv_tramo3s = os.path.join(
            temp_tramo3s, 'resultados', f"{header_tramo3s['filename']}.csv"
        )

        # ── 3. Chicamocha ────────────────────────────────────────────────────
        _actualizar_vertimiento(
            filepath_resultados=csv_tramo3s,
            filepath_excel=os.path.join(temp_chicamocha, 'PlantillaBaseQ2K.xlsx'),
            nombre_vertimiento='R. CHIQUITO',
        )
        header_c = {**header_chicamocha, 'filedir': temp_chicamocha}
        model_c = Q2KModel(temp_chicamocha, header_c)
        model_c.cargar_plantillas()
        model_c.configurar_modelo(
            reach_rates_custom=_construir_reach_rates(model_c, n_chicamocha, params['chicamocha']),
            q_cabecera=q_cabecera_chicamocha,
        )
        model_c.config.actualizar_rates(kpath=0.05, aPath=0.001)
        model_c.generar_archivo_q2k()
        model_c.ejecutar_simulacion()
        model_c.analizar_resultados(generar_graficas=False)

        metricas, kge = model_c.calcular_metricas_calibracion(pesos=pesos_kge)
        return (eval_id, float(kge), metricas)

    except Exception as e:
        print(f'Error en evaluación {eval_id}: {e}')
        return (eval_id, -999.0, {})

    finally:
        for d in [temp_vargas, temp_tramo3s, temp_chicamocha]:
            shutil.rmtree(d, ignore_errors=True)


# ─── CLASE PRINCIPAL ──────────────────────────────────────────────────────────

class CalibracionGlobal:
    """
    Calibra simultáneamente los parámetros de Canal Vargas, Tramo 3S y
    Chicamocha con un único algoritmo genético.

    La función de fitness ejecuta el pipeline completo con propagación de
    resultados entre subsistemas y retorna el KGE de Chicamocha.
    """

    def __init__(
            self,
            config_vargas: Dict[str, Any],
            config_tramo3s: Dict[str, Any],
            config_chicamocha: Dict[str, Any],
            parametros_vargas: Dict[str, Tuple[float, float, bool]],
            parametros_tramo3s: Dict[str, Tuple[float, float, bool]],
            parametros_chicamocha: Dict[str, Tuple[float, float, bool]],
            # Parámetros del GA
            num_generations: int = 100,
            population_size: int = 40,
            num_parents_mating: int = 16,
            parent_selection_type: str = "tournament",
            k_tournament: int = 3,
            crossover_type: str = "single_point",
            crossover_probability: float = 0.9,
            mutation_type: str = "random",
            mutation_probability: Union[float, List[float]] = 0.2,
            mutation_percent_genes: Union[int, float] = 20,
            mutation_by_replacement: bool = False,
            random_mutation_min_val: Optional[float] = None,
            random_mutation_max_val: Optional[float] = None,
            keep_elitism: int = 3,
            keep_parents: int = -1,
            stop_criteria: Optional[Union[str, List[str]]] = None,
            allow_duplicate_genes: bool = True,
            random_seed: Optional[int] = None,
            num_workers: Optional[int] = None,
            usar_paralelo: bool = True,
            pesos_kge: Optional[Dict[str, float]] = None,
    ):
        self.config_vargas     = config_vargas
        self.config_tramo3s    = config_tramo3s
        self.config_chicamocha = config_chicamocha

        self.parametros_vargas     = parametros_vargas
        self.parametros_tramo3s    = parametros_tramo3s
        self.parametros_chicamocha = parametros_chicamocha

        self.num_generations       = num_generations
        self.population_size       = population_size
        self.num_parents_mating    = num_parents_mating
        self.parent_selection_type = parent_selection_type
        self.k_tournament          = k_tournament
        self.crossover_type        = crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_type         = mutation_type
        self.mutation_probability  = mutation_probability
        self.mutation_percent_genes  = mutation_percent_genes
        self.mutation_by_replacement = mutation_by_replacement
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val
        self.keep_elitism          = keep_elitism
        self.keep_parents          = keep_parents
        self.stop_criteria         = stop_criteria
        self.allow_duplicate_genes = allow_duplicate_genes
        self.random_seed           = random_seed
        self.usar_paralelo         = usar_paralelo
        self.num_workers           = num_workers or min(4, mp.cpu_count() - 1)
        self.pesos_kge             = pesos_kge or PESOS_KGE_DEFAULT

        # Estado
        self.contador_evaluaciones  = 0
        self.mejor_kge              = -999.0
        self.mejor_metricas         = {}
        self.n_vargas               = 0
        self.n_tramo3s              = 0
        self.n_chicamocha           = 0
        self.gene_space             = []
        self.param_map              = []
        self.ga_instance            = None
        self.mejor_solucion         = None
        self.historial_generaciones = []
        self.txt_log_path           = None
        self._lock                  = threading.Lock()

    # ── Inicialización ────────────────────────────────────────────────────────

    def _detectar_n_reaches(self) -> None:
        """Lee las plantillas para determinar el número de reaches de cada subsistema."""
        for attr, config in [
            ('n_vargas',     self.config_vargas),
            ('n_tramo3s',    self.config_tramo3s),
            ('n_chicamocha', self.config_chicamocha),
        ]:
            m = Q2KModel(config['filepath'], config['header_dict'])
            m.cargar_plantillas()
            setattr(self, attr, len(m.data_reaches))

    def _configurar_genes(self) -> None:
        """
        Construye gene_space y param_map para los tres subsistemas.
        El cromosoma tiene la forma: [genes_vargas | genes_tramo3s | genes_chicamocha].
        Cada entrada de param_map es (subsistema, nombre_param, reach_idx_o_None).
        """
        self.gene_space = []
        self.param_map  = []

        subsistemas = [
            ('vargas',     self.parametros_vargas,     self.n_vargas),
            ('tramo3s',    self.parametros_tramo3s,    self.n_tramo3s),
            ('chicamocha', self.parametros_chicamocha, self.n_chicamocha),
        ]

        for subsistema, params, n_reaches in subsistemas:
            for param_name, (min_val, max_val, is_global) in params.items():
                if is_global:
                    self.gene_space.append({'low': min_val, 'high': max_val})
                    self.param_map.append((subsistema, param_name, None))
                else:
                    for i in range(n_reaches):
                        self.gene_space.append({'low': min_val, 'high': max_val})
                        self.param_map.append((subsistema, param_name, i))

    # ── GA callbacks ──────────────────────────────────────────────────────────

    def _escribir_generacion_txt(self, ga, gen: int, stats: dict) -> None:
        """Agrega los resultados parciales de una generación al log TXT."""
        best_idx          = int(np.argmax(ga.last_generation_fitness))
        best_solution_gen = ga.population[best_idx]

        sep   = '─' * 70
        sep_s = '─' * 46

        with open(self.txt_log_path, 'a', encoding='utf-8') as f:
            f.write(f'\n{"=" * 70}\n')
            f.write(f'  GENERACIÓN {gen}\n')
            f.write(f'{"=" * 70}\n')

            # ── estadísticas de fitness ──────────────────────────────────────
            f.write(f'  Mejor generación : {stats["mejor_fitness"]:>10.6f}\n')
            f.write(f'  Mejor global     : {stats["mejor_global"]:>10.6f}\n')
            f.write(f'  Promedio         : {stats["promedio"]:>10.6f} ± {stats["std"]:.6f}\n')
            f.write(f'  Mediana          : {stats["mediana"]:>10.6f}\n')
            f.write(f'  Min / Max        : {stats["min"]:>10.6f} / {stats["max"]:.6f}\n')

            # ── mejores parámetros de esta generación ────────────────────────
            f.write(f'\n  Mejores parámetros (generación {gen}):\n')
            f.write(f'  {sep_s}\n')
            f.write(f'  {"Subsistema":<12} {"Parámetro":<10} {"Reach":>7}  {"Valor":>12}\n')
            f.write(f'  {sep_s}\n')
            for gene_val, (subsistema, param_name, reach_idx) in zip(best_solution_gen, self.param_map):
                reach_str = str(reach_idx) if reach_idx is not None else 'global'
                f.write(f'  {subsistema:<12} {param_name:<10} {reach_str:>7}  {gene_val:>12.6f}\n')
            f.write(f'  {sep_s}\n')

            # ── KGEs del mejor global acumulado ──────────────────────────────
            if self.mejor_metricas:
                f.write(f'\n  KGEs por variable (mejor global acumulado):\n')
                f.write(f'  {sep}\n')
                f.write(f'  {"Variable":<34} {"KGE":>8}  {"Peso":>6}  {"Contrib":>10}\n')
                f.write(f'  {sep}\n')
                kge_total = 0.0
                for var, kge_var in self.mejor_metricas.items():
                    peso   = self.pesos_kge.get(var, 0.0)
                    contrib = kge_var * peso
                    kge_total += contrib
                    f.write(f'  {var:<34} {kge_var:>8.4f}  {peso:>6.2f}  {contrib:>10.4f}\n')
                f.write(f'  {sep}\n')
                f.write(f'  {"KGE PONDERADO GLOBAL":<34} {kge_total:>8.4f}\n')
            f.write('\n')

    def _fitness_function(self, ga, solution, solution_idx):
        """Corre el pipeline completo y retorna el KGE de Chicamocha como fitness.

        PyGAD llama esta función desde N threads simultáneos cuando se usa
        parallel_processing=['thread', N]. El Lock protege el contador y el
        registro del mejor KGE.
        """
        with self._lock:
            self.contador_evaluaciones += 1
            eval_id = self.contador_evaluaciones

        args = (
            solution, eval_id,
            self.config_vargas['filepath'],     self.config_vargas['header_dict'],     self.config_vargas['q_cabecera'],
            self.config_tramo3s['filepath'],    self.config_tramo3s['header_dict'],    self.config_tramo3s['q_cabecera'],
            self.config_chicamocha['filepath'], self.config_chicamocha['header_dict'], self.config_chicamocha['q_cabecera'],
            self.param_map, self.n_vargas, self.n_tramo3s, self.n_chicamocha,
            self.pesos_kge,
        )

        _, kge, metricas = _evaluar_pipeline_worker(args)

        with self._lock:
            if kge > self.mejor_kge:
                self.mejor_kge = kge
                self.mejor_metricas = metricas
                print(f"  *** Eval {eval_id} | NUEVO MEJOR KGE Chicamocha: {kge:.4f} ***")
            elif eval_id % 5 == 0:
                print(f"Eval {eval_id} | KGE Chicamocha: {kge:.4f}")

        return kge

    def _on_generation(self, ga):
        """Callback al completar cada generación."""
        gen = ga.generations_completed
        fitness = ga.last_generation_fitness

        stats = {
            'generacion':    gen,
            'mejor_fitness': float(np.max(fitness)),
            'mejor_global':  self.mejor_kge,
            'promedio':      float(np.mean(fitness)),
            'mediana':       float(np.median(fitness)),
            'std':           float(np.std(fitness)),
            'min':           float(np.min(fitness)),
            'max':           float(np.max(fitness)),
        }
        self.historial_generaciones.append(stats)

        if self.txt_log_path:
            self._escribir_generacion_txt(ga, gen, stats)

        print(f'\n{"=" * 65}')
        print(f'  GENERACIÓN {gen:3d}')
        print(f'  Mejor gen : {stats["mejor_fitness"]:.4f}  |  '
              f'Mejor global: {self.mejor_kge:.4f}  |  '
              f'Promedio: {stats["promedio"]:.4f} ± {stats["std"]:.4f}')

        if self.mejor_metricas:
            print(f'  {"─" * 61}')
            print(f'  {"Variable":<32} {"KGE":>8}  {"Peso":>6}  {"Contribución":>12}')
            print(f'  {"─" * 61}')
            kge_total = 0.0
            for var, kge_var in self.mejor_metricas.items():
                peso = self.pesos_kge.get(var, 0.0)
                contrib = kge_var * peso
                kge_total += contrib
                print(f'  {var:<32} {kge_var:>8.4f}  {peso:>6.2f}  {contrib:>12.4f}')
            print(f'  {"─" * 61}')
            print(f'  {"KGE PONDERADO GLOBAL (Chicamocha)":<32} {kge_total:>8.4f}')

        print('=' * 65)

    # ── Ejecución ─────────────────────────────────────────────────────────────

    def ejecutar(self, txt_log_path: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Ejecuta la calibración global.

        Args:
            txt_log_path: Ruta del archivo TXT donde se guardarán los resultados
                          parciales por generación (parámetros y KGEs). Si es
                          None no se genera el log.

        Returns:
            (mejor_solucion, kge_chicamocha)
        """
        # ── inicializar log TXT ──────────────────────────────────────────────
        self.txt_log_path = txt_log_path
        if txt_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(txt_log_path)), exist_ok=True)
            with open(txt_log_path, 'w', encoding='utf-8') as f:
                f.write('=' * 70 + '\n')
                f.write('CALIBRACIÓN GLOBAL — Canal Vargas → Tramo 3S → Chicamocha\n')
                f.write(f'Fecha            : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'Generaciones     : {self.num_generations}\n')
                f.write(f'Población        : {self.population_size}\n')
                f.write(f'Padres mating    : {self.num_parents_mating}\n')
                f.write(f'Mutación         : {self.mutation_type}  p={self.mutation_probability}\n')
                f.write(f'Stop criteria    : {self.stop_criteria}\n')
                f.write(f'Semilla          : {self.random_seed}\n')
                f.write('=' * 70 + '\n')

        print('\n' + '=' * 80)
        print('CALIBRACIÓN GLOBAL — Canal Vargas → Tramo 3S → Chicamocha')
        print('Objetivo: maximizar KGE de Chicamocha')
        print('=' * 80 + '\n')

        print('Detectando número de reaches...')
        self._detectar_n_reaches()
        print(f'  Canal Vargas : {self.n_vargas} reaches')
        print(f'  Tramo 3S     : {self.n_tramo3s} reaches')
        print(f'  Chicamocha   : {self.n_chicamocha} reaches')

        self._configurar_genes()
        n_genes = len(self.gene_space)
        print(f'  Total genes  : {n_genes}  '
              f'({self.n_vargas * len(self.parametros_vargas)} Vargas + '
              f'{self.n_tramo3s * len(self.parametros_tramo3s)} Tramo3S + '
              f'{self.n_chicamocha * len(self.parametros_chicamocha)} Chicamocha)\n')

        try:
            ga_kwargs = dict(
                num_generations=self.num_generations,
                num_parents_mating=self.num_parents_mating,
                fitness_func=self._fitness_function,
                sol_per_pop=self.population_size,
                num_genes=n_genes,
                gene_space=self.gene_space,
                parent_selection_type=self.parent_selection_type,
                crossover_type=self.crossover_type,
                crossover_probability=self.crossover_probability,
                mutation_type=self.mutation_type,
                mutation_probability=self.mutation_probability,
                mutation_percent_genes=self.mutation_percent_genes,
                mutation_by_replacement=self.mutation_by_replacement,
                keep_elitism=self.keep_elitism,
                keep_parents=self.keep_parents,
                allow_duplicate_genes=self.allow_duplicate_genes,
                on_generation=self._on_generation,
            )

            if self.parent_selection_type == "tournament":
                ga_kwargs['K_tournament'] = self.k_tournament
            if self.stop_criteria is not None:
                ga_kwargs['stop_criteria'] = self.stop_criteria
            if self.random_seed is not None:
                ga_kwargs['random_seed'] = self.random_seed
            if self.random_mutation_min_val is not None:
                ga_kwargs['random_mutation_min_val'] = self.random_mutation_min_val
            if self.random_mutation_max_val is not None:
                ga_kwargs['random_mutation_max_val'] = self.random_mutation_max_val
            if self.usar_paralelo:
                ga_kwargs['parallel_processing'] = ['thread', self.num_workers]

            self.ga_instance = pygad.GA(**ga_kwargs)
            self.ga_instance.run()

            self.mejor_solucion, _, _ = self.ga_instance.best_solution()

            print(f'\n✓ Calibración completada')
            print(f'  Mejor KGE Chicamocha : {self.mejor_kge:.6f}')
            print(f'  Total evaluaciones   : {self.contador_evaluaciones}\n')

            return self.mejor_solucion, self.mejor_kge

        except Exception:
            raise

    def correr_mejor_solucion(self, output_dir: str) -> None:
        """
        Re-corre el pipeline completo con los mejores parámetros y guarda
        todos los outputs (CSVs, gráficas comparativas) en output_dir.

        Crea la estructura:
            output_dir/
                Canal_vargas/resultados/
                Tramo_3s/resultados/
                Chicamocha/resultados/
                            comparacion/
        """
        if self.mejor_solucion is None:
            print("No hay solución calibrada. Ejecuta primero ejecutar().")
            return

        params = self.get_parametros_calibrados()
        os.makedirs(output_dir, exist_ok=True)

        print(f'\n{"=" * 65}')
        print('  RE-CORRIDA FINAL CON MEJORES PARÁMETROS')
        print(f'  Output: {output_dir}')
        print('=' * 65)

        # Directorios de trabajo (permanentes dentro de output_dir)
        dir_v = os.path.join(output_dir, 'Canal_vargas')
        dir_t = os.path.join(output_dir, 'Tramo_3s')
        dir_c = os.path.join(output_dir, 'Chicamocha')
        for d in [dir_v, dir_t, dir_c]:
            os.makedirs(d, exist_ok=True)

        _copiar_directorio(self.config_vargas['filepath'],     dir_v)
        _copiar_directorio(self.config_tramo3s['filepath'],    dir_t)
        _copiar_directorio(self.config_chicamocha['filepath'], dir_c)

        # ── 1. Canal Vargas ──────────────────────────────────────────────────
        print('\n  [1/3] Canal Vargas...')
        header_v = {**self.config_vargas['header_dict'], 'filedir': dir_v}
        model_v = Q2KModel(dir_v, header_v)
        model_v.cargar_plantillas()
        model_v.configurar_modelo(
            reach_rates_custom=_construir_reach_rates(model_v, self.n_vargas, params['vargas']),
            q_cabecera=self.config_vargas['q_cabecera'],
        )
        model_v.generar_archivo_q2k()
        model_v.ejecutar_simulacion()
        model_v.analizar_resultados(generar_graficas=True)

        csv_vargas = os.path.join(dir_v, 'resultados', f"{self.config_vargas['header_dict']['filename']}.csv")

        # ── 2. Tramo 3S ──────────────────────────────────────────────────────
        print('\n  [2/3] Tramo 3S...')
        _actualizar_vertimiento(
            filepath_resultados=csv_vargas,
            filepath_excel=os.path.join(dir_t, 'PlantillaBaseQ2K.xlsx'),
            nombre_vertimiento='CANAL VARGAS',
        )
        header_t = {**self.config_tramo3s['header_dict'], 'filedir': dir_t}
        model_t = Q2KModel(dir_t, header_t)
        model_t.cargar_plantillas()
        model_t.configurar_modelo(
            reach_rates_custom=_construir_reach_rates(model_t, self.n_tramo3s, params['tramo3s']),
            q_cabecera=self.config_tramo3s['q_cabecera'],
        )
        model_t.generar_archivo_q2k()
        model_t.ejecutar_simulacion()
        model_t.analizar_resultados(generar_graficas=True)

        csv_tramo3s = os.path.join(dir_t, 'resultados', f"{self.config_tramo3s['header_dict']['filename']}.csv")

        # ── 3. Chicamocha ────────────────────────────────────────────────────
        print('\n  [3/3] Chicamocha...')
        _actualizar_vertimiento(
            filepath_resultados=csv_tramo3s,
            filepath_excel=os.path.join(dir_c, 'PlantillaBaseQ2K.xlsx'),
            nombre_vertimiento='R. CHIQUITO',
        )
        header_c = {**self.config_chicamocha['header_dict'], 'filedir': dir_c}
        model_c = Q2KModel(dir_c, header_c)
        model_c.cargar_plantillas()
        model_c.configurar_modelo(
            reach_rates_custom=_construir_reach_rates(model_c, self.n_chicamocha, params['chicamocha']),
            q_cabecera=self.config_chicamocha['q_cabecera'],
        )
        model_c.config.actualizar_rates(kpath=0.05, aPath=0.001)
        model_c.generar_archivo_q2k()
        model_c.ejecutar_simulacion()
        model_c.analizar_resultados(generar_graficas=True)

        metricas, kge = model_c.calcular_metricas_calibracion(pesos=self.pesos_kge)

        # ── Resumen final ────────────────────────────────────────────────────
        print(f'\n{"=" * 65}')
        print('  RESULTADOS FINALES — CHICAMOCHA (mejor solución)')
        print(f'  {"─" * 61}')
        print(f'  {"Variable":<32} {"KGE":>8}  {"Peso":>6}  {"Contribución":>12}')
        print(f'  {"─" * 61}')
        kge_total = 0.0
        for var, kge_var in metricas.items():
            peso = self.pesos_kge.get(var, 0.0)
            contrib = kge_var * peso
            kge_total += contrib
            print(f'  {var:<32} {kge_var:>8.4f}  {peso:>6.2f}  {contrib:>12.4f}')
        print(f'  {"─" * 61}')
        print(f'  {"KGE PONDERADO GLOBAL":<32} {kge_total:>8.4f}')
        print(f'{"=" * 65}')
        print(f'\n  Outputs guardados en: {output_dir}\n')

    # ── Resultados ────────────────────────────────────────────────────────────

    def get_parametros_calibrados(self) -> Dict[str, Dict[str, List[float]]]:
        """Retorna los parámetros óptimos decodificados por subsistema."""
        if self.mejor_solucion is None:
            return {}

        params = {
            'vargas':     {p: [None] * self.n_vargas     for p in PARAM_NAMES},
            'tramo3s':    {p: [None] * self.n_tramo3s    for p in PARAM_NAMES},
            'chicamocha': {p: [None] * self.n_chicamocha for p in PARAM_NAMES},
        }
        n_map = {'vargas': self.n_vargas, 'tramo3s': self.n_tramo3s, 'chicamocha': self.n_chicamocha}

        for gene_idx, (subsistema, param_name, reach_idx) in enumerate(self.param_map):
            valor = float(self.mejor_solucion[gene_idx])
            if reach_idx is None:
                params[subsistema][param_name] = [valor] * n_map[subsistema]
            else:
                params[subsistema][param_name][reach_idx] = valor

        return params

    def imprimir_parametros_calibrados(self) -> None:
        """Imprime los parámetros óptimos de cada subsistema."""
        params = self.get_parametros_calibrados()
        if not params:
            print("No hay parámetros calibrados.")
            return

        print('\n' + '=' * 80)
        print('PARÁMETROS ÓPTIMOS CALIBRADOS')
        print('=' * 80)

        for subsistema, sub_params in params.items():
            print(f'\n  {subsistema.upper()}:')
            for param_name, valores in sub_params.items():
                valores_validos = [v for v in valores if v is not None]
                if not valores_validos:
                    continue
                if len(set(round(v, 8) for v in valores_validos)) == 1:
                    print(f'    {param_name:8s} (global):    {valores_validos[0]:.6f}')
                else:
                    vals_str = ', '.join(f'{v:.6f}' for v in valores if v is not None)
                    print(f'    {param_name:8s} (por reach): [{vals_str}]')

    def plotear_evolucion_fitness(
            self,
            filename: str = 'evolucion_fitness_global.png',
            dpi: int = 300,
    ) -> None:
        """Gráfica de evolución del mejor KGE de Chicamocha por generación."""
        if not self.historial_generaciones:
            print("No hay historial de generaciones.")
            return

        df = pd.DataFrame(self.historial_generaciones)
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df['generacion'], df['mejor_global'], 'o-',
                linewidth=2, markersize=5, label='Mejor global', color='#F18F01')
        ax.fill_between(df['generacion'],
                        df['promedio'] - df['std'],
                        df['promedio'] + df['std'],
                        alpha=0.2, color='#F18F01')
        ax.plot(df['generacion'], df['promedio'], '--',
                linewidth=1.5, alpha=0.7, color='#F18F01', label='Promedio ± 1σ')

        mejor_kge = df['mejor_global'].max()
        gen_mejor = df.loc[df['mejor_global'].idxmax(), 'generacion']
        ax.text(0.02, 0.98, f'KGE Máx: {mejor_kge:.4f}\n(Gen. {gen_mejor:.0f})',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

        ax.set_xlabel('Generación')
        ax.set_ylabel('KGE Chicamocha (Fitness)')
        ax.set_title('Calibración Global — Evolución del Fitness\n'
                     'Canal Vargas → Tramo 3S → Chicamocha')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'✓ Gráfica guardada: {filename}')

    def exportar_historial_csv(
            self,
            filename: str = 'historial_calibracion_global.csv',
    ) -> None:
        """Exporta el historial de generaciones a CSV."""
        if not self.historial_generaciones:
            return
        pd.DataFrame(self.historial_generaciones).to_csv(filename, index=False)
        print(f'✓ Historial exportado: {filename}')
