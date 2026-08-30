from qual2k.core.model import Q2KModel
from pathlib import Path
import pygad
import warnings
import os
import glob
import shutil
import tempfile
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import pandas as pd

warnings.filterwarnings('ignore')

# Configuración de estilo para publicación
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


class Calibracion:
    """
    Clase para calibración automática de parámetros de QUAL2K usando algoritmos genéticos.
    """

    def __init__(
            self,
            filepath: str,
            header_dict: Dict[str, Any],
            parametros: Dict[str, Tuple[float, float, bool]],
            # Parámetros básicos del GA
            num_generations: int = 100,
            population_size: int = 40,
            num_parents_mating: int = 16,
            # Parámetros de selección de padres
            parent_selection_type: str = "tournament",
            k_tournament: int = 3,
            # Parámetros de cruce
            crossover_type: str = "single_point",
            crossover_probability: float = 0.9,
            # Parámetros de mutación
            mutation_type: str = "random",
            mutation_probability: float = 0.2,
            mutation_percent_genes: Union[int, float] = 20,
            mutation_by_replacement: bool = False,
            random_mutation_min_val: Optional[float] = None,
            random_mutation_max_val: Optional[float] = None,
            # Parámetros de elitismo
            keep_elitism: int = 3,
            keep_parents: int = -1,
            # Criterios de parada
            stop_criteria: Optional[Union[str, List[str]]] = None,
            # Parámetros de diversidad
            allow_duplicate_genes: bool = True,
            # Semilla aleatoria para reproducibilidad
            random_seed: Optional[int] = None,
            # Parámetros de paralelismo
            num_workers: Optional[int] = None,
            usar_paralelo: bool = True,
            # Parámetros adicionales de Q2K
            q_cabecera: float = 1.06007E-06
    ):
        """
        Inicializa la clase de calibración.

        Args:
            filepath: Ruta al directorio con las plantillas
            header_dict: Diccionario con configuración del modelo
            parametros: Dict con {nombre_param: (min, max, es_global)}

            # Parámetros básicos del GA
            num_generations: Número de generaciones del GA
            population_size: Tamaño de la población
            num_parents_mating: Número de padres para apareamiento

            # Parámetros de selección de padres
            parent_selection_type: Tipo de selección ('sss', 'rws', 'sus', 'rank', 'random', 'tournament')
            k_tournament: Número de soluciones en torneo (solo para 'tournament')

            # Parámetros de cruce
            crossover_type: Tipo de cruce ('single_point', 'two_points', 'uniform', 'scattered')
            crossover_probability: Probabilidad de cruce (0.0 a 1.0)

            # Parámetros de mutación
            mutation_type: Tipo de mutación ('random', 'swap', 'inversion', 'scramble', 'adaptive')
            mutation_probability: Probabilidad de mutación (0.0 a 1.0)
            mutation_percent_genes: Porcentaje o número de genes a mutar
            mutation_by_replacement: Si reemplazar o sumar/restar en mutación
            random_mutation_min_val: Valor mínimo para mutación aleatoria
            random_mutation_max_val: Valor máximo para mutación aleatoria

            # Parámetros de elitismo
            keep_elitism: Número de mejores soluciones a mantener
            keep_parents: Número de padres a mantener (-1 = todos)

            # Criterios de parada
            stop_criteria: Criterios de parada ('reach_XXX', 'saturate_XXX')
                          Ejemplos: 'reach_0.95', 'saturate_50'

            # Parámetros de diversidad
            allow_duplicate_genes: Si permitir genes duplicados

            # Semilla aleatoria
            random_seed: Semilla para reproducibilidad (None = aleatorio)

            # Parámetros de paralelismo
            num_workers: Número de workers paralelos (None = auto)
            usar_paralelo: Si usar procesamiento paralelo

            # Parámetros adicionales de Q2K
            q_cabecera: Caudal de cabecera para el modelo
        """
        # Configuración del modelo
        self.filepath = filepath
        self.header_dict = header_dict
        self.parametros = parametros
        self.q_cabecera = q_cabecera

        # Parámetros básicos del GA
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents_mating = num_parents_mating

        # Parámetros de selección
        self.parent_selection_type = parent_selection_type
        self.k_tournament = k_tournament

        # Parámetros de cruce
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability

        # Parámetros de mutación
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_by_replacement = mutation_by_replacement
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val

        # Parámetros de elitismo
        self.keep_elitism = keep_elitism
        self.keep_parents = keep_parents

        # Criterios de parada
        self.stop_criteria = stop_criteria

        # Parámetros de diversidad
        self.allow_duplicate_genes = allow_duplicate_genes

        # Semilla aleatoria
        self.random_seed = random_seed

        # Configuración de paralelismo
        self.usar_paralelo = usar_paralelo
        self.num_workers = num_workers or min(4, mp.cpu_count() - 1)
        self.pool = None

        # Estado de la calibración
        self.contador_evaluaciones = 0
        self.mejor_kge = -999.0
        self.n_reaches = 0
        self.gene_space = []
        self.param_map = []
        self.ga_instance = None
        self.mejor_solucion = None
        self.historial_generaciones = []
        self.historial_poblacion = []  # Nuevo: guardar estadísticas de cada generación

    def _inicializar_modelo(self) -> Q2KModel:
        """Crea una instancia temporal del modelo para obtener configuración."""
        model = Q2KModel(self.filepath, self.header_dict)
        model.cargar_plantillas('PlantillaBaseQ2K.xlsx')
        return model

    def _configurar_genes(self):
        """Configura el espacio de genes y el mapeo de parámetros."""
        self.gene_space = []
        self.param_map = []

        for param_name, (min_val, max_val, is_global) in self.parametros.items():
            if is_global:
                self.gene_space.append({'low': min_val, 'high': max_val})
                self.param_map.append((param_name, None))
            else:
                for i in range(self.n_reaches):
                    self.gene_space.append({'low': min_val, 'high': max_val})
                    self.param_map.append((param_name, i))

        return len(self.gene_space)

    def _decodificar_solucion(self, solution: List[float]) -> Dict[str, List[float]]:
        """
        Decodifica una solución del GA en parámetros del modelo.

        Args:
            solution: Array de valores de genes

        Returns:
            Diccionario con listas de parámetros por reach
        """
        params = {
            'kaaa': [None] * self.n_reaches,
            'khc': [None] * self.n_reaches,
            'kdcs': [None] * self.n_reaches,
            'kdc': [None] * self.n_reaches,
            'khn': [None] * self.n_reaches,
            'kn': [None] * self.n_reaches,
            'ki': [None] * self.n_reaches,
            'khp': [None] * self.n_reaches,
            'kdt': [None] * self.n_reaches,
        }

        for gene_idx, (param_name, reach_idx) in enumerate(self.param_map):
            valor = solution[gene_idx]
            if reach_idx is None:
                params[param_name] = [valor] * self.n_reaches
            else:
                params[param_name][reach_idx] = valor

        return params

    @staticmethod
    def _evaluar_solucion_worker(args: Tuple) -> Tuple[int, float]:
        """
        Evalúa una solución en un worker paralelo.
        Esta función debe ser estática para ser serializable.
        """
        solution, eval_id, filepath, header_dict, param_map, n_reaches, q_cabecera = args

        temp_dir = tempfile.mkdtemp(prefix=f'q2k_eval_{eval_id}_')

        try:
            # Copiar plantilla y archivos necesarios
            plantilla_origen = os.path.join(filepath, 'PlantillaBaseQ2K.xlsx')
            plantilla_destino = os.path.join(temp_dir, 'PlantillaBaseQ2K.xlsx')
            shutil.copy2(plantilla_origen, plantilla_destino)

            for archivo in glob.glob(os.path.join(filepath, '*')):
                if os.path.isfile(archivo):
                    nombre = os.path.basename(archivo)
                    if not any(nombre.endswith(ext) for ext in ['.out', '.txt', '.dat', '.q2k']):
                        try:
                            shutil.copy2(archivo, os.path.join(temp_dir, nombre))
                        except:
                            pass

            # Configurar y ejecutar modelo
            header_dict_temp = header_dict.copy()
            header_dict_temp['filedir'] = temp_dir

            model = Q2KModel(temp_dir, header_dict_temp)
            model.cargar_plantillas('PlantillaBaseQ2K.xlsx')

            # Decodificar parámetros
            params = {
                'kaaa': [None] * n_reaches,
                'khc': [None] * n_reaches,
                'kdcs': [None] * n_reaches,
                'kdc': [None] * n_reaches,
                'khn': [None] * n_reaches,
                'kn': [None] * n_reaches,
                'ki': [None] * n_reaches,
                'khp': [None] * n_reaches,
                'kdt': [None] * n_reaches,
            }

            for gene_idx, (param_name, reach_idx) in enumerate(param_map):
                valor = solution[gene_idx]
                if reach_idx is None:
                    params[param_name] = [valor] * n_reaches
                else:
                    params[param_name][reach_idx] = valor

            reach_rates_custom = model.config.generar_reach_rates_custom(
                n=n_reaches,
                kaaa_list=params['kaaa'],
                khc_list=params['khc'],
                kdcs_list=params['kdcs'],
                kdc_list=params['kdc'],
                khn_list=params['khn'],
                kn_list=params['kn'],
                ki_list=params['ki'],
                khp_list=params['khp'],
                kdt_list=params['kdt']
            )

            model.configurar_modelo(reach_rates_custom=reach_rates_custom, q_cabecera=q_cabecera)
            model.generar_archivo_q2k()
            model.ejecutar_simulacion()
            model.analizar_resultados(generar_graficas=False)
            resultados, kge_global = model.calcular_metricas_calibracion(pesos = {
                "water_temp_c": 0.05,
                "conductivity": 0.05,
                "nitrate": 0.05,
                "pathogen": 0.10,
                "pH": 0.05,
                "total_suspended_solids": 0.05,
                "dissolved_oxygen": 0.30,
                "carbonaceous_bod_fast": 0.30,
                "total_kjeldahl_nitrogen": 0.02,
                "ammonium": 0.02,
                "total_phosphorus": 0.01
            })

            return (eval_id, kge_global)

        except Exception as e:
            print(f'Error en evaluación {eval_id}: {e}')
            return (eval_id, -999)

        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    def _fitness_function(self, ga, solution, solution_idx):
        """Función de fitness para el algoritmo genético."""
        self.contador_evaluaciones += 1
        eval_id = self.contador_evaluaciones

        args = (solution, eval_id, self.filepath, self.header_dict,
                self.param_map, self.n_reaches, self.q_cabecera)

        if self.usar_paralelo and self.pool is not None:
            resultado = self.pool.apply_async(self._evaluar_solucion_worker, (args,))
            eval_id_result, kge = resultado.get(timeout=300)
        else:
            eval_id_result, kge = self._evaluar_solucion_worker(args)

        if kge > self.mejor_kge:
            self.mejor_kge = kge
            print(f"  *** Eval {eval_id} | NUEVO MEJOR KGE: {kge:.4f} ***")
        elif eval_id % 5 == 0:
            print(f"Eval {eval_id} | KGE: {kge:.4f}")

        return kge

    def _on_generation(self, ga):
        """Callback ejecutado al completar cada generación."""
        gen = ga.generations_completed

        # Obtener estadísticas de la población actual
        population_fitness = ga.last_generation_fitness
        best_solution, best_fitness, _ = ga.best_solution()

        # Calcular estadísticas
        stats = {
            'generacion': gen,
            'mejor_fitness': best_fitness,
            'mejor_global': self.mejor_kge,
            'promedio': np.mean(population_fitness),
            'mediana': np.median(population_fitness),
            'std': np.std(population_fitness),
            'min': np.min(population_fitness),
            'max': np.max(population_fitness),
            'q25': np.percentile(population_fitness, 25),
            'q75': np.percentile(population_fitness, 75),
        }

        # Guardar historial
        self.historial_generaciones.append(stats)
        self.historial_poblacion.append(population_fitness.copy())

        print(f'\n{"=" * 60}')
        print(f'GENERACIÓN {gen} COMPLETADA')
        print(f'Mejor KGE de esta generación: {best_fitness:.4f}')
        print(f'Promedio poblacional: {stats["promedio"]:.4f} ± {stats["std"]:.4f}')
        print(f'Mejor KGE global: {self.mejor_kge:.4f}')
        print("=" * 60 + '\n')

    def plotear_evolucion_fitness(
            self,
            filename: str = 'evolucion_fitness.png',
            dpi: int = 300,
            formato: str = 'png',
            mostrar: bool = False,
            estilo: str = 'seaborn-v0_8-paper'
    ):
        """
        Genera una gráfica de calidad científica de la evolución del fitness.

        Args:
            filename: Nombre del archivo de salida
            dpi: Resolución de la imagen (300 para publicación)
            formato: Formato de salida ('png', 'pdf', 'svg', 'eps')
            mostrar: Si mostrar la gráfica además de guardarla
            estilo: Estilo de matplotlib a usar
        """
        if not self.historial_generaciones:
            print("No hay datos de calibración disponibles para graficar.")
            return

        # Convertir historial a DataFrame
        df = pd.DataFrame(self.historial_generaciones)

        # Configurar estilo
        try:
            plt.style.use(estilo)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')

        # Crear figura con subplots
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # ==================================================================
        # 1. Gráfica principal: Evolución del mejor fitness
        # ==================================================================
        ax1 = fig.add_subplot(gs[0, :])

        generaciones = df['generacion']
        ax1.plot(generaciones, df['mejor_fitness'], 'o-',
                 linewidth=2, markersize=4, color='#2E86AB',
                 label='Mejor de la generación', alpha=0.8)
        ax1.plot(generaciones, df['mejor_global'], 's-',
                 linewidth=2.5, markersize=4, color='#A23B72',
                 label='Mejor global', alpha=0.9)
        ax1.fill_between(generaciones, df['promedio'] - df['std'],
                         df['promedio'] + df['std'],
                         alpha=0.2, color='#F18F01', label='Promedio ± 1σ')
        ax1.plot(generaciones, df['promedio'], '--',
                 linewidth=1.5, color='#F18F01', alpha=0.7)

        ax1.set_xlabel('Generación', fontweight='bold')
        ax1.set_ylabel('KGE (Fitness)', fontweight='bold')
        ax1.set_title('Evolución del Fitness Durante la Calibración',
                      fontweight='bold', pad=15)
        ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)

        # Añadir anotación del mejor valor
        best_idx = df['mejor_global'].idxmax()
        ax1.annotate(f'Mejor KGE: {df.loc[best_idx, "mejor_global"]:.4f}\n(Gen. {df.loc[best_idx, "generacion"]})',
                     xy=(df.loc[best_idx, 'generacion'], df.loc[best_idx, 'mejor_global']),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # ==================================================================
        # 2. Distribución del fitness por generación (box plot)
        # ==================================================================
        ax2 = fig.add_subplot(gs[1, 0])

        # Seleccionar algunas generaciones para el box plot
        n_boxes = min(15, len(self.historial_poblacion))
        indices = np.linspace(0, len(self.historial_poblacion) - 1, n_boxes, dtype=int)
        data_boxes = [self.historial_poblacion[i] for i in indices]
        positions = [self.historial_generaciones[i]['generacion'] for i in indices]

        bp = ax2.boxplot(data_boxes, positions=positions, widths=max(2, self.num_generations / 30),
                         patch_artist=True, showfliers=True)

        # Colorear boxes
        for patch in bp['boxes']:
            patch.set_facecolor('#A8DADC')
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('#E63946')
            median.set_linewidth(2)

        ax2.set_xlabel('Generación', fontweight='bold')
        ax2.set_ylabel('Distribución de Fitness', fontweight='bold')
        ax2.set_title('Distribución Poblacional del Fitness', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        # ==================================================================
        # 3. Mejora relativa entre generaciones
        # ==================================================================
        ax3 = fig.add_subplot(gs[1, 1])

        mejora = df['mejor_global'].diff().fillna(0)
        mejora_acumulada = (df['mejor_global'] - df['mejor_global'].iloc[0])

        ax3.bar(generaciones, mejora, color='#06FFA5', alpha=0.6,
                label='Mejora por generación', edgecolor='black', linewidth=0.5)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(generaciones, mejora_acumulada, 'o-',
                      color='#E63946', linewidth=2, markersize=4,
                      label='Mejora acumulada')

        ax3.set_xlabel('Generación', fontweight='bold')
        ax3.set_ylabel('Δ KGE (Mejora)', fontweight='bold', color='#06FFA5')
        ax3_twin.set_ylabel('Mejora Acumulada', fontweight='bold', color='#E63946')
        ax3.set_title('Mejora del Fitness', fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.tick_params(axis='y', labelcolor='#06FFA5')
        ax3_twin.tick_params(axis='y', labelcolor='#E63946')

        # Leyendas
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # ==================================================================
        # 4. Convergencia (log scale)
        # ==================================================================
        ax4 = fig.add_subplot(gs[2, 0])

        gap = np.abs(df['mejor_global'] - df['mejor_fitness'])
        ax4.semilogy(generaciones, gap, 'o-', linewidth=2, markersize=4,
                     color='#457B9D', label='Gap (Mejor global - Mejor gen.)')
        ax4.semilogy(generaciones, df['std'], 's-', linewidth=2, markersize=4,
                     color='#F4A261', label='Desv. estándar poblacional')

        ax4.set_xlabel('Generación', fontweight='bold')
        ax4.set_ylabel('Valor (escala log)', fontweight='bold')
        ax4.set_title('Convergencia del Algoritmo', fontweight='bold')
        ax4.legend(loc='best', frameon=True)
        ax4.grid(True, alpha=0.3, linestyle='--', which='both')

        # ==================================================================
        # 5. Tabla resumen de estadísticas
        # ==================================================================
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        # Calcular estadísticas finales
        stats_finales = {
            'Mejor KGE alcanzado': f"{df['mejor_global'].max():.6f}",
            'KGE inicial': f"{df['mejor_global'].iloc[0]:.6f}",
            'Mejora total': f"{df['mejor_global'].max() - df['mejor_global'].iloc[0]:.6f}",
            'Generación óptima': f"{df.loc[df['mejor_global'].idxmax(), 'generacion']:.0f}",
            'Promedio final': f"{df['promedio'].iloc[-1]:.6f}",
            'Desv. std. final': f"{df['std'].iloc[-1]:.6f}",
            'Total evaluaciones': f"{self.contador_evaluaciones}",
            'Generaciones': f"{len(df)}",
        }

        # Crear tabla
        tabla_data = [[k, v] for k, v in stats_finales.items()]
        table = ax5.table(cellText=tabla_data, colLabels=['Métrica', 'Valor'],
                          cellLoc='left', loc='center',
                          colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Estilizar tabla
        for i in range(len(tabla_data) + 1):
            if i == 0:
                table[(i, 0)].set_facecolor('#457B9D')
                table[(i, 1)].set_facecolor('#457B9D')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    table[(i, 0)].set_facecolor('#E8F4F8')
                    table[(i, 1)].set_facecolor('#E8F4F8')

        ax5.set_title('Resumen de Estadísticas', fontweight='bold', pad=10)

        # ==================================================================
        # Título general y metadatos
        # ==================================================================
        fig.suptitle(f'Análisis de Calibración - Algoritmo Genético\n' +
                     f'Población: {self.population_size} | Generaciones: {self.num_generations} | ' +
                     f'Parámetros: {len(self.gene_space)}',
                     fontsize=14, fontweight='bold', y=0.98)

        # Guardar figura
        output_path = os.path.join(self.filepath, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    format=formato, facecolor='white', edgecolor='none')

        print(f'\n{"=" * 60}')
        print(f'Gráfica guardada en: {output_path}')
        print(f'Resolución: {dpi} DPI')
        print(f'Formato: {formato.upper()}')
        print("=" * 60)

        if mostrar:
            plt.show()
        else:
            plt.close()

    def plotear_fitness_simple(
            self,
            filename: str = 'fitness_simple.png',
            dpi: int = 300,
            formato: str = 'png',
            mostrar: bool = False
    ):
        """
        Genera una gráfica simple y clara de la evolución del fitness.
        Ideal para presentaciones.

        Args:
            filename: Nombre del archivo de salida
            dpi: Resolución de la imagen
            formato: Formato de salida
            mostrar: Si mostrar la gráfica
        """
        if not self.historial_generaciones:
            print("No hay datos de calibración disponibles para graficar.")
            return

        df = pd.DataFrame(self.historial_generaciones)

        # Crear figura simple
        fig, ax = plt.subplots(figsize=(10, 6))

        generaciones = df['generacion']

        # Línea principal
        ax.plot(generaciones, df['mejor_global'], 'o-',
                linewidth=3, markersize=6, color='#E63946',
                label='Mejor fitness global', markerfacecolor='white',
                markeredgewidth=2)

        # Banda de confianza
        ax.fill_between(generaciones,
                        df['promedio'] - df['std'],
                        df['promedio'] + df['std'],
                        alpha=0.3, color='#457B9D')

        ax.plot(generaciones, df['promedio'], '--',
                linewidth=2, color='#457B9D', alpha=0.7,
                label='Promedio poblacional')

        ax.set_xlabel('Generación', fontsize=14, fontweight='bold')
        ax.set_ylabel('KGE (Kling-Gupta Efficiency)', fontsize=14, fontweight='bold')
        ax.set_title('Evolución del Proceso de Calibración',
                     fontsize=16, fontweight='bold', pad=20)

        ax.legend(loc='lower right', fontsize=12, frameon=True,
                  fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=-1)

        # Anotación final
        valor_final = df['mejor_global'].iloc[-1]
        ax.text(0.02, 0.98, f'KGE Final: {valor_final:.4f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = os.path.join(self.filepath, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    format=formato, facecolor='white')

        print(f'Gráfica simple guardada en: {output_path}')

        if mostrar:
            plt.show()
        else:
            plt.close()

    def exportar_historial_csv(self, filename: str = 'historial_calibracion.csv'):
        """
        Exporta el historial de calibración a un archivo CSV.

        Args:
            filename: Nombre del archivo CSV
        """
        if not self.historial_generaciones:
            print("No hay datos de calibración disponibles para exportar.")
            return

        df = pd.DataFrame(self.historial_generaciones)
        output_path = os.path.join(self.filepath, filename)
        df.to_csv(output_path, index=False, float_format='%.6f')

        print(f'Historial exportado a: {output_path}')

    def _imprimir_configuracion(self, num_genes: int):
        """Imprime la configuración de la calibración."""
        print('\n' + '=' * 80)
        print('CALIBRACIÓN AUTOMÁTICA DE QUAL2K CON ALGORITMO GENÉTICO')
        print('=' * 80)
        print(f'\nNúmero de reaches: {self.n_reaches}')
        print(f'Modo: {"PARALELO" if self.usar_paralelo else "SERIAL"}')
        if self.usar_paralelo:
            print(f'Workers: {self.num_workers}')
        if self.random_seed is not None:
            print(f'Semilla aleatoria: {self.random_seed}')

        print(f'\n{"=" * 80}')
        print('CONFIGURACIÓN DE PARÁMETROS A CALIBRAR')
        print("=" * 80)
        for param_name, (min_val, max_val, is_global) in self.parametros.items():
            tipo = "GLOBAL (1 valor)" if is_global else f"POR TRAMO ({self.n_reaches} valores)"
            print(f'{param_name:8s} [{min_val:8.3f}, {max_val:8.3f}] → {tipo}')
        print(f'\nTotal de genes a calibrar: {num_genes}')
        print("=" * 80)

        print('\n' + '=' * 80)
        print('CONFIGURACIÓN DEL ALGORITMO GENÉTICO')
        print('=' * 80)
        print(f'\nParámetros básicos:')
        print(f'  • Generaciones: {self.num_generations}')
        print(f'  • Tamaño de población: {self.population_size}')
        print(f'  • Padres para apareamiento: {self.num_parents_mating}')

        print(f'\nSelección de padres:')
        print(f'  • Tipo: {self.parent_selection_type}')
        if self.parent_selection_type == "tournament":
            print(f'  • K-Tournament: {self.k_tournament}')

        print(f'\nCruce (Crossover):')
        print(f'  • Tipo: {self.crossover_type}')
        print(f'  • Probabilidad: {self.crossover_probability}')

        print(f'\nMutación:')
        print(f'  • Tipo: {self.mutation_type}')
        print(f'  • Probabilidad: {self.mutation_probability}')
        print(f'  • Genes a mutar: {self.mutation_percent_genes}%' if isinstance(self.mutation_percent_genes,
                                                                                 int) else f'  • Genes a mutar: {self.mutation_percent_genes}')
        print(f'  • Por reemplazo: {"Sí" if self.mutation_by_replacement else "No"}')

        print(f'\nElitismo y preservación:')
        print(f'  • Mantener élite: {self.keep_elitism}')
        print(f'  • Mantener padres: {self.keep_parents if self.keep_parents != -1 else "Todos"}')

        if self.stop_criteria:
            print(f'\nCriterios de parada:')
            if isinstance(self.stop_criteria, list):
                for criterio in self.stop_criteria:
                    print(f'  • {criterio}')
            else:
                print(f'  • {self.stop_criteria}')

        print(f'\nDiversidad:')
        print(f'  • Genes duplicados: {"Permitidos" if self.allow_duplicate_genes else "No permitidos"}')

        print('=' * 80)

    def _imprimir_resultados(self, solution, solution_fitness):
        """Imprime los resultados de la calibración."""
        print('\n' + '=' * 80)
        print('CALIBRACIÓN COMPLETADA')
        print('=' * 80)
        print(f'\nMejor KGE encontrado: {solution_fitness:.4f}')
        print(f'Total de evaluaciones: {self.contador_evaluaciones}')
        print(f'Generaciones completadas: {len(self.historial_generaciones)}')

        print(f'\n{"=" * 80}')
        print('PARÁMETROS ÓPTIMOS')
        print('=' * 80)

        gene_idx = 0
        for param_name, (min_val, max_val, is_global) in self.parametros.items():
            if is_global:
                valor = solution[gene_idx]
                print(f'{param_name:8s} (global):  {valor:.6f}')
                gene_idx += 1
            else:
                print(f'{param_name:8s} (por tramo):')
                for i in range(self.n_reaches):
                    valor = solution[gene_idx]
                    print(f'  Reach {i + 1}: {valor:.6f}')
                    gene_idx += 1
        print('=' * 80)

    def _guardar_resultados(self, solution, kge_final):
        """Guarda los resultados en un archivo de texto."""
        output_file = os.path.join(self.filepath, 'parametros_calibrados.txt')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('RESULTADOS DE CALIBRACIÓN - ALGORITMO GENÉTICO\n')
            f.write('=' * 80 + '\n\n')

            # Resultados generales
            f.write('RESULTADOS GENERALES:\n')
            f.write('-' * 80 + '\n')
            f.write(f'KGE Final: {kge_final:.6f}\n')
            f.write(f'Total de evaluaciones: {self.contador_evaluaciones}\n')
            f.write(f'Generaciones completadas: {len(self.historial_generaciones)}\n')
            if self.random_seed is not None:
                f.write(f'Semilla aleatoria: {self.random_seed}\n')
            f.write('\n')

            # Configuración del GA
            f.write('CONFIGURACIÓN DEL ALGORITMO GENÉTICO:\n')
            f.write('-' * 80 + '\n')
            f.write(f'Generaciones: {self.num_generations}\n')
            f.write(f'Tamaño de población: {self.population_size}\n')
            f.write(f'Padres para apareamiento: {self.num_parents_mating}\n')
            f.write(f'Tipo de selección: {self.parent_selection_type}\n')
            if self.parent_selection_type == "tournament":
                f.write(f'K-Tournament: {self.k_tournament}\n')
            f.write(f'Tipo de cruce: {self.crossover_type}\n')
            f.write(f'Probabilidad de cruce: {self.crossover_probability}\n')
            f.write(f'Tipo de mutación: {self.mutation_type}\n')
            f.write(f'Probabilidad de mutación: {self.mutation_probability}\n')
            f.write(f'Porcentaje de genes a mutar: {self.mutation_percent_genes}\n')
            f.write(f'Mantener élite: {self.keep_elitism}\n')
            f.write('\n')

            # Parámetros óptimos
            f.write('PARÁMETROS ÓPTIMOS:\n')
            f.write('-' * 80 + '\n')

            gene_idx = 0
            for param_name, (min_val, max_val, is_global) in self.parametros.items():
                if is_global:
                    valor = solution[gene_idx]
                    f.write(f'{param_name:8s} (global):  {valor:.6f}\n')
                    gene_idx += 1
                else:
                    f.write(f'{param_name:8s} (por tramo):\n')
                    for i in range(self.n_reaches):
                        valor = solution[gene_idx]
                        f.write(f'  Reach {i + 1}: {valor:.6f}\n')
                        gene_idx += 1

            # Historial de generaciones
            f.write('\n')
            f.write('HISTORIAL DE GENERACIONES:\n')
            f.write('-' * 80 + '\n')
            f.write(f'{"Gen":>5} | {"Mejor KGE Gen":>15} | {"Mejor KGE Global":>18} | {"Promedio":>12} | {"Std":>10}\n')
            f.write('-' * 80 + '\n')
            for hist in self.historial_generaciones:
                f.write(f'{hist["generacion"]:5d} | {hist["mejor_fitness"]:15.6f} | '
                        f'{hist["mejor_global"]:18.6f} | {hist["promedio"]:12.6f} | {hist["std"]:10.6f}\n')

        print(f'\nResultados guardados en: {output_file}')

    def _simular_con_mejor_solucion(self, solution):
        """Ejecuta una simulación final con los parámetros óptimos."""
        print('\n' + '=' * 80)
        print('SIMULACIÓN FINAL CON PARÁMETROS ÓPTIMOS')
        print('=' * 80)

        model_final = Q2KModel(self.filepath, self.header_dict)
        model_final.cargar_plantillas('PlantillaBaseQ2K.xlsx')

        params_final = self._decodificar_solucion(solution)

        reach_rates_final = model_final.config.generar_reach_rates_custom(
            n=self.n_reaches,
            kaaa_list=params_final['kaaa'],
            khc_list=params_final['khc'],
            kdcs_list=params_final['kdcs'],
            kdc_list=params_final['kdc'],
            khn_list=params_final['khn'],
            kn_list=params_final['kn'],
            ki_list=params_final['ki'],
            khp_list=params_final['khp'],
            kdt_list=params_final['kdt']
        )

        model_final.configurar_modelo(reach_rates_custom=reach_rates_final, q_cabecera=self.q_cabecera)
        model_final.generar_archivo_q2k()
        model_final.ejecutar_simulacion()
        model_final.analizar_resultados(generar_graficas=True)
        resultados_final, kge_final = model_final.calcular_metricas_calibracion(pesos = {
                "water_temp_c": 0.05,
                "conductivity": 0.05,
                "nitrate": 0.05,
                "pathogen": 0.10,
                "pH": 0.05,
                "total_suspended_solids": 0.05,
                "dissolved_oxygen": 0.30,
                "carbonaceous_bod_fast": 0.30,
                "total_kjeldahl_nitrogen": 0.02,
                "ammonium": 0.02,
                "total_phosphorus": 0.01
            })

        print(f'\nKGE final verificado: {kge_final:.4f}')

        return kge_final

    def ejecutar(self, generar_graficas: bool = True) -> Optional[Tuple[List[float], float]]:
        """
        Ejecuta el proceso completo de calibración.

        Args:
            generar_graficas: Si generar gráficas al finalizar

        Returns:
            Tupla con (mejor_solución, mejor_kge) o None si se interrumpe
        """
        # Inicializar
        model_temp = self._inicializar_modelo()
        self.n_reaches = len(model_temp.data_reaches)
        num_genes = self._configurar_genes()

        self._imprimir_configuracion(num_genes)

        # Crear pool de workers
        if self.usar_paralelo:
            self.pool = mp.Pool(processes=self.num_workers)
            print(f'\nPool de {self.num_workers} workers creado')

        # Construir argumentos del GA
        ga_kwargs = {
            'num_generations': self.num_generations,
            'num_parents_mating': self.num_parents_mating,
            'fitness_func': self._fitness_function,
            'sol_per_pop': self.population_size,
            'num_genes': num_genes,
            'gene_space': self.gene_space,
            'parent_selection_type': self.parent_selection_type,
            'crossover_type': self.crossover_type,
            'crossover_probability': self.crossover_probability,
            'mutation_type': self.mutation_type,
            'mutation_probability': self.mutation_probability,
            'mutation_percent_genes': self.mutation_percent_genes,
            'mutation_by_replacement': self.mutation_by_replacement,
            'keep_elitism': self.keep_elitism,
            'keep_parents': self.keep_parents,
            'allow_duplicate_genes': self.allow_duplicate_genes,
            'on_generation': self._on_generation,
        }

        # Agregar parámetros opcionales
        if self.parent_selection_type == "tournament":
            ga_kwargs['K_tournament'] = self.k_tournament

        if self.random_mutation_min_val is not None:
            ga_kwargs['random_mutation_min_val'] = self.random_mutation_min_val
        if self.random_mutation_max_val is not None:
            ga_kwargs['random_mutation_max_val'] = self.random_mutation_max_val

        if self.stop_criteria is not None:
            ga_kwargs['stop_criteria'] = self.stop_criteria

        if self.random_seed is not None:
            ga_kwargs['random_seed'] = self.random_seed

        # Configurar algoritmo genético
        self.ga_instance = pygad.GA(**ga_kwargs)

        # Ejecutar calibración
        print(f'\n{"=" * 80}')
        print('INICIANDO CALIBRACIÓN')
        print('=' * 80 + '\n')

        try:
            self.ga_instance.run()

            # Obtener mejor solución
            solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
            self.mejor_solucion = solution

            self._imprimir_resultados(solution, solution_fitness)

        except KeyboardInterrupt:
            print('\n\n¡CALIBRACIÓN INTERRUMPIDA POR USUARIO!\n')
            solution = None
            solution_fitness = None

        finally:
            # Cerrar pool
            if self.usar_paralelo and self.pool is not None:
                print('\nCerrando pool de workers...')
                self.pool.close()
                self.pool.join()
                print('Pool cerrado correctamente')

        # Simulación final
        if solution is not None:
            kge_final = self._simular_con_mejor_solucion(solution)
            self._guardar_resultados(solution, kge_final)

            # Generar gráficas
            if generar_graficas:
                print('\n' + '=' * 80)
                print('GENERANDO GRÁFICAS')
                print('=' * 80)
                self.plotear_evolucion_fitness(formato='png', dpi=300)
                self.plotear_evolucion_fitness(formato='pdf', filename='evolucion_fitness.pdf')
                self.plotear_fitness_simple()
                self.exportar_historial_csv()

            print('\n' + '=' * 80)
            print('PROCESO FINALIZADO')
            print('=' * 80 + '\n')

            return (solution, kge_final)

        return None

    def get_mejor_solucion(self) -> Optional[List[float]]:
        """Retorna la mejor solución encontrada."""
        return self.mejor_solucion

    def get_parametros_calibrados(self) -> Optional[Dict[str, List[float]]]:
        """Retorna los parámetros calibrados en formato de diccionario."""
        if self.mejor_solucion is None:
            return None
        return self._decodificar_solucion(self.mejor_solucion)

    def get_historial(self) -> List[Dict[str, Any]]:
        """Retorna el historial de generaciones."""
        return self.historial_generaciones

    def exportar_configuracion(self, filename: str = 'config_calibracion.txt'):
        """
        Exporta la configuración actual a un archivo.

        Args:
            filename: Nombre del archivo de salida
        """
        output_path = os.path.join(self.filepath, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('CONFIGURACIÓN DE CALIBRACIÓN\n')
            f.write('=' * 80 + '\n\n')

            f.write('PARÁMETROS BÁSICOS:\n')
            f.write(f'  num_generations = {self.num_generations}\n')
            f.write(f'  population_size = {self.population_size}\n')
            f.write(f'  num_parents_mating = {self.num_parents_mating}\n\n')

            f.write('SELECCIÓN DE PADRES:\n')
            f.write(f'  parent_selection_type = "{self.parent_selection_type}"\n')
            f.write(f'  k_tournament = {self.k_tournament}\n\n')

            f.write('CRUCE:\n')
            f.write(f'  crossover_type = "{self.crossover_type}"\n')
            f.write(f'  crossover_probability = {self.crossover_probability}\n\n')

            f.write('MUTACIÓN:\n')
            f.write(f'  mutation_type = "{self.mutation_type}"\n')
            f.write(f'  mutation_probability = {self.mutation_probability}\n')
            f.write(f'  mutation_percent_genes = {self.mutation_percent_genes}\n')
            f.write(f'  mutation_by_replacement = {self.mutation_by_replacement}\n\n')

            f.write('ELITISMO:\n')
            f.write(f'  keep_elitism = {self.keep_elitism}\n')
            f.write(f'  keep_parents = {self.keep_parents}\n\n')

            f.write('OTROS:\n')
            f.write(f'  random_seed = {self.random_seed}\n')
            f.write(f'  allow_duplicate_genes = {self.allow_duplicate_genes}\n')
            f.write(f'  stop_criteria = {self.stop_criteria}\n')

        print(f'Configuración exportada a: {output_path}')


# ============================================================================
# PRESETS DE CONFIGURACIÓN
# ============================================================================
class CalibracionPresets:
    """Presets comunes de configuración para calibración."""

    @staticmethod
    def exploracion_rapida():
        """Configuración para exploración rápida (pocas generaciones)."""
        return {
            'num_generations': 20,
            'population_size': 20,
            'num_parents_mating': 8,
            'mutation_probability': 0.3,
            'crossover_probability': 0.8,
        }

    @staticmethod
    def balanceado():
        """Configuración balanceada (por defecto)."""
        return {
            'num_generations': 100,
            'population_size': 40,
            'num_parents_mating': 16,
            'mutation_probability': 0.2,
            'crossover_probability': 0.9,
        }

    @staticmethod
    def intensivo():
        """Configuración intensiva (muchas generaciones)."""
        return {
            'num_generations': 200,
            'population_size': 80,
            'num_parents_mating': 32,
            'mutation_probability': 0.15,
            'crossover_probability': 0.95,
            'keep_elitism': 5,
        }

    @staticmethod
    def alta_diversidad():
        """Configuración con alta diversidad genética."""
        return {
            'num_generations': 100,
            'population_size': 60,
            'num_parents_mating': 20,
            'mutation_probability': 0.4,
            'mutation_percent_genes': 30,
            'crossover_type': 'uniform',
        }

    @staticmethod
    def convergencia_rapida():
        """Configuración para convergencia rápida."""
        return {
            'num_generations': 50,
            'population_size': 30,
            'num_parents_mating': 15,
            'mutation_probability': 0.1,
            'keep_elitism': 5,
            'parent_selection_type': 'rank',
            'stop_criteria': ['saturate_20'],
        }


