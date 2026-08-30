# calibrator_pipeline.py
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import pandas as pd
import warnings

from qual2k.core.calibrator import Calibracion

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


class CalibracionPipeline:
    """
    Pipeline para calibración secuencial de múltiples modelos QUAL2K.
    Calibra primero Canal Vargas, luego Tramo 3S, y finalmente Chicamocha.
    """

    def __init__(
            self,
            # Configuraciones de los modelos
            config_vargas: Dict[str, Any],
            config_tramo3s: Dict[str, Any],
            config_chicamocha: Dict[str, Any],
            # Parámetros a calibrar
            parametros_vargas: Dict[str, Tuple[float, float, bool]],
            parametros_tramo3s: Dict[str, Tuple[float, float, bool]],
            parametros_chicamocha: Dict[str, Tuple[float, float, bool]],
            # Parámetros del GA (compartidos)
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
            usar_paralelo: bool = True
    ):
        """
        Inicializa el pipeline de calibración.

        Args:
            config_vargas: Configuración del modelo Canal Vargas
            config_tramo3s: Configuración del modelo Tramo 3S
            config_chicamocha: Configuración del modelo Chicamocha
            parametros_vargas: Parámetros a calibrar para Canal Vargas
            parametros_tramo3s: Parámetros a calibrar para Tramo 3S
            parametros_chicamocha: Parámetros a calibrar para Chicamocha
            ... (resto de parámetros del GA)
        """
        # Configuraciones de modelos
        self.config_vargas = config_vargas
        self.config_tramo3s = config_tramo3s
        self.config_chicamocha = config_chicamocha

        # Parámetros a calibrar
        self.parametros_vargas = parametros_vargas
        self.parametros_tramo3s = parametros_tramo3s
        self.parametros_chicamocha = parametros_chicamocha

        # Parámetros del GA
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents_mating = num_parents_mating
        self.parent_selection_type = parent_selection_type
        self.k_tournament = k_tournament
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_by_replacement = mutation_by_replacement
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val
        self.keep_elitism = keep_elitism
        self.keep_parents = keep_parents
        self.stop_criteria = stop_criteria
        self.allow_duplicate_genes = allow_duplicate_genes
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.usar_paralelo = usar_paralelo

        # Calibradores individuales
        self.calibrador_vargas = None
        self.calibrador_tramo3s = None
        self.calibrador_chicamocha = None

        # Resultados
        self.resultado_vargas = None
        self.resultado_tramo3s = None
        self.resultado_chicamocha = None

    def _crear_calibrador(self, config: Dict, parametros: Dict) -> Calibracion:
        """Crea una instancia de Calibracion con la configuración dada."""
        return Calibracion(
            filepath=config['filepath'],
            header_dict=config['header_dict'],
            parametros=parametros,
            q_cabecera=config['q_cabecera'],
            # Parámetros del GA
            num_generations=self.num_generations,
            population_size=self.population_size,
            num_parents_mating=self.num_parents_mating,
            parent_selection_type=self.parent_selection_type,
            k_tournament=self.k_tournament,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_probability,
            mutation_percent_genes=self.mutation_percent_genes,
            mutation_by_replacement=self.mutation_by_replacement,
            random_mutation_min_val=self.random_mutation_min_val,
            random_mutation_max_val=self.random_mutation_max_val,
            keep_elitism=self.keep_elitism,
            keep_parents=self.keep_parents,
            stop_criteria=self.stop_criteria,
            allow_duplicate_genes=self.allow_duplicate_genes,
            random_seed=self.random_seed,
            num_workers=self.num_workers,
            usar_paralelo=self.usar_paralelo
        )

    def ejecutar(self, generar_graficas_individuales: bool = True) -> Dict[str, Any]:
        """
        Ejecuta la calibración secuencial de los tres modelos.

        Args:
            generar_graficas_individuales: Si generar gráficas para cada modelo

        Returns:
            Dict con los resultados de cada modelo
        """
        print('\n' + '=' * 80)
        print('PIPELINE DE CALIBRACIÓN SECUENCIAL')
        print('=' * 80)
        print('\nORDEN DE CALIBRACIÓN:')
        print('  1. Canal Vargas')
        print('  2. Tramo 3S')
        print('  3. Chicamocha')
        print('=' * 80 + '\n')

        resultados = {}

        # 1. Calibrar Canal Vargas
        print('\n' + '█' * 80)
        print('█' + ' ' * 78 + '█')
        print('█' + ' ' * 25 + 'CALIBRANDO: CANAL VARGAS' + ' ' * 29 + '█')
        print('█' + ' ' * 78 + '█')
        print('█' * 80 + '\n')

        self.calibrador_vargas = self._crear_calibrador(
            self.config_vargas,
            self.parametros_vargas
        )
        self.resultado_vargas = self.calibrador_vargas.ejecutar(
            generar_graficas=generar_graficas_individuales
        )
        resultados['vargas'] = self.resultado_vargas

        # 2. Calibrar Tramo 3S
        print('\n' + '█' * 80)
        print('█' + ' ' * 78 + '█')
        print('█' + ' ' * 26 + 'CALIBRANDO: TRAMO 3S' + ' ' * 32 + '█')
        print('█' + ' ' * 78 + '█')
        print('█' * 80 + '\n')

        self.calibrador_tramo3s = self._crear_calibrador(
            self.config_tramo3s,
            self.parametros_tramo3s
        )
        self.resultado_tramo3s = self.calibrador_tramo3s.ejecutar(
            generar_graficas=generar_graficas_individuales
        )
        resultados['tramo3s'] = self.resultado_tramo3s

        # 3. Calibrar Chicamocha
        print('\n' + '█' * 80)
        print('█' + ' ' * 78 + '█')
        print('█' + ' ' * 24 + 'CALIBRANDO: CHICAMOCHA' + ' ' * 32 + '█')
        print('█' + ' ' * 78 + '█')
        print('█' * 80 + '\n')

        self.calibrador_chicamocha = self._crear_calibrador(
            self.config_chicamocha,
            self.parametros_chicamocha
        )
        self.resultado_chicamocha = self.calibrador_chicamocha.ejecutar(
            generar_graficas=generar_graficas_individuales
        )
        resultados['chicamocha'] = self.resultado_chicamocha

        # Resumen final
        self._imprimir_resumen_final(resultados)

        return resultados

    def _imprimir_resumen_final(self, resultados: Dict):
        """Imprime un resumen final de todos los modelos."""
        print('\n' + '=' * 80)
        print('RESUMEN FINAL DE CALIBRACIÓN')
        print('=' * 80 + '\n')

        tabla = []
        for nombre, resultado in resultados.items():
            if resultado is not None:
                solucion, kge = resultado
                tabla.append([nombre.upper(), f'{kge:.6f}'])
            else:
                tabla.append([nombre.upper(), 'ERROR/INTERRUMPIDO'])

        # Imprimir tabla
        print(f'{"MODELO":<20} {"KGE FINAL":<15}')
        print('-' * 80)
        for fila in tabla:
            print(f'{fila[0]:<20} {fila[1]:<15}')

        print('=' * 80 + '\n')

    # ========================================================================
    # MÉTODOS DE VISUALIZACIÓN Y EXPORTACIÓN
    # ========================================================================

    def plotear_resultados_completos(self, modelo: str = 'all'):
        """
        Genera todas las gráficas para un modelo o todos.

        Args:
            modelo: 'vargas', 'tramo3s', 'chicamocha', o 'all'
        """
        modelos_a_plotear = []

        if modelo == 'all':
            if self.calibrador_vargas:
                modelos_a_plotear.append(('vargas', self.calibrador_vargas))
            if self.calibrador_tramo3s:
                modelos_a_plotear.append(('tramo3s', self.calibrador_tramo3s))
            if self.calibrador_chicamocha:
                modelos_a_plotear.append(('chicamocha', self.calibrador_chicamocha))
        else:
            calibrador = getattr(self, f'calibrador_{modelo}', None)
            if calibrador:
                modelos_a_plotear.append((modelo, calibrador))

        for nombre, calibrador in modelos_a_plotear:
            print(f'\n{"=" * 80}')
            print(f'GENERANDO GRÁFICAS PARA: {nombre.upper()}')
            print("=" * 80)

            # Gráfica completa (PNG)
            calibrador.plotear_evolucion_fitness(
                filename=f'evolucion_fitness_{nombre}.png',
                dpi=300,
                formato='png'
            )

            # Gráfica completa (PDF)
            calibrador.plotear_evolucion_fitness(
                filename=f'evolucion_fitness_{nombre}.pdf',
                dpi=300,
                formato='pdf'
            )

            # Gráfica simple
            calibrador.plotear_fitness_simple(
                filename=f'fitness_simple_{nombre}.png',
                dpi=300
            )

            # Exportar CSV
            calibrador.exportar_historial_csv(
                filename=f'historial_calibracion_{nombre}.csv'
            )

            # Exportar configuración
            calibrador.exportar_configuracion(
                filename=f'config_calibracion_{nombre}.txt'
            )

            print(f'✓ Gráficas y archivos generados para {nombre}')

    def generar_reporte_consolidado(self, filename: str = 'reporte_calibracion_consolidado.txt'):
        """
        Genera un reporte consolidado con todos los modelos calibrados.
        """
        # Buscar un directorio base común
        base_path = Path(self.config_vargas['filepath']).parent
        output_path = base_path / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('REPORTE CONSOLIDADO DE CALIBRACIÓN - PIPELINE\n')
            f.write('=' * 80 + '\n\n')

            # Información general
            f.write('CONFIGURACIÓN GENERAL DEL ALGORITMO GENÉTICO:\n')
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
            if self.random_seed is not None:
                f.write(f'Semilla aleatoria: {self.random_seed}\n')
            f.write(f'Procesamiento paralelo: {"Sí" if self.usar_paralelo else "No"}\n')
            if self.usar_paralelo:
                f.write(f'Número de workers: {self.num_workers}\n')
            f.write('\n\n')

            # Resultados por modelo
            modelos = [
                ('CANAL VARGAS', self.calibrador_vargas, self.resultado_vargas),
                ('TRAMO 3S', self.calibrador_tramo3s, self.resultado_tramo3s),
                ('CHICAMOCHA', self.calibrador_chicamocha, self.resultado_chicamocha)
            ]

            for nombre, calibrador, resultado in modelos:
                if calibrador is None or resultado is None:
                    continue

                f.write('=' * 80 + '\n')
                f.write(f'RESULTADOS: {nombre}\n')
                f.write('=' * 80 + '\n\n')

                solucion, kge_final = resultado

                # Estadísticas generales
                f.write('ESTADÍSTICAS GENERALES:\n')
                f.write('-' * 80 + '\n')
                f.write(f'Mejor KGE alcanzado: {kge_final:.6f}\n')
                f.write(f'Total de evaluaciones: {calibrador.contador_evaluaciones}\n')

                historial = calibrador.get_historial()
                if historial:
                    f.write(f'Generaciones completadas: {len(historial)}\n')
                    kge_inicial = historial[0]['mejor_global']
                    mejora_total = kge_final - kge_inicial
                    f.write(f'KGE inicial: {kge_inicial:.6f}\n')
                    f.write(f'Mejora total: {mejora_total:.6f}\n')

                    # Encontrar generación óptima
                    gen_optima = max(historial, key=lambda x: x['mejor_global'])
                    f.write(f'Generación óptima: {gen_optima["generacion"]}\n')

                # Parámetros óptimos
                params = calibrador.get_parametros_calibrados()
                if params:
                    f.write('\nPARÁMETROS ÓPTIMOS:\n')
                    f.write('-' * 80 + '\n')
                    for param_name, valores in params.items():
                        # Filtrar valores None
                        valores_validos = [v for v in valores if v is not None]

                        if not valores_validos:
                            # Si todos son None, saltar este parámetro
                            f.write(f'{param_name:8s}: No calibrado\n')
                            continue

                        # Verificar si es global (todos los valores no-None son iguales)
                        if len(set(valores_validos)) == 1:
                            f.write(f'{param_name:8s} (global):  {valores_validos[0]:.6f}\n')
                        else:
                            f.write(f'{param_name:8s} (por tramo):\n')
                            for i, val in enumerate(valores):
                                if val is not None:
                                    f.write(f'  Reach {i + 1}: {val:.6f}\n')
                                else:
                                    f.write(f'  Reach {i + 1}: No calibrado\n')

                # Evolución del fitness (últimas 10 generaciones)
                if historial and len(historial) > 0:
                    f.write('\nEVOLUCIÓN DEL FITNESS (últimas 10 generaciones):\n')
                    f.write('-' * 80 + '\n')
                    f.write(f'{"Gen":>5} | {"Mejor Gen":>12} | {"Mejor Global":>15} | '
                            f'{"Promedio":>12} | {"Std":>10}\n')
                    f.write('-' * 80 + '\n')
                    ultimas = historial[-10:] if len(historial) > 10 else historial
                    for h in ultimas:
                        f.write(f'{h["generacion"]:5d} | {h["mejor_fitness"]:12.6f} | '
                                f'{h["mejor_global"]:15.6f} | {h["promedio"]:12.6f} | '
                                f'{h["std"]:10.6f}\n')

                f.write('\n\n')

            # Tabla comparativa final
            f.write('=' * 80 + '\n')
            f.write('TABLA COMPARATIVA FINAL\n')
            f.write('=' * 80 + '\n\n')
            f.write(f'{"MODELO":<20} | {"KGE FINAL":<15} | {"EVALUACIONES":<15} | {"GENERACIONES":<15}\n')
            f.write('-' * 80 + '\n')

            for nombre, calibrador, resultado in modelos:
                if calibrador and resultado:
                    _, kge = resultado
                    historial = calibrador.get_historial()
                    n_gen = len(historial) if historial else 0
                    f.write(f'{nombre:<20} | {kge:<15.6f} | '
                            f'{calibrador.contador_evaluaciones:<15} | {n_gen:<15}\n')

        print(f'\n✓ Reporte consolidado generado: {output_path}')

    def plotear_comparacion_modelos(
            self,
            filename: str = 'comparacion_modelos.png',
            dpi: int = 300,
            formato: str = 'png',
            mostrar: bool = False
    ):
        """
        Genera una gráfica comparando la evolución de los 3 modelos.

        Args:
            filename: Nombre del archivo de salida
            dpi: Resolución de la imagen
            formato: Formato de salida ('png', 'pdf', 'svg')
            mostrar: Si mostrar la gráfica además de guardarla
        """
        # Verificar que hay datos disponibles
        modelos_data = []
        if self.calibrador_vargas and self.calibrador_vargas.historial_generaciones:
            modelos_data.append(('Canal Vargas', self.calibrador_vargas, '#2E86AB'))
        if self.calibrador_tramo3s and self.calibrador_tramo3s.historial_generaciones:
            modelos_data.append(('Tramo 3S', self.calibrador_tramo3s, '#A23B72'))
        if self.calibrador_chicamocha and self.calibrador_chicamocha.historial_generaciones:
            modelos_data.append(('Chicamocha', self.calibrador_chicamocha, '#F18F01'))

        if not modelos_data:
            print("No hay datos de calibración disponibles para comparar.")
            return

        # Crear figura
        fig, axes = plt.subplots(1, len(modelos_data), figsize=(6 * len(modelos_data), 5))
        if len(modelos_data) == 1:
            axes = [axes]

        fig.suptitle('Comparación de Calibración entre Modelos',
                     fontsize=16, fontweight='bold', y=0.98)

        for idx, (nombre, calibrador, color) in enumerate(modelos_data):
            historial = calibrador.get_historial()
            df = pd.DataFrame(historial)
            ax = axes[idx]

            # Evolución del mejor fitness
            ax.plot(df['generacion'], df['mejor_global'], 'o-',
                    linewidth=2.5, markersize=5, color=color,
                    label='Mejor global', alpha=0.9)

            # Banda de confianza
            ax.fill_between(df['generacion'],
                            df['promedio'] - df['std'],
                            df['promedio'] + df['std'],
                            alpha=0.2, color=color)

            ax.plot(df['generacion'], df['promedio'], '--',
                    linewidth=1.5, color=color, alpha=0.6,
                    label='Promedio ± 1σ')

            ax.set_xlabel('Generación', fontweight='bold', fontsize=11)
            ax.set_ylabel('KGE (Fitness)', fontweight='bold', fontsize=11)
            ax.set_title(nombre, fontweight='bold', pad=10, fontsize=12)
            ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(left=-1)

            # Mejor valor
            mejor_kge = df['mejor_global'].max()
            gen_mejor = df.loc[df['mejor_global'].idxmax(), 'generacion']
            ax.text(0.02, 0.98, f'KGE Máx: {mejor_kge:.4f}\n(Gen. {gen_mejor:.0f})',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

        plt.tight_layout()

        # Guardar
        base_path = Path(self.config_vargas['filepath']).parent
        output_path = base_path / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    format=formato, facecolor='white')

        print(f'✓ Gráfica comparativa guardada: {output_path}')

        if mostrar:
            plt.show()
        else:
            plt.close()

    def plotear_comparacion_barras(
            self,
            filename: str = 'comparacion_barras.png',
            dpi: int = 300,
            mostrar: bool = False
    ):
        """
        Genera gráfica de barras comparando métricas clave.
        """
        modelos = []
        kge_finales = []
        evaluaciones = []
        generaciones = []

        if self.resultado_vargas:
            modelos.append('Canal\nVargas')
            kge_finales.append(self.resultado_vargas[1])
            evaluaciones.append(self.calibrador_vargas.contador_evaluaciones)
            generaciones.append(len(self.calibrador_vargas.historial_generaciones))

        if self.resultado_tramo3s:
            modelos.append('Tramo\n3S')
            kge_finales.append(self.resultado_tramo3s[1])
            evaluaciones.append(self.calibrador_tramo3s.contador_evaluaciones)
            generaciones.append(len(self.calibrador_tramo3s.historial_generaciones))

        if self.resultado_chicamocha:
            modelos.append('Chicamocha')
            kge_finales.append(self.resultado_chicamocha[1])
            evaluaciones.append(self.calibrador_chicamocha.contador_evaluaciones)
            generaciones.append(len(self.calibrador_chicamocha.historial_generaciones))

        if not modelos:
            print("No hay resultados para comparar.")
            return

        # Crear figura
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Comparación de Métricas entre Modelos',
                     fontsize=14, fontweight='bold')

        colores = ['#2E86AB', '#A23B72', '#F18F01'][:len(modelos)]

        # Gráfica 1: KGE Final
        bars1 = axes[0].bar(modelos, kge_finales, color=colores, alpha=0.7,
                            edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('KGE Final', fontweight='bold', fontsize=11)
        axes[0].set_title('Calidad de Calibración', fontweight='bold')
        axes[0].set_ylim([0, max(kge_finales) * 1.2])
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

        # Añadir valores sobre las barras
        for bar, val in zip(bars1, kge_finales):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{val:.4f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Gráfica 2: Evaluaciones
        bars2 = axes[1].bar(modelos, evaluaciones, color=colores, alpha=0.7,
                            edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Total de Evaluaciones', fontweight='bold', fontsize=11)
        axes[1].set_title('Esfuerzo Computacional', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

        for bar, val in zip(bars2, evaluaciones):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{val:,}',
                         ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Gráfica 3: Generaciones
        bars3 = axes[2].bar(modelos, generaciones, color=colores, alpha=0.7,
                            edgecolor='black', linewidth=1.5)
        axes[2].set_ylabel('Generaciones Completadas', fontweight='bold', fontsize=11)
        axes[2].set_title('Convergencia', fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')

        for bar, val in zip(bars3, generaciones):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{val}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()

        # Guardar
        base_path = Path(self.config_vargas['filepath']).parent
        output_path = base_path / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

        print(f'✓ Gráfica de barras guardada: {output_path}')

        if mostrar:
            plt.show()
        else:
            plt.close()

    def exportar_todo(self):
        """
        Genera todos los archivos y gráficas de una vez.
        """
        print('\n' + '=' * 80)
        print('GENERANDO TODOS LOS OUTPUTS')
        print('=' * 80 + '\n')

        # Gráficas individuales para cada modelo
        print('\n--- Generando gráficas individuales ---')
        self.plotear_resultados_completos(modelo='all')

        # Reporte consolidado
        print('\n--- Generando reporte consolidado ---')
        self.generar_reporte_consolidado()

        # Gráficas comparativas
        print('\n--- Generando gráficas comparativas ---')
        self.plotear_comparacion_modelos(formato='png')
        self.plotear_comparacion_modelos(formato='pdf', filename='comparacion_modelos.pdf')
        self.plotear_comparacion_barras()

        print('\n' + '=' * 80)
        print('✓ TODOS LOS ARCHIVOS Y GRÁFICAS GENERADOS EXITOSAMENTE')
        print('=' * 80 + '\n')

        self._listar_archivos_generados()

    def _listar_archivos_generados(self):
        """Lista todos los archivos generados."""
        print('\nARCHIVOS GENERADOS:')
        print('-' * 80)

        archivos = [
            'INDIVIDUALES POR MODELO:',
            '  • evolucion_fitness_<modelo>.png (x3)',
            '  • evolucion_fitness_<modelo>.pdf (x3)',
            '  • fitness_simple_<modelo>.png (x3)',
            '  • historial_calibracion_<modelo>.csv (x3)',
            '  • parametros_calibrados.txt (x3)',
            '  • config_calibracion_<modelo>.txt (x3)',
            '',
            'ARCHIVOS CONSOLIDADOS:',
            '  • reporte_calibracion_consolidado.txt',
            '  • comparacion_modelos.png',
            '  • comparacion_modelos.pdf',
            '  • comparacion_barras.png',
        ]

        for item in archivos:
            print(item)

        print('-' * 80)