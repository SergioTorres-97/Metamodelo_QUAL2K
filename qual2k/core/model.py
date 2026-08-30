import pandas as pd
import os
import shutil
from typing import Dict, Any
from .config import Q2KConfig
from qual2k.processing.data_processor import Q2KDataProcessor
from qual2k.processing.file_writer import Q2KFileWriter
from qual2k.core.simulator import ejecutar_simulacion
from qual2k.analysis.results_analyzer import Q2KResultsAnalyzer
from qual2k.analysis.plotter import Q2KPlotter


class Q2KModel:
    """
    Clase principal que orquesta todo el proceso de modelación con QUAL2K.

    Uso básico:
        model = Q2KModel(filepath, header_dict)
        model.cargar_plantillas()
        model.configurar_modelo()
        model.generar_archivo_q2k()
        model.ejecutar_simulacion()
        resultados = model.analizar_resultados()
    """

    def __init__(self, filepath: str, header_dict: Dict[str, Any]):
        """
        Inicializa el modelo Q2K.

        Args:
            filepath: Ruta del directorio de trabajo
            header_dict: Diccionario con configuración del header
        """
        self.filepath = filepath
        self.config = Q2KConfig(header_dict)
        self.data_processor = Q2KDataProcessor()
        self.file_writer = Q2KFileWriter()
        self.results_analyzer = Q2KResultsAnalyzer()
        self.plotter = Q2KPlotter()

        # DataFrames de entrada
        self.data_reaches = None
        self.data_sources = None
        self.data_wq = None

        # Diccionarios de datos procesados
        self.q2k_data = {}

        # Resultados
        self.wq_data_model = None
        self.data_exp = None

    def cargar_plantillas(self, archivo_excel: str = 'PlantillaBaseQ2K.xlsx'):
        """
        Carga las plantillas desde el archivo Excel.

        Args:
            archivo_excel: Nombre del archivo Excel con las plantillas
        """
        ruta_completa = os.path.join(self.filepath, archivo_excel)
        print("=" * 70)
        print('CARGANDO PLANTILLAS')
        print("=" * 70)

        self.data_reaches = pd.read_excel(ruta_completa, sheet_name='REACHES')
        self.data_sources = pd.read_excel(ruta_completa, sheet_name='SOURCES')
        self.data_wq = pd.read_excel(ruta_completa, sheet_name='WQ_DATA')

        print(f'✅ Plantillas cargadas satisfactoriamente')

    def configurar_modelo(self,
                          numelem_default: int = 10,
                          q_cabecera: float = 1.06574E-06,
                          estacion_cabecera: str = 'CABECERA',
                          reach_rates_custom: Dict = None):
        """
        Configura todos los componentes del modelo.

        Args:
            numelem_default: Número de elementos por tramo
            q_cabecera: Caudal de cabecera
            estacion_cabecera: Nombre de la estación de cabecera
            reach_rates_custom: Diccionario personalizado para reach_rates (opcional)
        """
        print("=" * 70)
        print('CONFIGURANDO MODELO')
        print("=" * 70)

        # Procesar datos de tramos
        reach_dict = self.data_processor.crear_reach_dict(
            self.data_reaches,
            numelem_default,
            q_cabecera
        )

        # Procesar fuentes puntuales
        point_sources_dict = self.data_processor.crear_point_sources_dict(
            self.data_sources
        )

        # Procesar cabeceras
        headwaters_dict = self.data_processor.crear_headwaters_dict(
            self.data_reaches,
            self.data_wq,
            estacion_cabecera
        )

        # Procesar datos meteorológicos
        met_data_dict = self.data_processor.crear_met_data_dict(
            self.data_reaches
        )

        # Procesar datos de temperatura observados
        temperature_data_dict = self.data_processor.crear_temperature_data_dict(
            self.data_wq
        )

        # Procesar datos de calidad de agua observados
        wqdata = self.data_processor.crear_wqdata_dict(self.data_wq)

        # Configurar reach_rates
        if reach_rates_custom:
            reach_rates_dict = reach_rates_custom
        else:
            reach_rates_dict = self.config.generar_reach_rates_default(
                reach_dict['nr']
            )

        # Ensamblar todos los datos
        self.q2k_data = {
            "header": self.config.header_dict,
            "reach_data": reach_dict,
            "light_data": self.config.light_dict,
            "point_sources": point_sources_dict,
            "diffuse_sources": self.config.diffuse_sources_dict,
            "rates_general": self.config.rates_dict,
            "reach_rates": reach_rates_dict,
            "boundary_data": self.config.boundary_dict,
            "headwaters": headwaters_dict,
            "meteorological": met_data_dict,
            "temperature_data": temperature_data_dict,
            "hydraulics_data": self.config.hydraulics_data_dict,
            "wq_data": wqdata,
            "diel": self.config.diel_dict,
        }

        print(f'✅ Modelo configurado satisfactoriamente')

    def generar_archivo_q2k(self):
        """Genera el archivo .q2k y el mensaje.DAT"""
        print("=" * 70)
        print('GENERACIÓN DEL ARCHIVO .q2k')
        print("=" * 70)

        # Generar archivo .q2k
        ruta_q2k = os.path.join(
            self.config.header_dict['filedir'],
            f"{self.config.header_dict['filename']}.q2k"
        )
        self.file_writer.create_q2k_file(ruta_q2k, self.q2k_data)

        # Generar message.DAT
        self.file_writer.create_message(self.config.header_dict)

        print(f'✅ Archivo q2k generado satisfactoriamente')

    def ejecutar_simulacion(self):
        """Ejecuta la simulación FORTRAN"""
        print("=" * 70)
        print('EJECUTANDO SIMULACIÓN FORTRAN')
        print("=" * 70)

        exe_name = 'q2kfortran2_12.exe'
        exe_path = os.path.join(self.config.header_dict['filedir'], exe_name)

        # Copiar el ejecutable desde bin/ si no está ya en filedir
        if not os.path.exists(exe_path):
            bin_exe = os.path.join(
                os.path.dirname(__file__), '..', '..', 'bin', exe_name
            )
            bin_exe = os.path.abspath(bin_exe)
            if not os.path.exists(bin_exe):
                raise FileNotFoundError(
                    f"No se encontró el ejecutable FORTRAN en: {bin_exe}\n"
                    f"Asegúrate de que 'bin/{exe_name}' existe en la raíz del proyecto."
                )
            shutil.copy2(bin_exe, exe_path)

        ejecutar_simulacion(exe_path)

        print(f'✅ Simulación ejecutada satisfactoriamente')

    def analizar_resultados(self, generar_graficas: bool = True,
                            generar_comparacion: bool = True):
        """
        Analiza los resultados de la simulación.

        Args:
            generar_graficas: Si se deben generar las gráficas de perfiles
            generar_comparacion: Si se deben generar las gráficas Simulado vs Observado

        Returns:
            DataFrame con resultados experimentales (modelados + observados)
        """
        print("=" * 70)
        print('EXTRACCIÓN Y ANÁLISIS DE RESULTADOS')
        print("=" * 70)

        # Leer archivo .out
        filepath_out = os.path.join(
            self.config.header_dict['filedir'],
            f"{self.config.header_dict['filename']}.out"
        )

        # Crear carpeta de resultados
        resultados_dir = os.path.join(
            self.config.header_dict['filedir'],
            'resultados'
        )
        os.makedirs(resultados_dir, exist_ok=True)

        # Procesar resultados del modelo
        self.wq_data_model = self.results_analyzer.procesar_out_file(filepath_out)

        # Preparar datos observados
        data_obs = self.results_analyzer.preparar_datos_observados(self.data_wq)

        # Combinar modelados y observados
        self.data_exp = self.results_analyzer.combinar_modelados_observados(
            self.wq_data_model,
            data_obs
        )

        # Guardar resultados
        self.data_exp.to_csv(
            os.path.join(resultados_dir, f"{self.config.header_dict['filename']}.csv"),
            index=False
        )

        # Generar gráficas si se solicita
        if generar_graficas:
            self._generar_graficas(resultados_dir, generar_comparacion)

        print(f'✅ Resultados analizados satisfactoriamente')

        return self.data_exp

    def _tiene_datos_observados(self) -> bool:
        """Verifica si WQ_DATA tiene filas de observación más allá de CABECERA."""
        if self.data_wq is None or len(self.data_wq) <= 1:
            return False
        return True

    def _generar_graficas(self, resultados_dir: str, generar_comparacion: bool = True):
        """Genera todas las gráficas de resultados"""
        # Gráficas de parámetros modelados
        self.plotter.plot_all_params(self.wq_data_model, resultados_dir)

        # Gráficas comparativas (modelado vs observado)
        if not generar_comparacion:
            return
        if not self._tiene_datos_observados():
            print("ℹ️  WQ_DATA solo contiene CABECERA — se omiten gráficas Simulado vs Observado")
            return

        resultados_dir_cal_obs = os.path.join(resultados_dir, 'comparacion')
        os.makedirs(resultados_dir_cal_obs, exist_ok=True)
        self.plotter.plot_all_params_cal_obs(self.data_exp, resultados_dir_cal_obs)

    def calcular_metricas_calibracion(self, pesos: Dict[str, float] = None):
        """
        Calcula las métricas de calibración (KGE).

        Args:
            pesos: Diccionario con pesos para cada variable

        Returns:
            Tuple (resultados_por_variable, kge_global)
        """
        print("=" * 70)
        print('CÁLCULO DE MÉTRICAS DE CALIBRACIÓN')
        print("=" * 70)

        # Pesos por defecto
        if pesos is None:
            pesos = {
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
            }

        # Pares de columnas
        pares = [
            ("water_temp_c", "water_temp_c_obs"),
            ("conductivity","conductivity_obs"),
            ("nitrate","nitrate_obs"),
            ("pathogen", "pathogen_obs"),
            ("pH","pH_obs"),
            ("total_suspended_solids", "total_suspended_solids_obs"),
            ("dissolved_oxygen", "dissolved_oxygen_obs"),
            ("carbonaceous_bod_fast", "carbonaceous_bod_fast_obs"),
            ("total_kjeldahl_nitrogen", "total_kjeldahl_nitrogen_obs"),
            ("ammonium", "ammonium_obs"),
            ("total_phosphorus", "total_phosphorus_obs"),
        ]

        resultados, kge_global = self.results_analyzer.calcular_kge_global(
            self.data_exp,
            pares,
            pesos
        )

        print("KGE por variable:", resultados)
        print("KGE global ponderado:", kge_global)
        print(f'✅ Métricas calculadas satisfactoriamente')

        return resultados, kge_global

    def ejecutar_flujo_completo(self,
                                archivo_excel: str = 'PlantillaBaseQ2K.xlsx',
                                **kwargs):
        """
        Ejecuta el flujo completo de simulación.

        Args:
            archivo_excel: Nombre del archivo Excel con plantillas
            **kwargs: Argumentos adicionales para configurar_modelo()
        """
        self.cargar_plantillas(archivo_excel)
        self.configurar_modelo(**kwargs)
        self.generar_archivo_q2k()
        self.ejecutar_simulacion()
        self.analizar_resultados()
        self.calcular_metricas_calibracion()

        print("=" * 70)
        print('✅ FLUJO COMPLETO EJECUTADO SATISFACTORIAMENTE')
        print("=" * 70)