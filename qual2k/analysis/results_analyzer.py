import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Tuple
from qual2k.analysis import metricas

class Q2KResultsAnalyzer:
    """
    Analiza los resultados de las simulaciones QUAL2K.
    """

    @staticmethod
    def leer_secciones(ruta_out: str) -> Dict[str, str]:
        """
        Lee el archivo .out y separa secciones por título.

        Args:
            ruta_out: Ruta del archivo .out

        Returns:
            Diccionario con secciones {titulo: contenido}
        """
        secciones = {}
        seccion_actual = None
        contenido = []

        patron = re.compile(r'^\s*\*\*(.*?)\*\*\s*$')

        with open(ruta_out, "r", encoding="utf-8") as f:
            for linea in f:
                l = linea.rstrip()
                match = patron.match(l)

                if match:
                    if seccion_actual:
                        secciones[seccion_actual] = "\n".join(contenido)
                    seccion_actual = match.group(1).strip()
                    contenido = []
                else:
                    contenido.append(l)

        if seccion_actual:
            secciones[seccion_actual] = "\n".join(contenido)

        return secciones

    @staticmethod
    def clean_text(txt: str) -> str:
        """Limpia texto del archivo .out"""
        txt = re.sub(r'\n-+\n', '\n', txt)
        txt = re.sub(r'0(?=[A-Za-z_])', '0  ', txt)
        return txt

    @staticmethod
    def is_numeric_block(cel: str) -> bool:
        """Verifica si la celda contiene varios números"""
        patron = r'(?:[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)(?:\s+[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)+'
        return re.fullmatch(patron, cel.strip()) is not None

    def parse_section(self, secciones: Dict, name: str) -> pd.DataFrame:
        """
        Parsea una sección específica.

        Args:
            secciones: Diccionario de secciones
            name: Nombre de la sección

        Returns:
            DataFrame con datos parseados
        """
        txt = self.clean_text(secciones[name])
        filas = []

        for l in txt.split("\n"):
            l = l.strip()
            if not l:
                continue

            parts = re.split(r"\s{2,}", l)

            row = []
            for c in parts:
                c = c.strip()
                if not c:
                    continue

                if self.is_numeric_block(c):
                    row.extend(c.split())
                else:
                    row.append(c)

            filas.append(row)

        columnas = filas[0]
        datos = filas[2:]

        n_cols = len(columnas)
        datos_norm = []
        for row in datos:
            if len(row) > n_cols:
                datos_norm.append(row[:n_cols])
            elif len(row) < n_cols:
                datos_norm.append(row + [None] * (n_cols - len(row)))
            else:
                datos_norm.append(row)

        df = pd.DataFrame(datos_norm, columns=columnas)
        df = df.apply(pd.to_numeric, errors="ignore").reset_index(drop=True)
        return df

    def procesar_out_file(self, ruta_out: str) -> pd.DataFrame:
        """
        Procesa un archivo .out completo de QUAL2K.

        Args:
            ruta_out: Ruta del archivo .out

        Returns:
            DataFrame consolidado con resultados
        """
        secciones = self.leer_secciones(ruta_out)

        # Mapeos de columnas
        hyd_map = {
            'Trib': 'tributary',
            'Reach': 'river_reach',
            'Downstream': 'Distancia Longitudinal (km)',
            'Hydraulics': 'flow',
            "E'": 'energy_loss',
            'H': 'hydraulic_head',
            'Btop': 'channel_top_width',
            'Ac': 'cross_section_area',
            'U': 'flow_velocity',
            'trav time': 'travel_time',
            'slope': 'channel_slope',
            'Reaeration': 'reaeration_rate',
            'Reaeration formulas': 'reaeration_method',
            'drop (m)': 'elevation_drop_m'
        }

        temps_map = {
            'Reach': 'river_reach',
            'Distance': 'Distancia Longitudinal (km)',
            'Temp(C)': 'water_temp_c'
        }

        wq_map = {
            'Trib': 'tributary',
            'Reach': 'river_reach',
            'x': 'Distancia Longitudinal (km)',
            'cond': 'conductivity',
            'ISS': 'inorganic_suspended_solids',
            'DO': 'dissolved_oxygen',
            'CBODs': 'carbonaceous_bod_slow',
            'CBODf': 'carbonaceous_bod_fast',
            'No': 'nitrite',
            'NH4': 'ammonium',
            'NO3': 'nitrate',
            'PO': 'organic_phosphorus',
            'InorgP': 'inorganic_phosphorus',
            'Phyto': 'phytoplankton',
            'INp': 'inorganic_particulate_p',
            'IPp': 'organic_particulate_p',
            'Detritus': 'detritus',
            'Pathogen': 'pathogen',
            'Alk': 'alkalinity',
            'Const i': 'const_i',
            'Const ii': 'const_ii',
            'Const iii': 'const_iii',
            'pH': 'pH',
            'Bot Alg': 'benthic_algae',
            'QNb': 'nitrogen_flow_rate',
            'QPb': 'phosphorus_flow_rate',
            'TOC': 'total_organic_carbon',
            'TN': 'total_nitrogen',
            'TP': 'total_phosphorus',
            'TKN': 'total_kjeldahl_nitrogen',
            'TSS': 'total_suspended_solids',
            'CBODu': 'ultimate_cbod',
            'NH3': 'ammonia',
            'DO sat': 'do_saturation',
            'pH sat': 'ph_saturation'
        }

        # Procesar hidráulica
        hyd = self.parse_section(secciones, 'Hydraulics Summary')
        hyd = hyd.rename(columns=hyd_map)
        hyd = hyd[['Distancia Longitudinal (km)', 'flow', 'hydraulic_head',
                   'channel_top_width', 'cross_section_area',
                   'flow_velocity', 'travel_time']]
        hyd = hyd.sort_values('Distancia Longitudinal (km)')

        # Procesar temperatura
        temps = self.parse_section(secciones, 'Temperature Summary')
        temps = temps.rename(columns=temps_map)
        temps = temps[['Distancia Longitudinal (km)', 'water_temp_c']]
        temps = temps.sort_values('Distancia Longitudinal (km)')
        temps = temps.iloc[:, 0:2]

        # Procesar calidad de agua
        wq = self.parse_section(secciones, 'Water Quality Summary')
        wq = wq.rename(columns=wq_map)
        wq = wq[['Distancia Longitudinal (km)', 'conductivity',
                 'inorganic_suspended_solids', 'dissolved_oxygen',
                 'carbonaceous_bod_slow', 'carbonaceous_bod_fast', 'nitrite',
                 'ammonium', 'nitrate', 'organic_phosphorus',
                 'inorganic_phosphorus', 'detritus', 'pathogen', 'alkalinity',
                 'const_i', 'const_ii', 'const_iii', 'pH', 'total_nitrogen',
                 'total_phosphorus', 'total_kjeldahl_nitrogen',
                 'total_suspended_solids', 'ultimate_cbod', 'ammonia']]
        wq = wq.sort_values('Distancia Longitudinal (km)')

        # Merge final
        merged = pd.merge_asof(wq, temps, on='Distancia Longitudinal (km)', direction='nearest')
        merged = merged.drop_duplicates(subset='Distancia Longitudinal (km)', keep='first')
        merged = pd.merge_asof(merged, hyd, on='Distancia Longitudinal (km)', direction='nearest')
        merged = merged.drop_duplicates(subset='Distancia Longitudinal (km)', keep='first')

        return merged.sort_values('Distancia Longitudinal (km)', ascending=False).reset_index(drop=True)

    @staticmethod
    def preparar_datos_observados(dataWQ: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los datos observados desde el DataFrame de calidad de agua.

        Args:
            dataWQ: DataFrame con datos de calidad de agua observados

        Returns:
            DataFrame con datos observados formateados
        """
        mapeoObservados = {
            'X_QUAL2K': 'Distancia Longitudinal (km)',
            'CAUDAL': 'flow',
            'TEMPERATURA': 'water_temp_c',
            'CONDUCTIVIDAD': 'conductivity',
            'NITRATOS': 'nitrate',
            'ORTOFOSFATOS': 'inorganic_phosphorus',
            'COLIFORMES_TOTALES': 'pathogen',
            'E_COLI': 'const_ii',
            'ALCALINIDAD': 'alkalinity',
            'pH': 'pH',
            'SST': 'total_suspended_solids',
            'OXIGENO_DISUELTO': 'dissolved_oxygen',
            'DBO5': 'carbonaceous_bod_fast',
            'NTK': 'total_kjeldahl_nitrogen',
            'NITROGENO_AMONIACAL': 'ammonium',
            'FOSFORO_TOTAL': 'total_phosphorus',
        }

        dataWQ = dataWQ.rename(columns=mapeoObservados)
        dataWQ = dataWQ[[
            'Distancia Longitudinal (km)',
            'flow',
            'water_temp_c',
            'conductivity',
            'nitrate',
            'inorganic_phosphorus',
            'pathogen',
            'const_ii',
            'alkalinity',
            'pH',
            'total_suspended_solids',
            'dissolved_oxygen',
            'carbonaceous_bod_fast',
            'total_kjeldahl_nitrogen',
            'ammonium',
            'total_phosphorus'
        ]]

        dataWQ.columns = [f'{dataWQ.columns[i]}_obs' for i in range(dataWQ.shape[1])]
        dataWQ = dataWQ.rename(columns={'Distancia Longitudinal (km)_obs': 'Distancia Longitudinal (km)'})

        # Conversión mg/L → µg/L para N y P (QUAL2K opera en µg/L para estos)
        cols_np = ['nitrate_obs', 'inorganic_phosphorus_obs',
                   'total_kjeldahl_nitrogen_obs', 'ammonium_obs', 'total_phosphorus_obs']
        for col in cols_np:
            if col in dataWQ.columns:
                dataWQ[col] = dataWQ[col] * 1000

        dataWQ = dataWQ.sort_values(by=['Distancia Longitudinal (km)'])

        return dataWQ

    @staticmethod
    def combinar_modelados_observados(wq_model: pd.DataFrame,
                                      data_obs: pd.DataFrame) -> pd.DataFrame:
        """
        Combina datos modelados y observados con emparejamiento 1-a-1.
        Cada observación se empareja con su valor modelado más cercano una sola vez.

        Args:
            wq_model: DataFrame con datos modelados
            data_obs: DataFrame con datos observados

        Returns:
            DataFrame combinado con todas las filas del modelo
        """
        wq_model = wq_model.sort_values('Distancia Longitudinal (km)').reset_index(drop=True)
        data_obs = data_obs.sort_values('Distancia Longitudinal (km)').reset_index(drop=True)
        # Guardar copia del modelo para no modificarlo
        modelo_temp = wq_model.copy()

        # Lista para guardar los matches
        matches = []

        # Para cada observación, encontrar su modelo más cercano
        for idx_obs, row_obs in data_obs.iterrows():
            dist_obs = row_obs['Distancia Longitudinal (km)']

            # Calcular distancias absolutas con los modelos restantes
            distances = np.abs(modelo_temp['Distancia Longitudinal (km)'] - dist_obs)

            # Encontrar el índice del más cercano
            closest_idx = distances.idxmin()

            # Guardar el match
            match_info = {
                'idx_modelo': closest_idx,
                'dist_modelo': modelo_temp.loc[closest_idx, 'Distancia Longitudinal (km)']
            }

            # Agregar todas las columnas observadas
            for col in data_obs.columns:
                if col != 'Distancia Longitudinal (km)':
                    match_info[col] = row_obs[col]

            matches.append(match_info)

            # Remover este índice para que no se vuelva a usar
            modelo_temp = modelo_temp.drop(closest_idx)

        # Crear DataFrame con los matches
        df_matches = pd.DataFrame(matches)

        # Merge con el modelo completo (left join para mantener todas las filas)
        df_merge = wq_model.merge(
            df_matches,
            left_on='Distancia Longitudinal (km)',
            right_on='dist_modelo',
            how='left'
        )

        # Limpiar columnas auxiliares
        df_merge = df_merge.drop(['idx_modelo', 'dist_modelo'], axis=1, errors='ignore')
        return df_merge

    @staticmethod
    def calcular_kge_global(dataExp: pd.DataFrame,
                            pares: List[Tuple[str, str]],
                            pesos: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Calcula KGE por variable y KGE global ponderado.

        Args:
            dataExp: DataFrame con datos experimentales
            pares: Lista de tuplas (columna_sim, columna_obs)
            pesos: Diccionario con pesos por variable

        Returns:
            Tuple con (resultados_por_variable, kge_global)
        """

        # dataExp = dataExp.dropna()
        resultados = {}
        for sim_col, obs_col in pares:
            kge_val = metricas.kge(dataExp[obs_col], dataExp[sim_col])
            resultados[sim_col] = kge_val

        global_kge = sum(resultados[var] * pesos[var] for var in resultados)

        return resultados, global_kge