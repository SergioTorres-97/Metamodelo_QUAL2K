import pandas as pd
import numpy as np
from typing import Dict, Any


class Q2KDataProcessor:
    """
    Procesa datos de las plantillas Excel y los convierte a diccionarios
    para el formato QUAL2K.
    """

    @staticmethod
    def _transformar_calidad(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformaciones estándar de unidades y fracciones de calidad de agua."""
        df = df.copy()
        df['S_INORG'] = df['SST'] * 0.15       # 15% del SST es inorgánico
        df['DETRITUS'] = df['SST'] * 0.85      # 85% del SST es detritus
        # DBO lenta desde DBO5 (sustrayendo la fracción rápida)
        df['DBO_SLOW'] = df['DBO5'] / (1 - np.exp(-0.23 * 5)) - (1.46 * df['DBO5'])
        # Conversión mg/L → µg/L para compuestos de N y P
        for col in ['NTK', 'NITROGENO_AMONIACAL', 'NITRITOS', 'NITRATOS', 'FOSFORO_TOTAL', 'ORTOFOSFATOS']:
            df[col] = df[col] * 1000
        df['N_ORG'] = df['NTK'] - df['NITROGENO_AMONIACAL']
        df['N_ORG'] = np.where(df['N_ORG'] < 0, np.nan, df['N_ORG'])
        df['NO3'] = df['NITRITOS'] + df['NITRATOS']
        df['P_ORG'] = df['FOSFORO_TOTAL'] - df['ORTOFOSFATOS']
        df['P_ORG'] = np.where(df['P_ORG'] < 0, np.nan, df['P_ORG'])
        return df

    def crear_point_sources_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Crea el diccionario de fuentes puntuales desde el DataFrame.

        Args:
            df: DataFrame con datos de fuentes puntuales

        Returns:
            Diccionario de fuentes puntuales
        """
        df = df.fillna(0)

        df['CAUDAL_CAPT'] = df.apply(
            lambda x: x['CAUDAL'] if str(x['TIPO']).lower().startswith('capt') else 0,
            axis=1
        )
        df['CAUDAL_VERT'] = df.apply(
            lambda x: x['CAUDAL'] if str(x['TIPO']).lower().startswith('vert') else 0,
            axis=1
        )

        df = self._transformar_calidad(df)

        # Renombrar columnas
        relacion_q2k_parametros = {
            'NOMBRE_VERTIMIENTO': 'PtName',
            'X_QUAL2K': 'xptt',
            'CAUDAL_VERT': 'Qptt',
            'CAUDAL_CAPT': 'Qptta',
            "TEMPERATURA": "TepttMean",
            "CONDUCTIVIDAD": "Cond",
            "S_INORG": "ISS",
            "OXIGENO_DISUELTO": "DO",
            "DBO_SLOW": "CBODs",
            "DBO5": "CBODf",
            "N_ORG": "Norg",
            "NITROGENO_AMONIACAL": "NH4",
            "NO3": "NO3",
            "P_ORG": "Porg",
            "ORTOFOSFATOS": "Inorg_P",
            "DETRITUS": "Detr",
            "COLIFORMES_TOTALES": "Pathogens",
            "ALCALINIDAD": "Alk",
            "COLIFORMES_TERMOTOLERANTES": "Constituent_i",
            "E_COLI": "Constituent_ii",
            "pH": "pH"
        }

        df = df.rename(columns=relacion_q2k_parametros)

        # Construcción del diccionario
        sources_list = []

        for i, row in df.iterrows():
            constituents = [
                {"name": "Cond", "mean": row.get("Cond", 0), "amp": 0, "maxtime": 0},
                {"name": "ISS", "mean": row.get("ISS", 0), "amp": 0, "maxtime": 0},
                {"name": "DO", "mean": row.get("DO", 0), "amp": 0, "maxtime": 0},
                {"name": "CBODs", "mean": row.get("CBODs", 0), "amp": 0, "maxtime": 0},
                {"name": "CBODf", "mean": row.get("CBODf", 0), "amp": 0, "maxtime": 0},
                {"name": "Norg", "mean": row.get("Norg", 0), "amp": 0, "maxtime": 0},
                {"name": "NH4", "mean": row.get("NH4", 0), "amp": 0, "maxtime": 0},
                {"name": "NO3", "mean": row.get("NO3", 0), "amp": 0, "maxtime": 0},
                {"name": "Porg", "mean": row.get("Porg", 0), "amp": 0, "maxtime": 0},
                {"name": "Inorg_P", "mean": row.get("Inorg_P", 0), "amp": 0, "maxtime": 0},
                {"name": "Phyto", "mean": 0, "amp": 0, "maxtime": 0},
                {"name": "IntN", "mean": 0, "amp": 0, "maxtime": 0},
                {"name": "IntP", "mean": 0, "amp": 0, "maxtime": 0},
                {"name": "Detr", "mean": row.get("Detr", 0), "amp": 0, "maxtime": 0},
                {"name": "Pathogens", "mean": row.get("Pathogens", 0), "amp": 0, "maxtime": 0},
                {"name": "Alk", "mean": row.get("Alk", 0), "amp": 0, "maxtime": 0},
                {"name": "Constituent_i", "mean": row.get("Constituent_i", 0), "amp": 0, "maxtime": 0},
                {"name": "Constituent_ii", "mean": row.get("Constituent_ii", 0), "amp": 0, "maxtime": 0},
                {"name": "Constituent_iii", "mean": 0, "amp": 0, "maxtime": 0},
                {"name": "pH", "mean": row.get("pH", 7 if pd.notna(row.get("pH")) else 0), "amp": 0, "maxtime": 0},
            ]

            source = {
                "PtName": row.get("PtName", f"Vert_{i + 1}"),
                "PtHwID": 1,
                "xptt": row.get("xptt", 0),
                "Qptta": row.get("Qptta", 0),
                "Qptt": row.get("Qptt", 0),
                "TepttMean": row.get("TepttMean", 20),
                "TepttAmp": 0,
                "TepttMaxTime": 0,
                "constituents": constituents
            }

            sources_list.append(source)

        return {
            "npt": len(sources_list),
            "sources": sources_list
        }

    def crear_reach_dict(self, df: pd.DataFrame,
                         numElem_default: int = 20,
                         Qcabecera: float = 1.06574E-06) -> Dict[str, Any]:
        """
        Crea el diccionario de tramos desde el DataFrame.

        Args:
            df: DataFrame con datos de tramos
            numElem_default: Número de elementos por tramo
            Qcabecera: Caudal de cabecera

        Returns:
            Diccionario de tramos
        """
        reach_dict = {
            "nr": len(df),
            "nHw": 1,
            "ne": int(numElem_default * len(df)),
            "reaches": []
        }

        for _, row in df.iterrows():
            tramo = {
                "rlab1": str(row["EST_ARRIBA"]),
                "rlab2": str(row["EST_ABAJO"]),
                "rname": str(row["NOMBRE_TRAMO"]),
                "xrup": float(row["X_QUAL2K_ARRIBA"]),
                "xrdn": float(row["X_QUAL2K_ABAJO"]),
                "numElm": int(numElem_default),
                "elev1": float(row["ELEV_ARRIBA"]),
                "elev2": float(row["ELEV_ABAJO"]),
                "latd": 0, "latm": 0, "lats": 0,
                "lond": 0, "lonm": 0, "lons": 0,
                "Q": float(Qcabecera),
                "BB": 0, "SS1": 0, "SS2": 0,
                "s": -99999, "nm": 0,
                "alp1": float(row["ALPHA_1"]),
                "bet1": float(row["BETA_1"]),
                "alp2": float(row["ALPHA_2"]),
                "bet2": float(row["BETA_2"]),
                "Ediff": 0, "Frsed": 1e-05, "Frsod": 1e-05,
                "SODspec": 0, "JCH4spec": 0, "JNH4spec": 0, "JSRPspec": 0,
                "weirType": "", "Hweir": 0, "Bweir": 0,
                "adam": 1.25, "bdam": 0.9, "evap": 0
            }
            reach_dict["reaches"].append(tramo)

        return reach_dict

    def crear_headwaters_dict(self, df_tramo: pd.DataFrame,
                              df_calidad: pd.DataFrame,
                              estacion: str) -> Dict[str, Any]:
        """
        Crea el diccionario de cabeceras.

        Args:
            df_tramo: DataFrame con datos de tramos
            df_calidad: DataFrame con datos de calidad de agua
            estacion: Nombre de la estación de cabecera

        Returns:
            Diccionario de cabeceras
        """
        tramo = df_tramo[df_tramo['EST_ARRIBA'] == estacion].fillna(0)
        cal = self._transformar_calidad(df_calidad[df_calidad['NOMBRE_ESTACIONES'] == estacion].fillna(0))

        # Temperatura (lista 24h)
        TeHw = [float(cal.get("TEMPERATURA", 20))] * 24

        # Concentraciones (lista de listas 24h)
        cHw = [
            [float(cal.get("CONDUCTIVIDAD", 0))] * 24,
            [float(cal.get("S_INORG", 0))] * 24,
            [float(cal.get("OXIGENO_DISUELTO", 0))] * 24,
            [float(cal.get("DBO_SLOW", 0))] * 24,
            [float(cal.get("DBO5", 0))] * 24,
            [float(cal.get("N_ORG", 0))] * 24,
            [float(cal.get("NITROGENO_AMONIACAL", 0))] * 24,
            [float(cal.get("NO3", 0))] * 24,
            [float(cal.get("P_ORG", 0))] * 24,
            [float(cal.get("ORTOFOSFATOS", 0))] * 24,
            [0.0] * 24, [0.0] * 24, [0.0] * 24,
            [float(cal.get("DETRITUS", 0))] * 24,
            [float(cal.get("COLIFORMES_TOTALES", 0))] * 24,
            [float(cal.get("ALCALINIDAD", 0))] * 24,
            [float(cal.get("COLIFORMES_TERMOTOLERANTES", 0))] * 24,
            [float(cal.get("E_COLI", 0))] * 24,
            [0.0] * 24
        ]

        # pH 24h
        pHHw = [float(cal.get("pH", 7))] * 24

        # Diccionario final de la cabecera
        headwater = {
            "begRch": 1,
            "NameHw": str(cal['NOMBRE_ESTACIONES'].values[0]),
            "QHw": float(cal.get("CAUDAL", 0)),
            "elevHw": float(tramo.get("ELEV_ARRIBA", 0)),
            "weirTypeHw": "",
            "HweirHw": 0,
            "BweirHw": 0,
            "alp1Hw": float(tramo.get("ALPHA_1", 0)),
            "bet1Hw": float(tramo.get("BETA_1", 0)),
            "alp2Hw": float(tramo.get("ALPHA_2", 0)),
            "bet2Hw": float(tramo.get("BETA_2", 0)),
            "sHw": 0, "nmHw": 0, "bbHw": 0, "ss1Hw": 0, "ss2Hw": 0,
            "ediffHw": 0, "adamHw": 1.25, "bdamHw": 0.9,
            "TeHw": TeHw,
            "cHw": cHw,
            "pHHw": pHHw
        }

        return {
            "nHw": 1,
            "headwaters": [headwater]
        }

    def crear_met_data_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Crea el diccionario de datos meteorológicos.

        Args:
            df: DataFrame con datos meteorológicos

        Returns:
            Diccionario de datos meteorológicos
        """
        shadeHH = [[float(row["SOMBRA_[-]"])] * 24 for _, row in df.iterrows()]
        TaHH = [[float(row["TEMPERATURA_[C]"])] * 24 for _, row in df.iterrows()]
        TdHH = [[float(row["TEMPERATURA_ROCIO_[C]"])] * 24 for _, row in df.iterrows()]
        UwHH = [[float(row["VELOCIDAD_DEL_VIENTO_[MS]"])] * 24 for _, row in df.iterrows()]
        ccHH = [[float(row["COBERTURA_NUBES_[-]"])] * 24 for _, row in df.iterrows()]

        return {
            "nr": len(df),
            "shadeHH": shadeHH,
            "TaHH": TaHH,
            "TdHH": TdHH,
            "UwHH": UwHH,
            "ccHH": ccHH
        }

    def crear_temperature_data_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Crea el diccionario de datos de temperatura observados.

        Args:
            df: DataFrame con datos de temperatura

        Returns:
            Diccionario de datos de temperatura
        """
        data_list = []
        for _, row in df.iterrows():
            data = {
                "tedaHwID": 1,
                "xteda": float(row.get("X_QUAL2K", 0)),
                "tedaav": float(row.get("TEMPERATURA", 0)),
                "tedamn": None,
                "tedamx": None
            }
            data_list.append(data)

        return {
            "nteda": len(data_list),
            "data": data_list
        }

    def crear_wqdata_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Crea el diccionario de datos de calidad de agua observados.

        Args:
            df: DataFrame con datos de calidad de agua

        Returns:
            Diccionario de datos de calidad de agua
        """
        df = self._transformar_calidad(df)
        df['TKN'] = df['NTK']
        df['TN'] = df['N_ORG'] + df['NITROGENO_AMONIACAL'] + df['NO3']
        df['TP'] = df['P_ORG'] + df['ORTOFOSFATOS']

        df = df.replace(np.nan, -99999)

        # Definición de campos QUAL2K
        constituyentes_base = [
            "Cond", "ISS", "DO", "CBODs", "CBODf", "Norg", "NH4", "NO3", "Porg", "Inorg_P",
            "Phyto", "Detr", "Pathogens", "Alk", "Constituent_i", "Constituent_ii", "Constituent_iii",
            "pH", "Bot_Alg", "TN", "TP", "TSS", "NH3", "Sat_data", "SOD_data",
            "Sediment_1", "Sediment_2", "Sediment_3", "CBODu", "TOC", "TKN"
        ]

        mapeo = {
            "CONDUCTIVIDAD": "Cond",
            "S_INORG": "ISS",
            "OXIGENO_DISUELTO": "DO",
            "DBO_SLOW": "CBODs",
            "DBO5": "CBODf",
            "N_ORG": "Norg",
            "NITROGENO_AMONIACAL": "NH4",
            "NO3": "NO3",
            "P_ORG": "Porg",
            "ORTOFOSFATOS": "Inorg_P",
            "DETRITUS": "Detr",
            "COLIFORMES_TOTALES": "Pathogens",
            "ALCALINIDAD": "Alk",
            "COLIFORMES_TERMOTOLERANTES": "Constituent_i",
            "E_COLI": "Constituent_ii",
            "PH": "pH",
            "SST": "TSS",
            "TN": "TN",
            "TP": "TP",
            "TKN": "TKN"
        }

        # Construcción del diccionario
        stations = []
        for _, row in df.iterrows():
            cons_dict = {k: -99999 for k in constituyentes_base}

            for col, key in mapeo.items():
                if col in df.columns:
                    try:
                        cons_dict[key] = float(row[col])
                    except:
                        cons_dict[key] = -99999

            cons_dict["TN"] = float(row.get("TN", cons_dict["TN"]))
            cons_dict["TP"] = float(row.get("TP", cons_dict["TP"]))
            cons_dict["TKN"] = float(row.get("TKN", cons_dict["TKN"]))

            station = {
                "cwqHwID": 1,
                "dist": float(row.get("X_QUAL2K", 0)),
                "constituents": cons_dict
            }

            stations.append(station)

        return {
            "sheet_name": "WQ Data",
            "nwqd": len(stations),
            "stations": stations
        }