"""
Q2KJsonLoader
=============
Carga la configuración de una simulación QUAL2K desde un archivo JSON y
produce los mismos DataFrames que Q2KDataProcessor lee del Excel.

Uso típico (desde run_from_json.py):
    loader = Q2KJsonLoader("simulacion_001.json")
    loader.cargar()

    model = Q2KModel(loader.header_dict['filedir'], loader.header_dict)
    model.data_reaches = loader.data_reaches
    model.data_sources = loader.data_sources
    model.data_wq      = loader.data_wq

    if loader.rates_override:
        model.config.actualizar_rates(**loader.rates_override)
    if loader.light_override:
        model.config.actualizar_light(**loader.light_override)

    model.configurar_modelo(
        q_cabecera=loader.q_cabecera,
        estacion_cabecera=loader.estacion_cabecera,
        numelem_default=loader.numelem_default,
        reach_rates_custom=loader.reach_rates_custom,
    )

Estructura del JSON:
--------------------
{
  "header": { ... },            # Obligatorio
  "simulacion": { ... },        # Opcional (q_cabecera, numelem, etc.)
  "reaches": [ { ... }, ... ],  # Obligatorio
  "sources": [ { ... }, ... ],  # Obligatorio (puede ser [])
  "wq_data": [ { ... }, ... ],  # Obligatorio (al menos la cabecera)
  "rates":   { ... },           # Opcional – sobreescribe defaults
  "light":   { ... },           # Opcional – sobreescribe defaults
  "reach_rates": { ... }        # Opcional – tasas por tramo
}
"""

import json
import pandas as pd
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Mapeos: clave JSON → columna DataFrame (mismas que usa Q2KDataProcessor)
# ---------------------------------------------------------------------------

_REACHES_MAP = {
    "est_arriba":        "EST_ARRIBA",
    "est_abajo":         "EST_ABAJO",
    "nombre_tramo":      "NOMBRE_TRAMO",
    "x_arriba":          "X_QUAL2K_ARRIBA",
    "x_abajo":           "X_QUAL2K_ABAJO",
    "elev_arriba":       "ELEV_ARRIBA",
    "elev_abajo":        "ELEV_ABAJO",
    "alpha_1":           "ALPHA_1",
    "beta_1":            "BETA_1",
    "alpha_2":           "ALPHA_2",
    "beta_2":            "BETA_2",
    "sombra":            "SOMBRA_[-]",
    "temperatura_aire":  "TEMPERATURA_[C]",
    "temperatura_rocio": "TEMPERATURA_ROCIO_[C]",
    "velocidad_viento":  "VELOCIDAD_DEL_VIENTO_[MS]",
    "cobertura_nubes":   "COBERTURA_NUBES_[-]",
}

_SOURCES_MAP = {
    "nombre":                    "NOMBRE_VERTIMIENTO",
    "tipo":                      "TIPO",
    "x":                         "X_QUAL2K",
    "caudal":                    "CAUDAL",
    "temperatura":               "TEMPERATURA",
    "conductividad":             "CONDUCTIVIDAD",
    "sst":                       "SST",
    "dbo5":                      "DBO5",
    "ntk":                       "NTK",
    "nitrogeno_amoniacal":       "NITROGENO_AMONIACAL",
    "nitritos":                  "NITRITOS",
    "nitratos":                  "NITRATOS",
    "fosforo_total":             "FOSFORO_TOTAL",
    "ortofosfatos":              "ORTOFOSFATOS",
    "oxigeno_disuelto":          "OXIGENO_DISUELTO",
    "coliformes_totales":        "COLIFORMES_TOTALES",
    "alcalinidad":               "ALCALINIDAD",
    "coliformes_termotolerantes":"COLIFORMES_TERMOTOLERANTES",
    "e_coli":                    "E_COLI",
    "pH":                        "pH",
}

_WQDATA_MAP = {
    "nombre_estacion":           "NOMBRE_ESTACIONES",
    "x":                         "X_QUAL2K",
    "caudal":                    "CAUDAL",
    "temperatura":               "TEMPERATURA",
    "conductividad":             "CONDUCTIVIDAD",
    "sst":                       "SST",
    "dbo5":                      "DBO5",
    "ntk":                       "NTK",
    "nitrogeno_amoniacal":       "NITROGENO_AMONIACAL",
    "nitritos":                  "NITRITOS",
    "nitratos":                  "NITRATOS",
    "fosforo_total":             "FOSFORO_TOTAL",
    "ortofosfatos":              "ORTOFOSFATOS",
    "oxigeno_disuelto":          "OXIGENO_DISUELTO",
    "coliformes_totales":        "COLIFORMES_TOTALES",
    "alcalinidad":               "ALCALINIDAD",
    "coliformes_termotolerantes":"COLIFORMES_TERMOTOLERANTES",
    "e_coli":                    "E_COLI",
    "pH":                        "pH",
}

# Campos de reach_rates que acepta generar_reach_rates_custom / reach_rates_custom
_REACH_RATE_KEYS = [
    "kaaa", "vss_rch",
    "khc_rch", "kdcs_rch", "kdc_rch",
    "khn_rch", "von_rch", "kn_rch",
    "ki_rch", "vdi_rch",
    "khp_rch", "vop_rch", "vip_rch",
    "kga_rch", "krea_rch", "kexa_rch", "kdea_rch", "va_rch",
    "kgaF_rch", "kreaF_rch", "kexaF_rch", "kdeaF_rch",
    "kdt_rch", "vdt_rch", "ffast_rch",
]

# Aliases cortos que puede usar el JSON (sin el sufijo _rch)
_REACH_RATE_ALIASES = {k.replace("_rch", ""): k for k in _REACH_RATE_KEYS if k.endswith("_rch")}


class Q2KJsonLoader:
    """
    Carga y valida un JSON de simulación QUAL2K.

    Atributos públicos (disponibles tras llamar a cargar()):
        header_dict       : dict  – configuración del header para Q2KConfig
        data_reaches      : DataFrame – equivalente a la hoja REACHES del Excel
        data_sources      : DataFrame – equivalente a la hoja SOURCES del Excel
        data_wq           : DataFrame – equivalente a la hoja WQ_DATA del Excel
        rates_override    : dict | None – claves a sobreescribir en rates_dict
        light_override    : dict | None – claves a sobreescribir en light_dict
        reach_rates_custom: dict | None – dict con estructura reach_rates para
                            Q2KModel.configurar_modelo(reach_rates_custom=...)
        q_cabecera        : float
        estacion_cabecera : str
        numelem_default   : int
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self._raw: Dict[str, Any] = {}

        # Resultados públicos
        self.header_dict: Dict[str, Any] = {}
        self.data_reaches: Optional[pd.DataFrame] = None
        self.data_sources: Optional[pd.DataFrame] = None
        self.data_wq: Optional[pd.DataFrame] = None
        self.rates_override: Optional[Dict] = None
        self.light_override: Optional[Dict] = None
        self.reach_rates_custom: Optional[Dict] = None
        self.q_cabecera: float = 1.06574e-06
        self.estacion_cabecera: str = "CABECERA"
        self.numelem_default: int = 10

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def cargar(self) -> "Q2KJsonLoader":
        """Lee el JSON y construye todos los atributos. Retorna self (chainable)."""
        with open(self.json_path, encoding="utf-8") as f:
            self._raw = json.load(f)

        self._validar_secciones_obligatorias()
        self.header_dict = self._parsear_header()
        self._parsear_simulacion()
        self.data_reaches = self._parsear_dataframe(self._raw["reaches"], _REACHES_MAP, "reaches")
        self.data_sources = self._parsear_dataframe(self._raw.get("sources", []), _SOURCES_MAP, "sources")
        self.data_wq      = self._parsear_dataframe(self._raw["wq_data"], _WQDATA_MAP, "wq_data")
        self.rates_override = self._filtrar_comentarios(self._raw.get("rates")) or None
        self.light_override = self._filtrar_comentarios(self._raw.get("light")) or None
        self.reach_rates_custom = self._parsear_reach_rates()

        print(f"[Q2KJsonLoader] Cargado: {self.json_path}")
        print(f"  Tramos:  {len(self.data_reaches)}")
        print(f"  Fuentes: {len(self.data_sources)}")
        print(f"  WQ obs:  {len(self.data_wq)}")
        return self

    # ------------------------------------------------------------------
    # Parseo interno
    # ------------------------------------------------------------------

    def _validar_secciones_obligatorias(self):
        for seccion in ("header", "reaches", "wq_data"):
            if seccion not in self._raw:
                raise ValueError(f"[Q2KJsonLoader] Sección obligatoria '{seccion}' no encontrada en {self.json_path}")

    def _parsear_header(self) -> Dict[str, Any]:
        h = self._raw["header"]
        required = ("version", "rivname", "filename", "filedir",
                    "xmon", "xday", "xyear")
        for k in required:
            if k not in h:
                raise ValueError(f"[Q2KJsonLoader] Campo obligatorio '{k}' falta en 'header'")

        # Defaults para campos opcionales del header
        defaults = {
            "applabel":      h.get("rivname", ""),
            "timezonehour":  -5,
            "pco2":          0.000347,
            "dtuser":        4.16666666666667e-03,
            "tf":            5,
            "IMeth":         "Euler",
            "IMethpH":       "Brent",
        }
        return {**defaults, **h}

    def _parsear_simulacion(self):
        """Lee la sección 'simulacion' (opcional) con parámetros de corrida."""
        sim = self._raw.get("simulacion", {})
        self.q_cabecera        = float(sim.get("q_cabecera",       self.q_cabecera))
        self.estacion_cabecera = str(sim.get("estacion_cabecera",  self.estacion_cabecera))
        self.numelem_default   = int(sim.get("numelem_default",    self.numelem_default))

    @staticmethod
    def _parsear_dataframe(
        registros: List[Dict],
        mapeo: Dict[str, str],
        nombre: str,
    ) -> pd.DataFrame:
        """
        Convierte una lista de dicts del JSON en un DataFrame con los nombres
        de columna que espera Q2KDataProcessor.
        """
        if not registros:
            # DataFrame vacío con las columnas correctas
            return pd.DataFrame(columns=list(mapeo.values()))

        filas = []
        for i, reg in enumerate(registros):
            fila = {}
            for clave_json, col_df in mapeo.items():
                if clave_json in reg:
                    fila[col_df] = reg[clave_json]
                # Si la clave no está en el JSON la columna queda NaN
            filas.append(fila)

        df = pd.DataFrame(filas)

        # Asegurar que existan todas las columnas del mapeo (aunque con NaN)
        for col in mapeo.values():
            if col not in df.columns:
                df[col] = float("nan")

        return df

    @staticmethod
    def _filtrar_comentarios(d: Optional[Dict]) -> Optional[Dict]:
        """Elimina claves que empiezan con '_' (comentarios/metadatos en el JSON)."""
        if not d:
            return d
        return {k: v for k, v in d.items() if not k.startswith("_")}

    def _parsear_reach_rates(self) -> Optional[Dict]:
        """
        Convierte la sección 'reach_rates' del JSON al formato que espera
        Q2KModel.configurar_modelo(reach_rates_custom=...).

        El JSON puede especificar cada parámetro como:
          - lista de n valores (uno por tramo)
          - un escalar (se repite para todos los tramos)
          - null (usa el default del modelo)

        Ejemplo en JSON:
            "reach_rates": {
                "kaaa":  [1.82, 3.98, 3.76, 1.82],
                "kdc":   [1.31, 0.18, 0.08, 0.31],
                "kn":    0.001,
                "kdt":   null
            }
        """
        rr_json = self._raw.get("reach_rates")
        if not rr_json:
            return None

        n = len(self.data_reaches)

        def _expandir(valor, clave):
            if valor is None:
                return [None] * n
            if isinstance(valor, list):
                if len(valor) != n:
                    raise ValueError(
                        f"[Q2KJsonLoader] reach_rates['{clave}'] tiene {len(valor)} valores "
                        f"pero hay {n} tramos."
                    )
                return valor
            # escalar → broadcast
            return [valor] * n

        # Construir lista de dicts por tramo
        reaches_out = [{} for _ in range(n)]

        for clave_json, valor in rr_json.items():
            # Resolver nombre canónico (con o sin _rch)
            clave_canon = _REACH_RATE_ALIASES.get(clave_json, clave_json)
            if clave_canon not in _REACH_RATE_KEYS:
                # Puede ser "kaaa" (sin sufijo, sin alias entry) → aceptar tal cual
                clave_canon = clave_json

            valores = _expandir(valor, clave_json)
            for i, v in enumerate(valores):
                reaches_out[i][clave_canon] = v

        # Rellenar claves faltantes con None
        for tramo in reaches_out:
            for k in _REACH_RATE_KEYS:
                tramo.setdefault(k, None)

        return {"nr": n, "reaches": reaches_out}
