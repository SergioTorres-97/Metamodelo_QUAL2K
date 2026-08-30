from qual2k.core.model import Q2KModel
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

#Configuración
base = Path(__file__).parent.parent
filepath = f'{base}/data/templates/Chicamocha'
header_dict = {
    "version": "v2.12",
    "rivname": "Chicamocha",
    "filename": "Chicamocha",
    "filedir": filepath,
    "applabel": "Chicamocha (6/27/2012)",
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



model = Q2KModel(filepath, header_dict)
model.cargar_plantillas(filepath + '\\PlantillaBaseQ2K.xlsx')

n = len(model.data_reaches)
print(n)
reach_rates_custom = model.config.generar_reach_rates_custom(
    n=n,
    kaaa_list=[None] * n, # Tasa de aireación
    khc_list=[None] * n, # Hidrólisis de carbono
    kdcs_list=[None] * n, # Descomposición de carbono lento
    kdc_list=[None] * n, # Descomposición de carbono rápido
    khn_list=[None] * n, # Hidrólisis de nitrógeno
    kn_list=[None] * n, # Nitrificación
    ki_list=[None] * n, # Tasa de denitricación
    khp_list=[None] * n, # Hidrólisis de fósforo
    kdt_list=[None] * n # Detritos
)

model.configurar_modelo(reach_rates_custom=reach_rates_custom, q_cabecera= 1.06007E-06)
model.generar_archivo_q2k()
model.ejecutar_simulacion()
model.analizar_resultados()
resultados, kge_global = model.calcular_metricas_calibracion()

print(f'Kge: {kge_global}')