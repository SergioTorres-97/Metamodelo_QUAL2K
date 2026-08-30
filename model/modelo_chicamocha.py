import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qual2k.core.model import Q2KModel
import warnings
warnings.filterwarnings('ignore')

base = Path(__file__).parent.parent
filepath = str(base / 'data/templates/Chicamocha/Comprobacion')
header_dict = {
    "version": "v2.12",    "rivname": "Chicamocha",
    "filename": "Chicamocha",
    "filedir": filepath,
    "applabel": "Chicamocha (6/27/2012)",
    "xmon": 6, "xday": 27, "xyear": 2012,
    "timezonehour": -6,
    "pco2": 0.000347,
    "dtuser": 4.16666666666667E-03,
    "tf": 5,
    "IMeth": "Euler",
    "IMethpH": "Brent"
}

model = Q2KModel(filepath, header_dict)
model.cargar_plantillas()

n = len(model.data_reaches)
reach_rates_custom = model.config.generar_reach_rates_custom(
    n=n,
    kaaa_list=[2.576395, 3.608709, 3.894834, 3.923470, 3.691096, 3.651164, 2.189518],    # Tasa de aireación
    khc_list=[None] * n,                                                                    # Hidrólisis de carbono
    kdcs_list=[None] * n,                                                                   # Descomposición de carbono lento
    kdc_list=[1.490254, 0.256108, 1.134617, 0.362205, 0.066802, 0.262852, 0.259868],     # Descomposición de carbono rápido
    khn_list=[None] * n,                                                                    # Hidrólisis de nitrógeno
    kn_list=[0.001185, 0.000786, 0.000897, 0.000529, 0.000769, 0.000593, 0.000973],      # Nitrificación
    ki_list=[None] * n,                                                                     # Tasa de denitrificación
    khp_list=[1.095822, 1.220926, 0.662723, 0.770096, 0.544401, 1.302340, 0.260829],     # Hidrólisis de fósforo
    kdt_list=[0.107661, 0.473192, 0.191312, 0.426974, 0.136667, 0.107812, 0.446539]      # Detritos
)

model.configurar_modelo(reach_rates_custom=reach_rates_custom, q_cabecera=1.053E-06)
model.config.actualizar_rates(NINpmin=0.05, NIPpupmax=0.001)
model.generar_archivo_q2k()
model.ejecutar_simulacion()
model.analizar_resultados(generar_graficas=True)
_, kge_global = model.calcular_metricas_calibracion()

print(f'KGE: {kge_global}')
