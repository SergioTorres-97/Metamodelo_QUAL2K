# model/

Script de ejecución del modelo QUAL2K calibrado para el río Chicamocha completo
(7 reaches, tramo de comprobación), usando el paquete `qual2k`.

> Los scripts que existían aquí para Canal Vargas (`modelo_vargas.py`), el
> Tramo 3S / R. Tota-Chiquito (`modelo_tota_chiquito.py`) y el pipeline
> encadenado (`pipeline_modelo_calidad.py`) fueron retirados. Quedan
> referencias sueltas a esos tramos en `tests/uso_basico.py` (Canal Vargas) y
> `tests/pruebas.py` (Tramo 3S), como ejemplos de calibración con
> `qual2k.core.calibrator`.

## Archivo

| Script | Tramo | Reaches |
|---|---|---|
| `modelo_chicamocha.py` | Río Chicamocha (comprobación) | 7 |

## Uso

```bash
python model/modelo_chicamocha.py
```

Carga las plantillas Excel del tramo (`data/templates/Chicamocha/Comprobacion/`),
aplica las tasas cinéticas calibradas por reach (`kaaa`, `kdc`, `kn`, `khp`,
`kdt`), ejecuta QUAL2K, genera las gráficas de resultados y por último
imprime el KGE (Kling-Gupta Efficiency) global de la calibración.

Para correr una simulación equivalente a partir de un archivo JSON de
configuración (en vez de las plantillas Excel), ver
[`caso_estudio_chicamocha_t1/chicamocha_t1_simulacion.py`](../caso_estudio_chicamocha_t1/chicamocha_t1_simulacion.py)
y `scripts/run_from_json.py`.

## Requisitos

- Tener la carpeta `data/templates/Chicamocha/Comprobacion/` con la plantilla
  Excel (`PlantillaBaseQ2K.xlsx`).
- El ejecutable FORTRAN `bin/q2kfortran2_12.exe` presente en la raíz del
  proyecto (se copia automáticamente al directorio de trabajo en cada
  ejecución).
