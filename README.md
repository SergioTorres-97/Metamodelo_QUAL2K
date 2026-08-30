# Metamodelo QUAL2K

**Desarrollo de un metamodelo basado en simulaciones de QUAL2K para la estimación de la DBO en el río Jordán – Tunja**

Proyecto de grado — Maestría en Ciencia de Datos
Escuela Colombiana de Ingeniería Julio Garavito

**Autor:** Sergio Torres
**Año:** 2026

---

## Descripción general

Este repositorio contiene el desarrollo completo del proyecto de grado: una interfaz en Python para el modelo de calidad de agua **QUAL2K** (cuyo motor de cálculo es un ejecutable FORTRAN), construida para automatizar la preparación de datos, la ejecución del modelo, el análisis de resultados y la calibración mediante algoritmo genético.

Sobre esa base se construyó, para el caso de estudio del **tramo T1 del río Jordán** (Tunja), un flujo completo de:

- **Análisis de sensibilidad global** (LHS + SRCC) para identificar los predictores más influyentes sobre la DBO.
- Generación de una **base de datos de escenarios** simulados con QUAL2K.
- Entrenamiento y optimización de **metamodelos** de aprendizaje automático (XGBoost, LightGBM, CatBoost y una red neuronal) que aproximan la respuesta de QUAL2K.
- Validación de su **eficiencia computacional** frente al modelo mecanicista.
- Una **aplicación interactiva** (Streamlit) que sirve el metamodelo entrenado para predicción de DBO en tiempo real.

La metodología detallada de todo este flujo — no solo el resumen de este README — está en **[`docs/metodologia.md`](docs/metodologia.md)**.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/SergioTorres-97/Metamodelo_QUAL2K.git
cd Metamodelo_QUAL2K
```

### 2. Crear entorno virtual e instalar dependencias

El paquete `qual2k` se instala desde el subdirectorio `qual2k/` (ahí vive su `pyproject.toml`), no desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

cd qual2k
pip install -e .
cd ..
```

Esto instala también las dependencias de los metamodelos (`xgboost`, `lightgbm`, `catboost`, `torch`, `optuna`, `statsmodels`) y de la app interactiva (`streamlit`, `plotly`). Ver [`qual2k/requirements.txt`](qual2k/requirements.txt) para versiones fijadas.

### 3. Plantillas de datos

Los scripts que usan plantillas Excel (`model/modelo_chicamocha.py`, `tests/*.py`) requieren `PlantillaBaseQ2K.xlsx` para cada tramo. Estas plantillas **no están incluidas en el repositorio** y deben colocarse en:

```
data/templates/<nombre_tramo>/PlantillaBaseQ2K.xlsx
```

Los scripts basados en JSON (`caso_estudio_t_rio_jordan/`, `scripts/sensibilidad.py`) no requieren estas plantillas — usan `caso_estudio_t_rio_jordan/t_rio_jordan_simulacion.json` directamente.

El ejecutable FORTRAN `bin/q2kfortran2_12.exe` **sí está incluido** y se copia automáticamente al directorio de trabajo al ejecutar cada simulación.

---

## Estructura del proyecto

```
Metamodelo_QUAL2K/
├── bin/
│   └── q2kfortran2_12.exe            # Ejecutable FORTRAN (motor de cálculo)
│
├── qual2k/                            # Paquete Python principal (pip install -e .)
│   ├── core/
│   │   ├── model.py                   # Q2KModel — orquestador principal
│   │   ├── config.py                  # Gestión de parámetros y tasas cinéticas
│   │   ├── simulator.py               # Wrapper para ejecución del .exe
│   │   ├── calibrator.py              # Calibración con algoritmo genético (pygad)
│   │   ├── calibrator_general.py      # Pipeline de calibración multi-tramo
│   │   └── calibrator_global.py       # Calibración global (múltiples parámetros a la vez)
│   ├── processing/
│   │   ├── data_processor.py          # Lee Excel → diccionarios
│   │   ├── file_writer.py             # Escribe archivos .q2k
│   │   └── json_loader.py             # Q2KJsonLoader — carga configuración desde JSON
│   └── analysis/
│       ├── results_analyzer.py        # Parsea archivos .out
│       ├── plotter.py                 # Gráficos de resultados
│       └── metricas.py                # KGE, NSE, RMSE, PBIAS (calibración QUAL2K)
│
├── model/                             # Ejecución directa (vía plantillas Excel)
│   └── modelo_chicamocha.py           # Río Chicamocha completo, 7 reaches, calibrado (comprobación)
│
├── scripts/                           # Motores reutilizables (vía JSON)
│   ├── run_from_json.py               # Corre una simulación QUAL2K desde un JSON
│   ├── sensibilidad.py                # Motor de análisis de sensibilidad (LHS + SRCC)
│   ├── figura_srcc_explicacion.py     # Figura didáctica: cómo se calcula el SRCC espacial
│   └── lr_diagnostico.py              # Diagnóstico de supuestos de regresión lineal
│
├── caso_estudio_t_rio_jordan/         # Caso de estudio: tramo T1, río Jordán
│   ├── t_rio_jordan_simulacion.json   # Configuración base calibrada del tramo T1
│   ├── t_rio_jordan_simulacion.py     # Corre una simulación individual (JSON)
│   ├── t_rio_jordan_sensibilidad.py   # Análisis de sensibilidad (36 parámetros, LHS+SRCC)
│   ├── t_rio_jordan_metamodelo_bd.py  # Genera la BD SQLite de escenarios (LHS)
│   └── t_rio_jordan_costo_computacional.py  # QUAL2K vs metamodelo: tiempos de cómputo
│
├── metamodelo/                        # Entrenamiento de metamodelos de DBO
│   ├── datos.py                       # Carga la BD SQLite, define FEATURES/TARGET, split
│   ├── metricas.py                    # R², RMSE, MAE, bias + intervalo conformal
│   ├── xgboost_trainer.py             # XGBoost + Optuna (early stopping)
│   ├── lgbm_trainer.py                # LightGBM + Optuna (early stopping)
│   ├── catboost_trainer.py            # CatBoost + Optuna (early stopping)
│   ├── nn_trainer.py                  # Red neuronal (MLP, PyTorch) + Optuna + conformal
│   ├── lr_trainer.py                  # Regresión lineal (baseline / diagnóstico)
│   └── exportar.py                    # Exporta la BD SQLite completa a Excel
│
├── app_streamlit.py                   # App interactiva: predicción de DBO con el metamodelo NN
├── assets/logo_escuela.png            # Logo institucional usado por la app
├── .streamlit/config.toml             # Tema visual de la app
│
├── docs/
│   └── metodologia.md                 # Metodología completa (las 6 fases + app)
│
├── tests/                             # Scripts de prueba y calibración (tramos antiguos)
├── data/                              # Plantillas Excel (no versionadas)
│   └── templates/
└── resultados/                        # Salidas: sensibilidad, BD, modelos entrenados, figuras
```

---

## Flujo interno del simulador

```
PlantillaBaseQ2K.xlsx  o  configuracion.json
        │
        ▼ data_processor.py  /  json_loader.py
  Diccionarios Python
        │
        ▼ config.py
  Tasas cinéticas + parámetros
        │
        ▼ file_writer.py
   Archivo .q2k
        │
        ▼ simulator.py  (invoca q2kfortran2_12.exe)
   Archivo .out
        │
        ▼ results_analyzer.py
   DataFrame de resultados
        │
        ▼ plotter.py + metricas.py
  Gráficos + KGE / NSE / RMSE
```

---

## Caso de estudio: tramo T1 del río Jordán (Tunja)

El río Jordán es el cauce urbano que atraviesa Tunja y recibe sus principales vertimientos (bypass y efluente tratado de la PTAR Veolia, entre otros) antes de tributar al río Chicamocha. El tramo T1 (`CABECERA` → `PLAYA ARRIBA`, 28.57 km) es el objeto de estudio central de este proyecto: sobre él se construyó todo el flujo de análisis de sensibilidad, generación de escenarios y entrenamiento de metamodelos.

El repositorio incluye además, de forma independiente, la calibración del modelo QUAL2K para el **río Chicamocha completo** (7 reaches, `model/modelo_chicamocha.py`), usada como modelo de comprobación adicional y no como parte del flujo de metamodelado.

La métrica de calibración principal es el **KGE (Kling-Gupta Efficiency)**, calculado sobre múltiples parámetros de calidad del agua (OD, DBO, NTK, NH₄, fósforo, *E. coli*, entre otros).

---

## Flujo de trabajo del metamodelo (tramo T1)

Metodología completa en **[`docs/metodologia.md`](docs/metodologia.md)**. Resumen ejecutable:

```bash
# 1. Simulación individual del tramo T1 (config. calibrada)
python caso_estudio_t_rio_jordan/t_rio_jordan_simulacion.py

# 2. Análisis de sensibilidad global (LHS + SRCC, 36 parámetros)
python caso_estudio_t_rio_jordan/t_rio_jordan_sensibilidad.py

# 3. Generar la base de datos de entrenamiento (LHS sobre 16 variables sensibles)
python caso_estudio_t_rio_jordan/t_rio_jordan_metamodelo_bd.py --n 6000

# 4. Entrenar un metamodelo (ejemplo: XGBoost, con búsqueda de hiperparámetros)
python metamodelo/xgboost_trainer.py
# variantes equivalentes: lgbm_trainer.py, catboost_trainer.py, nn_trainer.py

# 5. Comparar el costo computacional QUAL2K vs metamodelo (NN)
python caso_estudio_t_rio_jordan/t_rio_jordan_costo_computacional.py --n 100

# 6. App interactiva de predicción (sirve el metamodelo NN entrenado)
streamlit run app_streamlit.py
```

Cada script en `caso_estudio_t_rio_jordan/` y `metamodelo/` imprime sus rutas de salida (Excel, figuras, modelos serializados) en `resultados/t_rio_jordan_sensibilidad/` y `resultados/t_rio_jordan_metamodelo/`.

### App interactiva

`app_streamlit.py` sirve directamente el metamodelo de red neuronal ya entrenado (no ejecuta QUAL2K/FORTRAN): permite ajustar los 16 predictores del tramo con sliders y predecir la DBO en uno o varios puntos longitudinales, con su intervalo de predicción al 95 % (conformal prediction). Requiere que `resultados/t_rio_jordan_metamodelo/` contenga `nn_dbo.pt`, `nn_scaler_x.joblib`, `nn_scaler_y.joblib` y `nn_conformal.json` (generados por `metamodelo/nn_trainer.py`).

---

## Dependencias principales

| Paquete | Uso |
|---|---|
| `pandas`, `numpy`, `scipy` | Manejo de datos, muestreo LHS, estadística |
| `matplotlib`, `seaborn`, `plotly` | Visualización estática e interactiva |
| `openpyxl` | Lectura/escritura de archivos Excel |
| `scikit-learn` | Splits, métricas, permutation importance |
| `xgboost`, `lightgbm`, `catboost` | Metamodelos de boosting |
| `torch` | Red neuronal (MLP) |
| `optuna` | Optimización bayesiana de hiperparámetros (TPE) |
| `statsmodels` | Diagnóstico de supuestos de regresión lineal |
| `pygad` | Algoritmo genético para calibración de QUAL2K |
| `streamlit` | App interactiva de predicción |

Ver [`qual2k/requirements.txt`](qual2k/requirements.txt) para versiones exactas.

---

## Autoría

Proyecto de grado de la **Maestría en Ciencia de Datos**, Escuela Colombiana de Ingeniería Julio Garavito.

**Sergio Torres** — 2026
