# Metodología para el desarrollo de un metamodelo de DBO basado en QUAL2K

## Fase 1. Modelo QUAL2K base y alcance del metamodelo

La metodología se orienta al desarrollo de un metamodelo supervisado capaz de reproducir la respuesta del modelo QUAL2K para la demanda bioquímica de oxígeno carbonácea rápida (DBO) en el tramo de estudio del sistema fluvial Jordán-Chicamocha, asociado al entorno urbano de Tunja. El metamodelo se plantea como un modelo sustituto del simulador mecanicista, con el propósito de reducir el tiempo de evaluación de escenarios y facilitar el análisis exploratorio de alternativas de gestión, manteniendo como referencia física las simulaciones generadas por QUAL2K.

El procedimiento metodológico se organiza en siete fases: (1) configuración del modelo QUAL2K base, (2) automatización computacional, (3) análisis de sensibilidad global, (4) selección de variables y generación de la base de datos, (5) preprocesamiento y entrenamiento de metamodelos, (6) validación predictiva, explicabilidad y eficiencia computacional, y (7) despliegue del metamodelo seleccionado en una herramienta interactiva de predicción. Los resultados derivados de cada fase, tales como índices de sensibilidad, desempeño de modelos, importancia de variables y ganancias de tiempo, deben presentarse posteriormente en el capítulo de resultados.

El flujo general de trabajo es el siguiente:

```text
Modelo QUAL2K base calibrado
        |
        v
Automatización en Python de entradas, ejecución y salidas
        |
        v
Análisis de sensibilidad global LHS + SRCC
        |
        v
Selección de variables sensibles + inclusión de x_km
        |
        v
Generación de base de datos de escenarios QUAL2K
        |
        v
Entrenamiento de metamodelos no lineales (+ baseline lineal de referencia)
        |
        v
Validación predictiva, explicabilidad y eficiencia computacional
        |
        v
Despliegue del metamodelo en herramienta interactiva (Streamlit)
```

### 1.1 Configuración del tramo de estudio

El modelo base se configuró en QUAL2K v2.12 para representar el tramo T1 del río Chicamocha, desde la sección `CABECERA` hasta la sección `PLAYA ABAJO`. El tramo tiene una longitud longitudinal de 28.57 km y se representa como un sistema unidimensional en estado estacionario, adecuado para evaluar cambios espaciales de calidad del agua bajo condiciones de frontera y aportes externos definidos.

El archivo de configuración base corresponde a `examples/chicamocha_t1_simulacion.json`. En dicho archivo se integran los elementos requeridos para la simulación:

- Encabezado general del modelo QUAL2K.
- Condición de frontera aguas arriba.
- Geometría y discretización del tramo.
- Fuentes puntuales, afluentes y captaciones.
- Relaciones hidráulicas caudal-velocidad y caudal-profundidad.
- Tasas cinéticas calibradas.
- Condiciones meteorológicas y parámetros numéricos de simulación.

La simulación se definió con el método de integración de Euler explícito, un paso de tiempo de 0.0041666667 días y un tiempo total de simulación de 5 días. El modelo base se utiliza como simulador de referencia para generar las respuestas que posteriormente alimentan el entrenamiento del metamodelo.

### 1.2 Condición de frontera aguas arriba

La condición de frontera en `CABECERA` define el estado hidrológico y fisicoquímico inicial del agua que ingresa al tramo. En la Fase 1 se incorporó la condición aguas arriba completa disponible para el modelo QUAL2K, sin realizar todavía una selección de variables sensibles. Esta condición incluyó el caudal de entrada y los constituyentes de calidad del agua requeridos para inicializar la simulación, tales como DBO5, oxígeno disuelto, temperatura, pH, nutrientes, sólidos, conductividad y demás variables presentes en el archivo de configuración base.

La función de esta condición dentro del modelo base es establecer el estado inicial del sistema fluvial antes de recibir aportes laterales, vertimientos, afluentes o captaciones. La evaluación de cuáles componentes de la frontera aguas arriba deben perturbarse o conservarse fijos se realiza posteriormente en las fases de análisis de sensibilidad y selección de predictores.

### 1.3 Entradas, salidas, fuentes puntuales y afluentes

En la configuración del modelo base se incorporaron todas las entradas y salidas hidráulicas identificadas para el tramo de estudio. Esta fase no corresponde todavía a una selección de variables sensibles, sino al inventario completo de elementos que modifican el balance de caudal y masa en QUAL2K. Por tanto, se incluyeron vertimientos tratados, descargas no tratadas, afluentes naturales, captaciones y demás intercambios laterales presentes en el tramo.

Cada entrada o salida se localizó longitudinalmente mediante su coordenada `x` dentro del tramo y se caracterizó, según su tipo, mediante caudal y variables fisicoquímicas disponibles. En las entradas con carga de calidad de agua se consideraron variables como DBO5, oxígeno disuelto, temperatura, nutrientes, sólidos, conductividad, pH y otros constituyentes requeridos por QUAL2K. En las salidas o captaciones se representó principalmente la extracción de caudal, de acuerdo con la estructura del modelo.

Dentro del inventario se incluyen, entre otros, los aportes y retiros asociados a:

- Vertimientos municipales o industriales.
- Descargas tratadas y no tratadas.
- Afluentes naturales.
- Captaciones o retiros de agua.
- Intercambios laterales registrados en el archivo de configuración.

La posterior priorización de algunas fuentes o variables, como caudales y concentraciones de DBO5 de aportes específicos, se realiza únicamente en las fases de análisis de sensibilidad y selección de predictores. De esta forma, el modelo QUAL2K base conserva la representación completa del sistema, mientras que el metamodelo se entrena con el subconjunto de variables que resulte metodológicamente justificado.

### 1.4 Representación hidráulica

La hidráulica del tramo se representa mediante relaciones potenciales de gasto:

```math
U = \alpha_1 Q^{\beta_1}
```

```math
H = \alpha_2 Q^{\beta_2}
```

donde `U` es la velocidad media, `H` es la profundidad, `Q` es el caudal, `alpha_1` y `alpha_2` son coeficientes de escala, y `beta_1` y `beta_2` son exponentes hidráulicos. Estas relaciones condicionan el tiempo de residencia, la advección, la dilución y la interacción entre transporte hidráulico y procesos de degradación de materia orgánica.

### 1.5 Tasas cinéticas

El modelo QUAL2K incluye tasas cinéticas que representan procesos biogeoquímicos relevantes para el balance de oxígeno, materia orgánica, nutrientes y otros constituyentes de calidad del agua. En la Fase 1 se incorporaron las tasas cinéticas calibradas necesarias para ejecutar el modelo base completo. Entre las tasas registradas en la configuración se encuentran:

| Parámetro | Proceso representado |
|---|---|
| `kaaa` | Reaireación superficial |
| `kdc` | Oxidación de DBO carbonácea rápida |
| `kn` | Nitrificación |
| `khp` | Hidrólisis del fósforo orgánico |
| `kdt` | Disolución de detrito |

En esta fase, dichas tasas se usan como parte de la parametrización calibrada del modelo mecanicista. La definición de cuáles tasas se perturban, así como los rangos relativos o absolutos utilizados, corresponde a la Fase 3 de análisis de sensibilidad.

## Fase 2. Automatización computacional del modelo QUAL2K

### 2.1 Propósito de la automatización

Para ejecutar múltiples escenarios, se implementó una interfaz de automatización en Python sobre el modelo QUAL2K. Esta automatización permite modificar parámetros, ejecutar el motor FORTRAN y extraer resultados de forma sistemática, evitando la edición manual de archivos de entrada y reduciendo errores operativos.

### 2.2 Lenguaje, entorno y paquetería general

La automatización y el desarrollo de los metamodelos se implementaron en el lenguaje Python, debido a su capacidad para integrar procesamiento de datos, ejecución de modelos externos, análisis estadístico, aprendizaje automático y visualización científica dentro de un mismo flujo reproducible. El paquete local `qual2k` se estructuró como una interfaz entre los archivos de configuración del caso de estudio y el ejecutable FORTRAN de QUAL2K (`q2kfortran2_12.exe`).

El entorno de trabajo se definió para Python 3.10 o superior, de acuerdo con la configuración del proyecto. Las dependencias se organizaron en torno a seis grupos funcionales:

| Grupo | Paquetes principales | Uso metodológico |
|---|---|---|
| Cálculo numérico y datos | `numpy`, `pandas`, `scipy` | Manejo de arreglos, tablas, muestreo, estadística y transformación de datos |
| Lectura y escritura de archivos | `openpyxl`, `json`, `sqlite3`, `joblib` | Lectura de plantillas o salidas, almacenamiento tabular, base SQLite y serialización de modelos |
| Visualización científica | `matplotlib`, `seaborn` | Construcción de gráficos de sensibilidad, desempeño, residuos e importancia |
| Modelación y validación ML | `scikit-learn` | Partición de datos, métricas, escalamiento, validación cruzada, Random Forest, ExtraTrees e importancia por permutación |
| Boosting y redes neuronales | `xgboost`, `lightgbm`, `catboost`, `torch` | Entrenamiento de metamodelos no lineales y red neuronal MLP |
| Optimización y análisis complementario | `optuna`, `statsmodels`, `pygad` | Búsqueda bayesiana de hiperparámetros, análisis estadístico auxiliar y calibración previa del modelo |

La paquetería permite que el flujo completo sea ejecutable desde scripts, manteniendo separación entre el modelo mecanicista (`qual2k`), la generación de escenarios (`examples` y `scripts`) y el entrenamiento de metamodelos (`metamodelo`).

### 2.3 Flujo computacional

El flujo computacional automatizado comprende:

1. Lectura del archivo JSON base.
2. Modificación programática de parámetros hidráulicos, cinéticos, condiciones de cabecera y fuentes.
3. Escritura del archivo `.q2k` requerido por QUAL2K.
4. Ejecución del motor `q2kfortran2_12.exe`.
5. Lectura del archivo de salida `.out`.
6. Conversión de resultados a estructuras tabulares.
7. Extracción de perfiles longitudinales de variables de calidad del agua.
8. Almacenamiento de escenarios y resultados en archivos estructurados.

### 2.4 Componentes de código utilizados

Los scripts utilizados para esta fase son:

- `scripts/sensibilidad.py`: funciones generales para muestreo, modificación de configuraciones, ejecución de simulaciones y cálculo de sensibilidad.
- `examples/chicamocha_t1_sensibilidad.py`: configuración específica del análisis de sensibilidad.
- `examples/chicamocha_t1_metamodelo_bd.py`: generación de la base de datos de simulaciones para entrenamiento.
- `metamodelo/datos.py`: carga de la base de datos y definición del vector de predictores.

## Fase 3. Análisis de sensibilidad global

### 3.1 Propósito

El análisis de sensibilidad global se realiza para identificar los parámetros de entrada que tienen mayor influencia sobre la respuesta del modelo QUAL2K, en particular sobre la DBO carbonácea rápida. Esta fase permite reducir la dimensionalidad del problema antes del entrenamiento del metamodelo, priorizar variables físicamente relevantes y evitar la inclusión de predictores con baja contribución al comportamiento de la variable objetivo.

El análisis se plantea de forma global, variando simultáneamente todos los parámetros dentro de rangos plausibles. Este enfoque permite capturar efectos combinados y relaciones monotónicas no necesariamente lineales entre entradas y salidas.

### 3.2 Muestreo por Hipercubo Latino

La exploración del espacio de parámetros se realiza mediante Muestreo por Hipercubo Latino (`Latin Hypercube Sampling`, LHS), implementado con `scipy.stats.qmc.LatinHypercube`. Para cada parámetro se define una distribución uniforme dentro de un rango físico u operativo plausible.

El LHS divide el rango de cada variable en estratos equiprobables y toma una muestra por estrato, lo que mejora la cobertura del espacio de entrada respecto a un muestreo aleatorio simple. Cada fila del diseño LHS representa una combinación de parámetros que se utiliza para ejecutar una simulación completa de QUAL2K.

La configuración general del análisis contempla:

| Elemento | Configuración metodológica |
|---|---|
| Tipo de muestreo | LHS uniforme |
| Variables de entrada | Parámetros hidráulicos, cinéticos, fuentes y cabecera |
| Variables de salida | OD, DBO rápida, temperatura, amonio y nitrato |
| Ejecución | Automatizada y paralelizada |
| Indicador de sensibilidad | SRCC |

### 3.3 Parámetros evaluados

Se evaluaron **36 parámetros** en total, agrupados en cuatro familias: hidráulicos (4), cinéticos (5), fuentes puntuales y afluentes (23) y condición de frontera aguas arriba (5).

#### 3.3.1 Parámetros hidráulicos

El rango es absoluto: los valores muestreados son los valores directos del parámetro, no multiplicadores.

| Parámetro | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---:|---:|---:|---|
| `alpha_1` | 0.0958 | 0.03 | 0.40 | m/s·(m³/s)^−β |
| `beta_1` | 0.7558 | 0.35 | 0.95 | adimensional |
| `alpha_2` | 1.1037 | 0.30 | 2.50 | m·(m³/s)^−β |
| `beta_2` | 0.1403 | 0.05 | 0.45 | adimensional |

#### 3.3.2 Tasas cinéticas

El rango es relativo: el valor muestreado multiplica al valor calibrado. Los valores calibrados son los ajustados durante la fase de calibración del modelo QUAL2K base.

| Parámetro | Proceso | Valor calibrado | Rango (factor) |
|---|---|---:|---:|
| `kaaa` | Reaireación superficial | 2.576 | 0.5 a 3.0 × calibrado |
| `kdc` | Oxidación DBO carbonácea rápida | 1.490 | 0.3 a 3.0 × calibrado |
| `kn` | Nitrificación | 0.001185 | 0.3 a 3.0 × calibrado |
| `khp` | Hidrólisis del fósforo orgánico | 1.096 | 0.3 a 3.0 × calibrado |
| `kdt` | Disolución de detrito | 0.108 | 0.3 a 3.0 × calibrado |

#### 3.3.3 Fuentes puntuales y afluentes

Se evaluaron seis fuentes puntuales y afluentes del tramo T1. Para cada una se variaron las variables fisicoquímicas disponibles en el modelo base: caudal, DBO5, oxígeno disuelto y temperatura, según corresponda. Los rangos se definen a partir del valor calibrado y límites físicos u operativos plausibles. El tipo de rango es absoluto en todos los casos (los valores muestreados son los valores directos, no multiplicadores del calibrado).

**BY-PASS-VEOLIA AGUAS DE TUNJA S.A. E.S.P.**
Bypass sin tratamiento; mayor carga orgánica del tramo.

| Variable | Nombre en código | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---|---:|---:|---:|---|
| Caudal | `caudal_bypass` | 0.270 | 0.050 | 0.500 | m³/s |
| DBO5 | `dbo5_bypass` | 263.0 | 50.0 | 600.0 | mg/L |
| Oxígeno disuelto | `od_bypass` | 3.34 | 0.5 | 6.0 | mg/L |
| Temperatura | `temp_bypass` | 20.85 | 15.0 | 30.0 | °C |

**VEOLIA AGUAS DE TUNJA S.A. E.S.P.**
Efluente tratado de la PTAR principal; segunda mayor carga del tramo.

| Variable | Nombre en código | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---|---:|---:|---:|---|
| Caudal | `caudal_veolia` | 0.190 | 0.050 | 0.500 | m³/s |
| DBO5 | `dbo5_veolia` | 32.75 | 5.0 | 150.0 | mg/L |
| Oxígeno disuelto | `od_veolia` | 4.44 | 1.0 | 8.0 | mg/L |

**URBASER TUNJA S.A E.S.P.**
PTAR secundaria; efluente tratado con carga orgánica baja.

| Variable | Nombre en código | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---|---:|---:|---:|---|
| DBO5 | `dbo5_urbaser` | 1.8 | 0.5 | 30.0 | mg/L |
| Oxígeno disuelto | `od_urbaser` | 4.83 | 1.0 | 8.0 | mg/L |
| Temperatura | `temp_urbaser` | 18.4 | 14.0 | 25.0 | °C |

> El caudal de URBASER (0.0015 m³/s) es muy pequeño respecto al total del tramo y se mantuvo fijo en el valor calibrado.

**R. LA VEGA**
Afluente natural; agua limpia, caudal variable.

| Variable | Nombre en código | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---|---:|---:|---:|---|
| Caudal | `caudal_la_vega` | 0.030 | 0.001 | 1.000 | m³/s |
| DBO5 | `dbo5_la_vega` | 3.63 | 1.0 | 15.0 | mg/L |
| Oxígeno disuelto | `od_la_vega` | 5.50 | 3.0 | 9.0 | mg/L |
| Temperatura | `temp_la_vega` | 17.3 | 12.0 | 22.0 | °C |

**Q. HONDA**
Vertimiento con DBO5 elevada relativa a su caudal.

| Variable | Nombre en código | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---|---:|---:|---:|---|
| Caudal | `caudal_honda` | 0.002 | 0.001 | 0.500 | m³/s |
| DBO5 | `dbo5_honda` | 58.0 | 10.0 | 200.0 | mg/L |
| Oxígeno disuelto | `od_honda` | 3.55 | 0.5 | 7.0 | mg/L |
| Temperatura | `temp_honda` | 17.6 | 13.0 | 23.0 | °C |

**R. PIEDRAS**
Afluente con el mayor caudal del tramo; agua limpia con alta capacidad de dilución.

| Variable | Nombre en código | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---|---:|---:|---:|---|
| Caudal | `caudal_piedras` | 0.263 | 0.020 | 2.000 | m³/s |
| DBO5 | `dbo5_piedras` | 5.0 | 1.0 | 20.0 | mg/L |
| Oxígeno disuelto | `od_piedras` | 5.51 | 3.0 | 9.0 | mg/L |
| Temperatura | `temp_piedras` | 15.5 | 10.0 | 20.0 | °C |

En total, esta sección aporta 23 parámetros al diseño LHS: 4 (BY-PASS-VEOLIA) + 3 (VEOLIA) + 3 (URBASER) + 4 (R. LA VEGA) + 4 (Q. HONDA) + 4 (R. PIEDRAS) + 1 parámetro fijo (caudal URBASER).

#### 3.3.4 Condición de frontera aguas arriba

El rango es absoluto. El caudal de cabecera es bajo porque corresponde a la cabecera alta del río antes de recibir los aportes del área urbana.

| Variable | Nombre en código | Valor calibrado | Rango mínimo | Rango máximo | Unidad |
|---|---|---:|---:|---:|---|
| Caudal | `caudal_cabecera` | 0.029 | 0.005 | 0.150 | m³/s |
| DBO5 | `dbo5_cabecera` | 2.5 | 0.5 | 8.0 | mg/L |
| Oxígeno disuelto | `od_cabecera` | 6.2 | 4.0 | 9.0 | mg/L |
| Temperatura | `temp_cabecera` | 17.6 | 12.0 | 22.0 | °C |
| pH | `pH_cabecera` | 7.1 | 6.5 | 8.5 | adimensional |

### 3.4 Variables de respuesta

El análisis de sensibilidad se aplica sobre variables de salida representativas del estado de calidad del agua:

| Variable en QUAL2K | Interpretación |
|---|---|
| `dissolved_oxygen` | Oxígeno disuelto |
| `carbonaceous_bod_fast` | DBO carbonácea rápida |
| `water_temp_c` | Temperatura del agua |
| `ammonium` | Amonio |
| `nitrate` | Nitrato |

La variable prioritaria para el metamodelo es `carbonaceous_bod_fast`, que se almacena posteriormente como `dbo_mg_L`.

### 3.5 Coeficiente de correlación de rangos de Spearman

La sensibilidad de cada parámetro se cuantifica mediante el Coeficiente de Correlación de Rangos de Spearman (`SRCC`). Para un parámetro de entrada `x` y una variable de salida `y`, el SRCC mide la asociación monotónica entre sus rangos:

```math
\rho_s = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
```

donde `d_i` es la diferencia entre los rangos de `x_i` y `y_i`, y `n` es el número de simulaciones válidas. El SRCC toma valores entre -1 y 1. Valores positivos indican relaciones directas, valores negativos relaciones inversas, y valores cercanos a cero baja asociación monotónica.

Se calcula sensibilidad en dos niveles:

- Sensibilidad global sobre la media espacial de cada variable de salida.
- Sensibilidad espacial por punto longitudinal, usando el perfil simulado por QUAL2K.

Los valores, mapas de calor y diagramas de tornado derivados de este análisis deben presentarse en el capítulo de resultados.

## Fase 4. Selección de variables y generación de la base de datos

### 4.1 Criterios de selección de variables

La selección de variables para el metamodelo se realiza combinando criterios estadísticos y físicos. El criterio estadístico se basa en la magnitud del SRCC respecto a la DBO carbonácea rápida. El criterio físico permite conservar variables que, aunque no tengan la mayor asociación promedio, controlan procesos relevantes como transporte, dilución, carga orgánica o tiempo de residencia.

Se priorizan las siguientes familias de predictores:

- Variables directamente asociadas a carga orgánica: caudales y DBO de fuentes.
- Variables hidráulicas que modifican velocidad, profundidad y tiempo de residencia.
- Tasas cinéticas relacionadas con reaeración, oxidación de DBO y detrito.
- Condiciones de frontera aguas arriba.
- Posición longitudinal del punto de predicción.

### 4.2 Variables muestreadas para construir escenarios

La generación de escenarios se configura con variables seleccionadas a partir del análisis de sensibilidad y del criterio físico:

| Grupo | Variables |
|---|---|
| Hidráulicas | `alpha_1`, `beta_1`, `alpha_2` |
| Cinéticas | `kaaa`, `kdc`, `kdt` |
| BY-PASS-VEOLIA | `caudal_bypass`, `dbo5_bypass` |
| VEOLIA | `caudal_veolia`, `dbo5_veolia` |
| R. LA VEGA | `caudal_la_vega`, `dbo5_la_vega` |
| Q. HONDA | `dbo5_honda` |
| R. PIEDRAS | `caudal_piedras`, `dbo5_piedras` |
| Cabecera | `caudal_cabecera`, `dbo5_cabecera`, `od_cabecera` |

Estas variables se almacenan en la base de datos para mantener trazabilidad del escenario. Posteriormente, el vector final usado por los entrenadores puede excluir variables con baja contribución directa sobre la DBO, según el criterio de selección adoptado.

### 4.3 Vector de predictores del metamodelo

El vector de predictores usado para el entrenamiento se define en `metamodelo/datos.py`. Está compuesto por variables hidrológicas, cinéticas, de carga y una coordenada espacial:

```text
FEATURES = [
  alpha_1,
  beta_1,
  kaaa,
  kdc,
  kdt,
  caudal_bypass,
  dbo5_bypass,
  caudal_veolia,
  dbo5_veolia,
  caudal_la_vega,
  dbo5_la_vega,
  dbo5_honda,
  caudal_piedras,
  dbo5_piedras,
  caudal_cabecera,
  dbo5_cabecera,
  x_km
]
```

La variable objetivo es:

```text
TARGET = dbo_mg_L
```

correspondiente a la concentración de DBO carbonácea rápida simulada por QUAL2K en cada punto longitudinal.

### 4.4 Incorporación de la distancia longitudinal `x_km`

QUAL2K genera un perfil longitudinal de DBO para cada escenario. Para que el metamodelo pueda reproducir la variación espacial de la DBO, cada simulación se transforma en múltiples registros, uno por cada punto longitudinal del perfil. En cada registro se repiten las variables del escenario y se añade la coordenada `x_km`.

La función aproximada por el metamodelo puede escribirse como:

```math
\widehat{DBO} = f(\theta, x_{km})
```

donde `theta` representa las variables hidrológicas, cinéticas y de carga, y `x_km` identifica la ubicación longitudinal del punto de predicción.

La inclusión de `x_km` permite que el modelo aprenda:

- Gradientes longitudinales de DBO.
- Efectos locales de fuentes puntuales.
- Cambios por dilución y transporte.
- Atenuación o incremento de DBO a lo largo del tramo.

### 4.5 Generación de la base de datos

La base de datos se genera con el script `examples/chicamocha_t1_metamodelo_bd.py`. Para cada escenario se realiza:

1. Generación de una muestra LHS en el espacio de variables seleccionadas.
2. Copia del JSON base.
3. Aplicación de perturbaciones en parámetros, fuentes y cabecera.
4. Ejecución de QUAL2K.
5. Extracción del perfil longitudinal de `carbonaceous_bod_fast`.
6. Renombramiento de la variable de salida como `dbo_mg_L`.
7. Almacenamiento de entradas, `x_km` y DBO simulada en SQLite.
8. Limpieza de archivos temporales de ejecución.

La base se almacena en:

```text
resultados/chicamocha_t1_metamodelo/simulaciones_Q2K.db
```

en la tabla:

```text
simulaciones
```

### 4.6 Estructura de la tabla de simulaciones

La tabla `simulaciones` contiene el identificador de escenario, variables muestreadas, coordenada espacial y variable objetivo:

```text
sim_id
alpha_1
beta_1
alpha_2
kaaa
kdc
kdt
caudal_bypass
dbo5_bypass
caudal_veolia
dbo5_veolia
caudal_la_vega
dbo5_la_vega
dbo5_honda
caudal_piedras
dbo5_piedras
caudal_cabecera
dbo5_cabecera
od_cabecera
x_km
dbo_mg_L
```

El campo `sim_id` permite agrupar todos los puntos espaciales de una misma simulación y es fundamental para realizar particiones de entrenamiento y prueba sin fuga de información.

### 4.7 Partición de datos por simulación

Dado que cada simulación produce múltiples filas espaciales, la partición aleatoria por fila podría generar fuga de información: puntos de un mismo escenario podrían quedar simultáneamente en entrenamiento y prueba. Para evitarlo, la división se realiza por `sim_id`.

La partición metodológica se define como:

| Conjunto | Fracción |
|---|---:|
| Entrenamiento | 70 % |
| Validación | 10 % |
| Prueba | 20 % |

El conjunto de entrenamiento se usa para ajustar los modelos, el conjunto de validación para seleccionar hiperparámetros, aplicar detención temprana y calcular importancia de variables, y el conjunto de prueba para la evaluación final.

## Fase 5. Preprocesamiento y entrenamiento de metamodelos

### 5.1 Tratamiento de variables de entrada

Los modelos basados en árboles se entrenan con las variables en escala original, debido a que los árboles de decisión son invariantes a transformaciones monotónicas de escala. Esto aplica a Random Forest, ExtraTrees, XGBoost, LightGBM y CatBoost.

Para la red neuronal multicapa, las variables de entrada se estandarizan con `StandardScaler`, ajustado exclusivamente sobre el conjunto de entrenamiento y aplicado posteriormente a validación y prueba. Esta práctica evita transferir información estadística de los conjuntos de evaluación al proceso de entrenamiento.

### 5.2 Transformación de la variable objetivo

La DBO puede presentar una distribución asimétrica positiva, especialmente en escenarios con cargas orgánicas elevadas. Para reducir la influencia excesiva de valores extremos en modelos basados en pérdidas cuadráticas, los modelos de boosting y ensambles de árboles se entrenan sobre la transformación:

```math
y' = \log(1 + y)
```

Las predicciones se transforman nuevamente a unidades originales mediante:

```math
\widehat{y} = \exp(\widehat{y'}) - 1
```

Todas las métricas se calculan en mg/L, después de aplicar la transformación inversa.

En la red neuronal, el objetivo se escala mediante `StandardScaler` y se recupera a escala original antes de la evaluación.

### 5.3 Metamodelos evaluados

Se evalúan algoritmos no lineales capaces de capturar interacciones entre caudales, cargas, cinéticas y posición espacial. Los metamodelos considerados son:

- Random Forest.
- ExtraTrees.
- XGBoost.
- LightGBM.
- CatBoost.
- Red neuronal multicapa.

Los modelos de regresión lineal no se incluyen en esta metodología, debido a que el objetivo de esta etapa es construir y comparar metamodelos no lineales con capacidad para aproximar la respuesta compleja de QUAL2K.

#### 5.3.0 Protocolo de optimización de hiperparámetros

Para los modelos con costo de entrenamiento no trivial a la escala del conjunto de datos (1,560,000 filas, 30,000 simulaciones), la búsqueda de hiperparámetros se realiza mediante optimización bayesiana con Optuna (muestreador TPE — *Tree-structured Parzen Estimator*), bajo un protocolo único aplicado de forma consistente a XGBoost, LightGBM, CatBoost y la red neuronal:

- **Partición**: se reutiliza el mismo split train/validación/prueba (70/10/20 %) descrito en la sección 5.2, agrupado por `sim_id` para evitar que filas de un mismo escenario aparezcan simultáneamente en particiones distintas.
- **Evaluación por trial**: cada combinación de hiperparámetros propuesta por Optuna se entrena una única vez sobre el conjunto de entrenamiento y se evalúa contra el conjunto de validación fijo, en vez de repetir el ajuste sobre varios pliegues de validación cruzada (`GroupKFold`). Esto evita multiplicar el costo de entrenamiento por el número de pliegues.
- **Detención temprana**: los modelos de boosting (XGBoost, LightGBM, CatBoost) aprovechan su mecanismo nativo de *early stopping* contra el conjunto de validación, de modo que cada trial se detiene en su número óptimo de árboles sin necesidad de que el número de árboles sea, en sí mismo, un hiperparámetro buscado. La red neuronal utiliza el mismo criterio de paciencia que su entrenamiento final, complementado con poda de trials (`MedianPruner`) que descarta configuraciones poco prometedoras antes de completar el presupuesto máximo de épocas, comparando su trayectoria de error de validación contra la de trials previos.
- **Métrica de selección**: R² (o RMSE, según el modelo) evaluado en el conjunto de validación, en la misma escala en que se entrena el modelo (log1p de la DBO para los modelos de boosting).

Este protocolo se adoptó tras observar que una búsqueda con validación cruzada `GroupKFold` sin detención temprana —usada inicialmente para Random Forest— resultaba computacionalmente inviable a esta escala de datos. Random Forest y ExtraTrees no cuentan con un mecanismo de *early stopping* nativo como los modelos de boosting, por lo que se les aplica una variante propia del protocolo, basada en la estimación *out-of-bag* (OOB) en vez de un split de validación adicional (ver sección 5.3.1). Con esta adaptación, todos los modelos de la comparación final —incluidos Random Forest y ExtraTrees— cuentan con una búsqueda de hiperparámetros automatizada y computacionalmente viable.

#### 5.3.1 Random Forest y ExtraTrees

El script `metamodelo/rf_trainer.py` implementa `RandomForestRegressor` y `ExtraTreesRegressor`. Ambos métodos construyen ensambles de árboles de decisión y promedian sus predicciones para reducir varianza. Random Forest usa muestras bootstrap y selección aleatoria de variables por nodo; ExtraTrees incorpora mayor aleatoriedad en los puntos de corte, incrementando la diversidad del ensamble.

La configuración inicial (usada cuando la búsqueda de hiperparámetros está desactivada) considera:

| Hiperparámetro | Valor inicial |
|---|---:|
| `n_estimators` | 500 |
| `max_depth` | Sin límite |
| `min_samples_leaf` | 5 |
| `min_samples_split` | 2 |
| `max_features` | `sqrt` |
| `bootstrap` | Verdadero |
| `ccp_alpha` | 0.0 |

Una primera versión de la búsqueda bayesiana usaba validación cruzada `GroupKFold` por `sim_id`, evitando que filas del mismo escenario aparecieran en pliegues distintos. Sin embargo, dado que Random Forest no cuenta con un mecanismo de detención temprana —cada árbol del ensamble se construye por completo en cada evaluación—, esa búsqueda resultó computacionalmente inviable a la escala del conjunto de datos: una ejecución con `n_trials=20` y `cv_folds=5` no alcanzó a completarse tras más de 3 días de cómputo.

Para resolverlo, la búsqueda se rediseñó aprovechando que Random Forest y ExtraTrees ya dejan, por construcción, una fracción de las filas fuera de la muestra bootstrap de cada árbol (*out-of-bag*, OOB). Ese subconjunto cumple el mismo papel que un fold de validación, sin requerir `GroupKFold` ni una partición adicional. Cada trial de Optuna crece el bosque en bloques de árboles mediante `warm_start` y se detiene apenas el R² OOB deja de mejorar durante un número fijo de bloques consecutivos — el equivalente, para modelos de árboles sin *early stopping* nativo, del mecanismo usado en los modelos de boosting (sección 5.3.0). Un `MedianPruner` de Optuna complementa este criterio, descartando entre trials las configuraciones cuya trayectoria de R² OOB es claramente peor que la de trials anteriores con el mismo número de árboles.

El estudio de Optuna se persiste además en una base SQLite (`optuna_rf.db`) en vez de mantenerse solo en memoria, de modo que una búsqueda interrumpida puede reanudarse desde el último trial completado sin perder el cómputo ya invertido — una precaución relevante dado el tiempo de cómputo que exige esta búsqueda incluso con detención temprana.

Con este rediseño, la búsqueda de 20 trials se completó en un tiempo de cómputo práctico, y Random Forest queda incluido en la comparación final de metamodelos con hiperparámetros optimizados, en las mismas condiciones que XGBoost, LightGBM, CatBoost y la red neuronal. La búsqueda optimizada se ejecutó para `RandomForestRegressor`; `ExtraTreesRegressor` comparte el mismo mecanismo (`--modelo extra`) pero no se ejecutó de forma independiente en esta versión, por lo que se documenta aquí únicamente con su configuración inicial.

#### 5.3.2 XGBoost

El script `metamodelo/xgboost_trainer.py` implementa `XGBRegressor`. XGBoost construye árboles secuenciales mediante boosting por gradiente, incorporando regularización, submuestreo y detención temprana.

El script permite activar una búsqueda bayesiana de hiperparámetros mediante Optuna, siguiendo el protocolo de split único train/validación con detención temprana descrito en la sección 5.3.0.

La configuración inicial contempla:

| Hiperparámetro | Valor inicial |
|---|---:|
| `n_estimators` | 600 |
| `max_depth` | 8 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `min_child_weight` | 5 |
| `gamma` | 0.1 |
| `early_stopping_rounds` | 40 |

#### 5.3.3 LightGBM

El script `metamodelo/lgbm_trainer.py` implementa `LGBMRegressor`. LightGBM utiliza árboles con crecimiento hoja-a-hoja y particiones basadas en histogramas, lo cual resulta adecuado para bases de datos tabulares de gran tamaño.

El script permite activar una búsqueda bayesiana de hiperparámetros mediante Optuna, siguiendo el mismo protocolo de split único train/validación con detención temprana descrito en la sección 5.3.0.

La configuración inicial es:

| Hiperparámetro | Valor inicial |
|---|---:|
| `n_estimators` | 800 |
| `num_leaves` | 127 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `min_child_samples` | 20 |
| `early_stopping` | 40 |

#### 5.3.4 CatBoost

El script `metamodelo/catboost_trainer.py` implementa `CatBoostRegressor`. CatBoost usa boosting ordenado y árboles simétricos, lo que aporta regularización implícita y reduce sesgos asociados al ajuste secuencial.

El script permite activar una búsqueda bayesiana de hiperparámetros mediante Optuna, siguiendo el mismo protocolo de split único train/validación con detención temprana descrito en la sección 5.3.0.

La configuración inicial es:

| Hiperparámetro | Valor inicial |
|---|---:|
| `iterations` | 1000 |
| `depth` | 6 |
| `learning_rate` | 0.05 |
| `l2_leaf_reg` | 3.0 |
| `subsample` | 0.8 |
| `rsm` | 0.8 |
| `min_data_in_leaf` | 20 |
| `early_stopping_rounds` | 50 |

#### 5.3.5 Red neuronal multicapa

El script `metamodelo/nn_trainer.py` implementa una red neuronal tipo perceptrón multicapa (`MLP`) en PyTorch. La arquitectura general es:

```text
Entrada -> Linear -> ReLU -> Dropout -> ... -> Linear -> Salida
```

con 17 entradas (16 variables sensibles + `x_km`) y 1 salida (DBO en escala normalizada).

**Espacio de búsqueda de arquitectura.** A diferencia de los modelos de boosting, aquí la búsqueda bayesiana (Optuna, protocolo de la sección 5.3.0) también optimiza la arquitectura de la red: número de capas ocultas (entre 2 y 3) y número de neuronas por capa, elegido de forma categórica entre `{32, 64, 128, 256}` para cada capa — un espacio discreto y acotado (se excluyeron explícitamente capas de 512 neuronas y `batch_size=128` por su costo de cómputo en CPU sin mejora observada en trials preliminares), además de la tasa de aprendizaje (`log-uniform`, 1e-4 a 1e-2), el dropout (0.0 a 0.5), el tamaño de lote (categórico entre `{256, 512, 1024}`) y la regularización `weight_decay` (`log-uniform`, 1e-6 a 1e-2). Cada trial de búsqueda se limita a un máximo de 60 épocas y se somete a poda (`MedianPruner`, `n_startup_trials=3`, `n_warmup_steps=8`) si su trayectoria de error de validación resulta claramente peor que la de trials anteriores en la misma época.

**Arquitectura final seleccionada.** El modelo que actualmente se distribuye en `resultados/chicamocha_t1_metamodelo/nn_dbo.pt` (y que usan tanto `chicamocha_t1_costo_computacional.py` como `app_streamlit.py`) corresponde a los hiperparámetros hallados por una búsqueda previa, fijados en el bloque `__main__` de `nn_trainer.py` para reproducibilidad (`buscar=False`):

| Elemento | Valor final |
|---|---:|
| Arquitectura (capas ocultas) | `[32, 128]` (2 capas: 17→32→128→1) |
| Épocas máximas | 300 |
| Optimizador | Adam |
| Tasa de aprendizaje inicial | 0.002607 |
| Tamaño de lote | 256 |
| Dropout | 0.01029 |
| `weight_decay` | 5.337e-06 |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, paciencia 15 épocas, `min_lr` 1e-6) |
| Paciencia de early stopping | 50 épocas |

**Checkpointing.** El entrenamiento guarda el estado completo (pesos, optimizador, época, hiperparámetros) en `nn_checkpoint.pt` al final de cada época. Si el script se reinicia con exactamente los mismos hiperparámetros (arquitectura, `lr`, `batch_size`, etc.), retoma automáticamente desde el último checkpoint en vez de reentrenar desde cero; el checkpoint se elimina al completar el entrenamiento con éxito. Esto protege el cómputo invertido ante interrupciones del entorno de ejecución, igual que la persistencia en SQLite de la búsqueda de Random Forest (sección 5.3.1).

La pérdida implementada corresponde a un RMSE ponderado:

```math
L = \sqrt{\frac{1}{N}\sum_{i=1}^{N} w_i(\widehat{y_i} - y_i)^2}
```

con:

```math
w_i = 1 + \frac{y_{orig,i}}{y_{max}}
```

Esta ponderación busca dar mayor relevancia a escenarios con concentraciones elevadas de DBO.

#### 5.3.6 Regresión lineal — baseline y diagnóstico de supuestos

Los modelos de regresión lineal se excluyen de la comparación de metamodelos no lineales (sección 5.3), pero sí se ejecutan como **baseline de referencia** y como **evidencia empírica** de que la relación entre los predictores y la DBO no es adecuadamente lineal, justificando así el uso de modelos no lineales.

`metamodelo/lr_trainer.py` ajusta regresión lineal ordinaria (OLS) y, opcionalmente, `RidgeCV`/`LassoCV` (búsqueda de `alpha` por validación cruzada, 5 pliegues), sobre la partición 80/20 (train/test, por `sim_id`) y **en la escala original de la DBO (mg/L)**, sin la transformación `log1p` usada por los demás modelos.

Adicionalmente, tanto `lr_trainer.py` como el script independiente `scripts/lr_diagnostico.py` evalúan los supuestos clásicos del modelo lineal, sobre una submuestra de 10 000 filas (por el costo de las pruebas que involucran inversión de matrices):

| Supuesto | Prueba |
|---|---|
| Linealidad | RESET (Ramsey) |
| Independencia de residuos | Durbin-Watson |
| Homocedasticidad | Breusch-Pagan |
| Normalidad de residuos | Anderson-Darling, Kolmogorov-Smirnov |
| Multicolinealidad | VIF (factor de inflación de varianza), número de condición |
| Observaciones influyentes | Distancia de Cook, leverage, residuos estudentizados |

Los resultados de este diagnóstico (ver `resultados_lr.xlsx` y `figuras_lr_diagnostico/`) sustentan, junto con el desempeño comparativamente bajo de la regresión lineal frente a los metamodelos no lineales, la decisión metodológica de priorizar Random Forest, ExtraTrees, XGBoost, LightGBM, CatBoost y la red neuronal como candidatos al metamodelo final.

## Fase 6. Validación, explicabilidad y eficiencia computacional

### 6.1 Validación predictiva

La validación predictiva compara las predicciones del metamodelo con las salidas de QUAL2K en el conjunto de prueba. Las métricas se calculan en unidades originales de DBO (mg/L).

Se emplean las siguientes métricas:

```math
R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i-\widehat{y_i})^2}{\sum_{i=1}^{N}(y_i-\bar{y})^2}
```

```math
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat{y_i})^2}
```

```math
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i-\widehat{y_i}|
```

```math
BIAS = \frac{1}{N}\sum_{i=1}^{N}(\widehat{y_i}-y_i)
```

El `R2` permite evaluar la proporción de varianza explicada, el `RMSE` penaliza errores grandes, el `MAE` cuantifica el error promedio absoluto y el `BIAS` identifica tendencias sistemáticas de sobrestimación o subestimación.

Además del desempeño predictivo, se buscó evaluar la estabilidad del modelo entre los subconjuntos de entrenamiento, validación y prueba, como criterio para identificar señales de sobreajuste: un metamodelo apropiado debía mantener un desempeño consistente en datos no utilizados durante el ajuste. Este reporte explícito de las tres métricas (R², RMSE, MAE, sesgo) para Train, Val y Test se implementó de forma completa para **Random Forest** (`rf_trainer.py`, hoja `metricas` de `resultados_rf.xlsx`). Para XGBoost, LightGBM, CatBoost, la red neuronal y la regresión lineal, el conjunto de validación se usa activamente durante el ajuste — como criterio de detención temprana (boosting y red neuronal) y como señal de la búsqueda bayesiana de hiperparámetros (sección 5.3.0) —, pero el reporte final de métricas de estos modelos compara únicamente Train y Test; la estabilidad se sustenta en que la propia detención temprana evita ajustar el modelo más allá del punto de mejor desempeño en validación, más que en una comparación explícita de las tres métricas para los tres conjuntos.

### 6.1.1 Intervalos de predicción por conformal prediction

Además de la predicción puntual, la red neuronal reporta un intervalo de predicción mediante *split conformal prediction* (Lei et al., 2018), implementado en `metamodelo/metricas.py::intervalo_conformal`. El procedimiento calcula el margen `qhat` como el cuantil empírico de los residuos absolutos del conjunto de prueba:

```math
\hat{q} = \text{cuantil}_{1-\alpha}\big(\,|y_i - \widehat{y_i}|\,\big)
```

y construye el intervalo de predicción para una nueva observación como `[ŷ − q̂, ŷ + q̂]`. Se calculan dos niveles de cobertura, `α = 0.10` (90 %) y `α = 0.05` (95 %), y ambos `qhat` se guardan en `resultados/chicamocha_t1_metamodelo/nn_conformal.json`. Este intervalo, al no depender de supuestos distribucionales sobre los residuos, es el que se muestra en la aplicación interactiva (`app_streamlit.py`, ver Fase 7) junto con la predicción puntual.

### 6.2 Análisis gráfico de desempeño

Además de las métricas numéricas, se generan gráficos de diagnóstico para cada metamodelo:

- Curvas de aprendizaje.
- Gráficos observado versus predicho.
- Diagramas de residuos.
- Gráficos de importancia de variables.

Estos productos permiten evaluar visualmente ajuste, dispersión, sesgo, heterogeneidad de errores y comportamiento del modelo en distintos rangos de DBO.

### 6.3 Validación de explicabilidad

La explicabilidad se evalúa contrastando la importancia de variables de los metamodelos con los resultados del análisis de sensibilidad del modelo QUAL2K. Esta comparación permite verificar si el metamodelo reproduce relaciones coherentes con los procesos físico-químicos del sistema.

Para los modelos basados en árboles se calculan:

- Importancia nativa del modelo, basada en reducción de impureza o ganancia.
- Importancia por permutación, calculada como la caída en `R2` al permutar cada variable.

Para la red neuronal se emplea importancia por permutación mediante una envoltura compatible con `sklearn`. La importancia por permutación se calcula sobre el conjunto de validación y permite una comparación agnóstica entre familias de modelos.

### 6.4 Validación de eficiencia computacional

La eficiencia computacional se evalúa comparando el tiempo requerido por QUAL2K para simular un escenario con el tiempo requerido por el metamodelo para producir un perfil equivalente de DBO.

La ganancia de eficiencia se calcula como:

```math
Ganancia = \frac{t_{QUAL2K}}{t_{metamodelo}}
```

donde `t_QUAL2K` corresponde al tiempo de ejecución del modelo mecanicista completo y `t_metamodelo` al tiempo de inferencia del modelo sustituto. Esta comparación permite cuantificar la utilidad práctica del metamodelo para exploración rápida de escenarios.

Esta validación se implementa en `examples/chicamocha_t1_costo_computacional.py`: genera N escenarios LHS nuevos (por defecto sobre las mismas 16 variables sensibles de la Fase 4) y, para cada uno, cronometra por separado la corrida completa de QUAL2K (escritura de config + ejecución FORTRAN + lectura de resultados) y la inferencia del metamodelo de red neuronal sobre los mismos puntos `x_km`, forzando la inferencia a CPU para comparar en igualdad de condiciones de hardware. El primer llamado a la red neuronal en el proceso incluye el costo de inicialización de PyTorch (carga de kernels/threads), no representativo del régimen estable; ese punto se excluye del promedio, la desviación estándar y el speedup reportados, aunque se conserva en los datos crudos (`costo_computacional.xlsx`) para trazabilidad. El script reporta el tiempo promedio ± desviación estándar de cada enfoque, el speedup resultante, y una extrapolación de tiempo acumulado (`costo_computacional.png`) a tamaños típicos de un análisis de incertidumbre o calibración global (500, 5 000 y 50 000 corridas).

### 6.5 Selección del metamodelo final

La selección del metamodelo final se realiza considerando simultáneamente:

- Desempeño predictivo en el conjunto de prueba.
- Magnitud del sesgo.
- Estabilidad entre entrenamiento, validación y prueba.
- Coherencia física de la importancia de variables.
- Costo computacional de entrenamiento e inferencia.
- Facilidad de serialización e integración posterior en una herramienta interactiva.

El modelo seleccionado debe reportarse en el capítulo de resultados, junto con su desempeño, análisis de errores, explicación de variables y comparación de eficiencia frente a QUAL2K.

## Fase 7. Herramienta interactiva de predicción

Como cierre práctico del flujo, el metamodelo de red neuronal seleccionado (arquitectura final, sección 5.3.5) se sirve mediante una aplicación interactiva construida con Streamlit (`app_streamlit.py`, ejecutable con `streamlit run app_streamlit.py`). La aplicación **no ejecuta QUAL2K/FORTRAN**: carga directamente el modelo entrenado (`nn_dbo.pt`), los escaladores (`nn_scaler_x.joblib`, `nn_scaler_y.joblib`) y la calibración conformal (`nn_conformal.json`), por lo que la predicción es prácticamente instantánea, en línea con la ganancia de eficiencia cuantificada en la sección 6.4.

La interfaz agrupa los 16 predictores del tramo T1 en tarjetas por fuente (parámetros hidráulicos, tasas cinéticas, cabecera, R. La Vega, By-Pass Veolia, Veolia tratada, Q. Honda, R. Piedras), cada uno ajustable mediante deslizador o valor numérico exacto dentro de su rango de sensibilidad (Fase 3.3). Permite tres modos de predicción a lo largo del tramo (0 a 34.56 km):

- Un solo punto `x_km`.
- Una lista arbitraria de puntos `x_km`.
- Un perfil longitudinal completo (5 a 50 puntos equiespaciados).

El resultado se presenta como valor puntual (o curva, para varios puntos) junto con su intervalo de predicción al 95 % obtenido por conformal prediction (sección 6.1.1), graficado como banda de incertidumbre alrededor de la curva de DBO predicha. Esta herramienta materializa el propósito planteado en la Fase 1: reducir el tiempo de evaluación de escenarios y facilitar el análisis exploratorio de alternativas de gestión sin requerir una instalación funcional de QUAL2K.

## Material complementario A. Productos computacionales generados

El flujo metodológico contempla la generación de los siguientes artefactos:

| Producto | Ruta |
|---|---|
| Configuración base QUAL2K | `examples/chicamocha_t1_simulacion.json` |
| Script de sensibilidad | `examples/chicamocha_t1_sensibilidad.py` |
| Motor general de sensibilidad | `scripts/sensibilidad.py` |
| Salidas de sensibilidad | `resultados/chicamocha_t1_sensibilidad/` |
| Generador de base de datos | `examples/chicamocha_t1_metamodelo_bd.py` |
| Base SQLite | `resultados/chicamocha_t1_metamodelo/simulaciones_Q2K.db` |
| Definición de predictores | `metamodelo/datos.py` |
| Entrenador Random Forest / ExtraTrees | `metamodelo/rf_trainer.py` |
| Entrenador XGBoost | `metamodelo/xgboost_trainer.py` |
| Entrenador LightGBM | `metamodelo/lgbm_trainer.py` |
| Entrenador CatBoost | `metamodelo/catboost_trainer.py` |
| Entrenador MLP | `metamodelo/nn_trainer.py` |
| Baseline / diagnóstico regresión lineal | `metamodelo/lr_trainer.py`, `scripts/lr_diagnostico.py` |
| Modelos serializados | `resultados/chicamocha_t1_metamodelo/*.joblib` y `*.pt` |
| Métricas y predicciones | `resultados/chicamocha_t1_metamodelo/resultados_*.xlsx` |
| Calibración conformal (intervalos NN) | `resultados/chicamocha_t1_metamodelo/nn_conformal.json` |
| Exportador de la BD a Excel | `metamodelo/exportar.py` |
| Comparación de costo computacional | `examples/chicamocha_t1_costo_computacional.py` |
| Salidas de costo computacional | `resultados/chicamocha_t1_metamodelo/costo_computacional.{xlsx,png}` |
| Aplicación interactiva de predicción | `app_streamlit.py` |
| Figura didáctica del SRCC espacial | `scripts/figura_srcc_explicacion.py` (ilustrativa, con datos sintéticos — no forma parte del flujo de resultados) |

Este apartado se incluye como material de soporte para reproducibilidad, pero los valores generados por estos productos deben discutirse en el capítulo de resultados.

## Material complementario B. Trazabilidad y reproducibilidad

La trazabilidad del procedimiento se garantiza mediante:

- Uso de archivos JSON para registrar la configuración explícita del modelo base.
- Semillas aleatorias fijas en muestreo, particiones y entrenamiento.
- Almacenamiento de cada escenario con identificador `sim_id`.
- Separación por simulación completa para evitar fuga de información espacial.
- Conservación de resultados en bases de datos, archivos tabulares y figuras.
- Serialización de modelos entrenados y escaladores.
- Registro de configuraciones de hiperparámetros en archivos de salida.

Este esquema permite auditar el origen de los datos, repetir entrenamientos, comparar metamodelos y actualizar la base de simulaciones si el modelo QUAL2K es recalibrado.

## Material complementario C. Síntesis del flujo metodológico

La metodología propuesta parte de un modelo QUAL2K base calibrado, automatiza su ejecución con Python y utiliza análisis de sensibilidad global para identificar variables relevantes. Posteriormente, genera una base de datos sintética de escenarios, incorpora la distancia longitudinal `x_km` como predictor espacial y entrena metamodelos no lineales para aproximar la DBO simulada por QUAL2K, contrastados contra un baseline de regresión lineal que justifica empíricamente la necesidad de modelos no lineales. Los metamodelos se validan mediante métricas predictivas, análisis de explicabilidad, intervalos de predicción por conformal prediction y comparación de eficiencia computacional. Finalmente, el metamodelo seleccionado se despliega en una aplicación interactiva que permite explorar escenarios sin ejecutar QUAL2K.

La presentación de valores numéricos, rankings, gráficos definitivos y selección final del modelo debe reservarse para el capítulo de resultados.
