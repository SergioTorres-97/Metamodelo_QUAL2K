# Desarrollo y evaluación de metamodelos

Este apartado presenta el desarrollo de los metamodelos sustitutos de QUAL2K para la
demanda bioquímica de oxígeno carbonácea rápida (DBO) en el tramo T1 del río Chicamocha,
así como su evaluación predictiva, de explicabilidad y de eficiencia computacional. Se
describe primero la construcción y el entrenamiento de los modelos (desarrollo) y,
posteriormente, se reportan y analizan los resultados obtenidos (evaluación). Los valores
numéricos provienen de las corridas registradas en
`resultados/chicamocha_t1_metamodelo/`, sobre la base de datos de 20 000 simulaciones
descrita en §3.3.

## 1. Desarrollo de los metamodelos

### 1.1 Conjunto de datos de entrenamiento

El entrenamiento se realizó sobre la base de datos de escenarios sintéticos generada con
QUAL2K (fase de generación de la base de datos, §3.3). La base final, almacenada en
`simulaciones_Q2K.db` (tabla `simulaciones`), contiene **20 000 simulaciones
independientes**, cada una discretizada en **52 puntos longitudinales** a lo largo de los
28.57 km del tramo (de la cabecera a Playa Abajo), para un total de **1 040 000
registros**. Cada registro reúne 11 predictores del escenario, la coordenada longitudinal
`x_km` y la DBO simulada por QUAL2K como variable objetivo (`dbo_mg_L`).

La variable objetivo cubre un rango amplio, entre 0.15 y 551.25 mg/L, con una media de
96.22 mg/L. Esta dispersión refleja la diversidad de los escenarios muestreados, que van
desde condiciones de agua limpia hasta descargas con alta carga orgánica, y condiciona
las decisiones de preprocesamiento descritas a continuación.

La partición se realizó **por simulación** (`sim_id`) y no por fila, para impedir que
puntos longitudinales de un mismo escenario aparecieran simultáneamente en conjuntos
distintos (fuga de información espacial). Las fracciones adoptadas fueron 70 % para
entrenamiento, 10 % para validación y 20 % para prueba; el conjunto de prueba corresponde
así a 4 000 simulaciones (208 000 registros), reservadas exclusivamente para la
evaluación final.

### 1.2 Preprocesamiento

Se aplicaron dos tratamientos diferenciados según la familia de modelo:

- **Escala de los predictores.** Los modelos de boosting (XGBoost, LightGBM, CatBoost) se
  entrenaron con las variables en escala original, aprovechando su invarianza frente a
  transformaciones monotónicas. La red neuronal, en cambio, requiere entradas
  estandarizadas: se ajustó un `StandardScaler` únicamente sobre el conjunto de
  entrenamiento y se aplicó posteriormente a validación y prueba.
- **Transformación de la variable objetivo.** Dada la asimetría positiva de la DBO, los
  modelos de boosting se entrenaron sobre la transformación `y' = log(1 + y)`, revirtiendo
  las predicciones a mg/L antes de calcular las métricas. En la red neuronal, el objetivo
  se estandarizó con `StandardScaler` y se recuperó a escala original para la evaluación.
  Todas las métricas reportadas están en mg/L.

### 1.3 Familias de metamodelos y baseline de referencia

Se desarrollaron tres metamodelos de boosting (XGBoost, LightGBM y CatBoost) y una red
neuronal multicapa (MLP), capaces de capturar interacciones entre caudales, cargas,
cinéticas y posición longitudinal. Adicionalmente, se ajustó una regresión lineal
ordinaria como **baseline de referencia**, no para competir como metamodelo, sino para
cuantificar el desempeño alcanzable con un modelo lineal y así justificar empíricamente el
uso de modelos no lineales (ver Sección 2.4).

### 1.4 Optimización de hiperparámetros

Los hiperparámetros de LightGBM y XGBoost se ajustaron sobre esta base de datos mediante
optimización bayesiana con Optuna (muestreador TPE, 20 *trials*), bajo un protocolo común:
partición única entrenamiento/validación agrupada por `sim_id`, evaluación de cada
combinación una sola vez contra el conjunto de validación fijo, y detención temprana. Para
CatBoost y la red neuronal se reutilizaron como configuración final los hiperparámetros
hallados en una búsqueda bayesiana previa (sin repetir la búsqueda sobre esta base), por lo
que su configuración debe leerse como un punto de partida razonable y no como el óptimo
específico de estos 20 000 escenarios. La configuración final de cada modelo se resume en
la Tabla 1.

**Tabla 1. Hiperparámetros finales de cada metamodelo.**

| Modelo | Configuración final | Origen |
|---|---|---|
| XGBoost | `n_estimators`=2000 (385 usadas, *early stopping*), `max_depth`=9, `learning_rate`=0.0285, `subsample`=0.863, `colsample_bytree`=0.988, `reg_alpha`=0.912, `reg_lambda`=3.790, `min_child_weight`=20, `gamma`=3.432 | Búsqueda Optuna sobre esta base (20 *trials*) |
| LightGBM | `n_estimators`=6000 (6000 usadas), `num_leaves`=39, `learning_rate`=0.0319, `subsample`=0.673, `colsample_bytree`=0.905, `reg_alpha`=0.0009, `reg_lambda`=0.201, `min_child_samples`=74, `max_depth`=8 | Búsqueda Optuna sobre esta base (20 *trials*) |
| CatBoost | `iterations`=2000 (1999 usadas), `depth`=9, `learning_rate`=0.0844, `l2_leaf_reg`=4.819, `subsample`=0.728, `rsm`=0.780, `min_data_in_leaf`=67 | Reutilizado de búsqueda previa |
| Red neuronal (MLP) | Arquitectura `[32, 128]` (12→32→128→1), `lr`=0.0026, `batch_size`=256, `dropout`=0.0103, `weight_decay`=5.34e-6, Adam + `ReduceLROnPlateau`; convergió en 251 épocas (paciencia 50) | Reutilizado de búsqueda previa |

El entrenamiento de la red neuronal minimizó un RMSE ponderado, con pesos crecientes en
función de la magnitud de la DBO, con el fin de dar mayor relevancia a los escenarios de
carga orgánica elevada.

## 2. Evaluación de los metamodelos

La evaluación se realizó sobre el conjunto de prueba (208 000 registros de 4 000
simulaciones no vistas durante el entrenamiento), con métricas calculadas en mg/L.

### 2.1 Desempeño predictivo

La Tabla 2 reporta el desempeño de los cuatro metamodelos y del baseline lineal sobre el
conjunto de prueba, ordenados por R².

**Tabla 2. Desempeño predictivo sobre el conjunto de prueba (mg/L).**

| Modelo | R² | RMSE | MAE | Sesgo |
|---|---:|---:|---:|---:|
| Red neuronal (MLP) | 0.9993 | 2.36 | 1.61 | +0.05 |
| CatBoost | 0.9974 | 4.48 | 2.48 | −0.30 |
| LightGBM | 0.9961 | 5.46 | 3.07 | −0.51 |
| XGBoost | 0.9881 | 9.51 | 5.35 | −1.96 |
| Regresión lineal (baseline) | 0.6025 | 54.94 | 43.02 | −0.09 |

**Análisis.** La red neuronal obtiene el mejor desempeño, explicando el 99.93 % de la
varianza de la DBO con un RMSE de 2.36 mg/L, sobre una variable cuyo rango supera los
550 mg/L. Los modelos de boosting la siguen a mayor distancia (R² entre 0.988 y 0.997).
Esta brecha es más amplia que la observada en corridas anteriores del pipeline sobre una
base de datos distinta, y es consistente con que CatBoost y la red neuronal partieron de
hiperparámetros reutilizados (no reoptimizados sobre estos 20 000 escenarios), mientras que
LightGBM y XGBoost sí recibieron una búsqueda Optuna fresca; aun así, ninguno de los tres
modelos de boosting iguala el desempeño de la red neuronal.

Dos rasgos del sesgo merecen atención. Primero, la red neuronal presenta un sesgo
prácticamente nulo (+0.05 mg/L), frente al sesgo negativo sistemático de los modelos de
boosting (entre −0.30 y −1.96 mg/L): estos tienden a **subestimar** la DBO. Este
comportamiento es consistente con la naturaleza de los ensambles de árboles, que predicen
por promediado/combinación de hojas y no extrapolan bien hacia los picos extremos de DBO
que ocurren aguas abajo de la descarga del bypass; la ponderación por magnitud incorporada
en la pérdida de la red neuronal (Sección 1.4) actúa justamente en sentido contrario,
corrigiendo esa subestimación en los escenarios de carga alta. Segundo, XGBoost es el que
más se aleja de la red neuronal (RMSE = 9.51 mg/L, sesgo = −1.96 mg/L); esto coincide con
que, en esta corrida, su búsqueda de hiperparámetros convergió a una regularización alta
(`gamma`=3.43) que detuvo el entrenamiento en solo 385 de los 2000 árboles disponibles.

### 2.2 Estabilidad y ausencia de sobreajuste

La comparación entre el desempeño en entrenamiento y en prueba (Tabla 3) permite descartar
sobreajuste en los modelos evaluados.

**Tabla 3. Estabilidad entrenamiento vs. prueba (R² / RMSE en mg/L).**

| Modelo | R² train | R² test | RMSE train | RMSE test |
|---|---:|---:|---:|---:|
| Red neuronal (MLP) | 0.9994 | 0.9993 | 2.17 | 2.36 |
| CatBoost | 0.9992 | 0.9974 | 2.42 | 4.48 |
| LightGBM | 0.9990 | 0.9961 | 2.76 | 5.46 |
| XGBoost | 0.9940 | 0.9881 | 6.74 | 9.51 |

**Análisis.** La brecha entre entrenamiento y prueba es reducida en todos los casos. La red
neuronal es la más estable (RMSE que pasa de 2.17 a 2.36 mg/L), lo que indica una
generalización robusta favorecida por la partición por `sim_id` y el amplio tamaño
muestral. En XGBoost la brecha es mayor (6.74 → 9.51 mg/L), coherente con la limitación de
árboles útiles señalada en 2.1; no se trata de sobreajuste (el desempeño en prueba no es
peor, en términos relativos, que en entrenamiento), sino de una capacidad de ajuste menor
en esta configuración concreta.

### 2.3 Análisis gráfico de residuos

Los gráficos de diagnóstico generados para cada metamodelo (dispersión observado vs.
predicho, residuos y curvas de aprendizaje, en `figuras_*/`) confirman las lecturas
numéricas. Para la red neuronal y los modelos de boosting, la nube observado-predicho se
alinea estrechamente con la diagonal en todo el rango de DBO, y los residuos se distribuyen
sin estructura sistemática. En los modelos de boosting se aprecia una leve tendencia al
aplanamiento en los valores altos de DBO, coherente con su sesgo negativo. La curva de
aprendizaje de la red neuronal muestra un descenso conjunto de los errores de entrenamiento
y validación que converge sin divergencia entre ambos, señal adicional de ausencia de
sobreajuste.

### 2.4 Baseline lineal y justificación de la no linealidad

El baseline de regresión lineal alcanza un R² de 0.6025 sobre prueba (RMSE de 54.94 mg/L),
lo que indica que un modelo lineal deja sin explicar cerca del 40 % de la varianza de la
DBO. Para fundamentar formalmente la inadecuación del modelo lineal, se evaluaron sus
supuestos clásicos (Tabla 4).

**Tabla 4. Diagnóstico de supuestos del modelo lineal.**

| Supuesto | Prueba | Resultado | Estado |
|---|---|---|---|
| Linealidad | RESET (Ramsey) | F = 54 394.1 (p < 0.05) | Violado |
| Independencia | Durbin-Watson | DW = 0.168 | Violado (autocorrelación positiva) |
| Homocedasticidad | Breusch-Pagan | LM = 153 288.8 (p < 0.05) | Violado (heterocedasticidad) |
| Normalidad | Anderson-Darling / KS | A² = 1216.7 ; D = 0.020 | Violado |
| Multicolinealidad | VIF / número de condición | VIF_máx = 1.00 ; cond. = 5490.8 | VIF aceptable; condición severa |

**Análisis.** Todos los supuestos estructurales del modelo lineal se incumplen. La prueba
RESET confirma la existencia de estructura no lineal no capturada, y el estadístico
Durbin-Watson cercano a cero refleja la fuerte autocorrelación esperable entre puntos
longitudinales consecutivos de una misma simulación. Es de notar que la multicolinealidad
entre predictores es prácticamente nula (VIF ≈ 1.0), consecuencia directa del muestreo por
Hipercubo Latino, que genera variables de entrada casi ortogonales; el número de condición
elevado es atribuible a las diferencias de escala entre predictores, no a redundancia entre
ellos. En conjunto, el R² moderado del baseline y la violación sistemática de los supuestos
justifican empíricamente la decisión metodológica de emplear metamodelos no lineales.

### 2.5 Explicabilidad e importancia de variables

La importancia de variables se calculó por permutación (caída de R² al permutar cada
predictor) sobre un conjunto de evaluación, lo que permite una comparación agnóstica entre
familias de modelos. La Tabla 5 presenta el ranking de la red neuronal, representativo del
patrón común a los cuatro metamodelos.

**Tabla 5. Importancia por permutación — red neuronal (todos los predictores).**

| Predictor | Importancia | Interpretación física |
|---|---:|---|
| `x_km` | 1.459 | Posición longitudinal: gradiente espacial de la DBO |
| `dbo5_bypass` | 0.307 | Carga orgánica del bypass sin tratamiento (mayor aporte) |
| `dbo5_veolia` | 0.286 | Carga del efluente tratado de la PTAR principal |
| `caudal_la_vega` | 0.214 | Dilución por el afluente R. La Vega |
| `caudal_bypass` | 0.100 | Caudal del bypass: masa de DBO descargada |
| `caudal_veolia` | 0.089 | Caudal del efluente Veolia |
| `alpha_1` | 0.043 | Coeficiente hidráulico (velocidad, tiempo de residencia) |
| `caudal_cabecera` | 0.033 | Caudal de cabecera: dilución de fondo |
| `dbo5_la_vega` | 0.020 | Carga orgánica del afluente R. La Vega |
| `dbo5_cabecera` | 0.007 | DBO de cabecera: condición de frontera aguas arriba |
| `kaaa` | 0.005 | Reaireación superficial |
| `kdc` | 0.003 | Oxidación de la DBO carbonácea rápida |

**Análisis.** El ranking de importancia es **consistente entre los cuatro metamodelos**:
todos coinciden en el mismo orden de los predictores dominantes —`x_km`, seguido de la carga
del bypass (`dbo5_bypass`) y del efluente Veolia (`dbo5_veolia`), y la dilución de La Vega
(`caudal_la_vega`)—, con las tasas cinéticas (`kaaa`, `kdc`) en las últimas posiciones. Esta
convergencia es relevante en dos sentidos. Primero, como validación de explicabilidad: los
predictores más influyentes son precisamente los que controlan la carga orgánica y la
dilución del tramo, en coherencia con los procesos físico-químicos representados por
QUAL2K. El metamodelo no solo reproduce el valor de la DBO, sino también la jerarquía causal
correcta de sus factores. Segundo, la coincidencia entre modelos de familias distintas
(boosting y redes) refuerza que el patrón hallado es una propiedad del sistema modelado y no
un artefacto de un algoritmo particular. Cabe notar que, respecto a corridas anteriores del
pipeline sobre una base de datos con más predictores, `dbo5_veolia` ganó relevancia relativa
(pasó a la tercera posición), reflejando el nuevo espacio de muestreo de la base actual.

### 2.6 Intervalos de predicción por conformal prediction

Además de la predicción puntual, se calibraron intervalos de predicción distribution-free
sobre la red neuronal, bajo dos esquemas complementarios, ambos calibrados sobre el
conjunto de prueba (208 000 registros, no usado en entrenamiento ni en la selección de
hiperparámetros) y con garantía teórica de cobertura marginal ≥ 1 − α bajo intercambiabilidad.

**a) Split conformal prediction (margen fijo).** Sobre los residuos absolutos de la red
puntual se calculó el margen `qhat` (Lei et al., 2018):

| Cobertura nominal | Margen `qhat` |
|---|---:|
| 90 % | ± 3.37 mg/L |
| 95 % | ± 4.39 mg/L |

**b) Conformalized Quantile Regression, CQR (ancho adaptativo).** Es el esquema
efectivamente desplegado en la herramienta interactiva. Se entrenó una segunda red con la
misma arquitectura, partición y ponderación que el modelo puntual, pero con pérdida
*pinball* sobre cinco cuantiles (0.025, 0.05, mediana, 0.95, 0.975); los cuantiles se
calibraron con la corrección conformal de Romano, Patterson & Candès (2019), produciendo
intervalos de ancho variable.

Para evitar la circularidad de calibrar y evaluar con el mismo conjunto, el conjunto de
prueba (4 000 simulaciones, 208 000 registros) se dividió, por `sim_id` y con una semilla
independiente de la usada en la partición train/val/test, en dos mitades disjuntas
(`scripts/validar_cqr_honesto.py`):

- **Calibración** (2 000 simulaciones, 104 000 registros): se usa únicamente para calcular
  la corrección conformal `q_correction`. Es la que efectivamente queda embebida en
  `nn_cqr.json` y en la herramienta interactiva.
- **Evaluación** (2 000 simulaciones, 104 000 registros, *held-out*): nunca participa en la
  calibración; se usa únicamente para medir, de forma honesta, el **PICP** (*Prediction
  Interval Coverage Probability*, fracción de valores reales de QUAL2K dentro del intervalo)
  y el **MPIW** (*Mean Prediction Interval Width*, ancho promedio del intervalo en mg/L):

| Nivel | PICP (held-out) | MPIW (held-out) | Ancho mín. | Ancho máx. |
|---|---:|---:|---:|---:|
| 90 % | 89.83 % | 6.62 mg/L | 0.00 mg/L | 133.55 mg/L |
| 95 % | 94.57 % | 8.23 mg/L | 0.00 mg/L | 151.89 mg/L |

**Análisis.** Los márgenes del esquema (a) son estrechos frente al rango de la variable
(0.15–551.25 mg/L): con 95 % de cobertura nominal, la incertidumbre de predicción puntual
es de apenas ±4.39 mg/L. El esquema (b), adoptado en la herramienta interactiva, reporta un
PICP medido en un subconjunto que nunca participó en la calibración (89.83 % y 94.57 %,
frente a coberturas nominales de 90 % y 95 %) y un MPIW de 6.6–8.2 mg/L en promedio, con
intervalos más angostos donde el modelo tiene mayor confianza y más anchos en zonas de mayor
incertidumbre (hasta ≈134–152 mg/L en los casos más extremos). La cercanía entre el PICP
observado y el nominal —con una diferencia de apenas 0.2–1.7 puntos porcentuales, del orden
del error de muestreo esperable con 104 000 observaciones— constituye evidencia honesta de
que el método generaliza a datos no usados en su calibración, algo que no podía afirmarse
midiendo la cobertura sobre el propio conjunto de calibración. El MPIW resultó prácticamente
idéntico al obtenido en la medición circular original (6.56→6.62 mg/L; 8.18→8.23 mg/L), lo
que indica que el ancho de los intervalos es estable frente a qué mitad del conjunto de
prueba se usa para calibrar.

### 2.7 Eficiencia computacional

La ganancia de eficiencia se cuantificó comparando, sobre 100 escenarios nuevos, el tiempo
de una corrida completa de QUAL2K (escritura de configuración, ejecución del motor FORTRAN
y lectura de resultados) frente al tiempo de inferencia de la red neuronal sobre los mismos
puntos longitudinales, forzando la inferencia a CPU para comparar en igualdad de hardware.
El primer llamado a la red (inicialización de PyTorch, ~754 ms) se excluyó del promedio por
no ser representativo del régimen estable.

**Tabla 7. Comparación de tiempos de ejecución (100 escenarios).**

| Enfoque | Tiempo por escenario | Aceleración |
|---|---:|---:|
| QUAL2K (mecanicista) | 3.43 ± 0.54 s | — |
| Metamodelo (red neuronal, CPU) | 0.0018 ± 0.0008 s | ≈ 2065× |

La Tabla 8 extrapola estos tiempos a tamaños de campaña típicos de un análisis de
incertidumbre o de calibración global.

**Tabla 8. Tiempo acumulado extrapolado.**

| N.º de corridas | QUAL2K | Metamodelo |
|---:|---:|---:|
| 500 | 28.6 min | 0.9 s |
| 5 000 | 4.77 h | 9.1 s |
| 50 000 | 47.65 h | 1.5 min |

**Análisis.** El metamodelo es del orden de **2065 veces más rápido** que QUAL2K. La
diferencia se traduce en que una campaña de 50 000 escenarios —inviable de forma
interactiva con el modelo mecanicista, que requeriría cerca de dos días de cómputo
continuo— se resuelve con el metamodelo en menos de dos minutos. Esta ganancia materializa
el propósito del trabajo: habilitar el análisis exploratorio rápido de alternativas de
gestión sin ejecutar el simulador completo.

### 2.8 Selección del metamodelo final

La selección integró desempeño predictivo, sesgo, estabilidad, coherencia física de la
importancia de variables, costo computacional y facilidad de despliegue. La **red neuronal
multicapa** resultó seleccionada como metamodelo final por combinar:

- El mejor desempeño predictivo (R² = 0.9993; RMSE = 2.36 mg/L en prueba).
- El sesgo más bajo (+0.05 mg/L) y la mayor estabilidad entrenamiento-prueba.
- Importancia de variables físicamente coherente con QUAL2K y con el análisis de
  sensibilidad.
- Inferencia prácticamente instantánea (≈2065× más rápida que QUAL2K) y una huella de
  serialización reducida (modelo de ~22–24 KB frente a los 3.8–23 MB de los modelos de
  boosting), lo que facilita su integración en la herramienta interactiva de predicción.
- La disponibilidad de intervalos de predicción calibrados por conformal prediction,
  incluido el esquema de ancho adaptativo (CQR) desplegado en la herramienta.

## 3. Síntesis

El desarrollo evaluó tres metamodelos de boosting, una red neuronal multicapa y un baseline
lineal sobre una base de 1.04 millones de registros derivados de 20 000 simulaciones de
QUAL2K. La regresión lineal —con R² de 0.60 y todos sus supuestos violados— confirmó la
necesidad de modelos no lineales. Entre los metamodelos no lineales, la red neuronal
multicapa ofreció el mejor equilibrio entre exactitud (R² = 0.9993), sesgo casi nulo,
estabilidad, explicabilidad coherente con la física del sistema, cuantificación de
incertidumbre y eficiencia (≈2065× más rápida que QUAL2K), por lo que fue seleccionada como
metamodelo final para su despliegue en la herramienta interactiva de predicción.
