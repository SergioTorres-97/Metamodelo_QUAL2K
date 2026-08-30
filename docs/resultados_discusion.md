# Resultados y discusión

Este capítulo presenta los resultados del flujo metodológico y su discusión de forma
integrada: cada resultado se acompaña de su interpretación inmediata. Se organiza en seis
secciones que siguen las fases productoras de datos: análisis de sensibilidad global
(§1), base de datos de escenarios (§2), desempeño de los metamodelos (§3), justificación
de la no linealidad (§4), explicabilidad (§5) y eficiencia computacional y despliegue
(§6). Los valores numéricos provienen de `resultados/chicamocha_t1_sensibilidad/` y
`resultados/chicamocha_t1_metamodelo/`.

## 1. Análisis de sensibilidad global

El análisis de sensibilidad se ejecutó sobre 500 simulaciones LHS de QUAL2K, cuantificando
la asociación monotónica entre cada parámetro de entrada y la DBO carbonácea rápida
mediante el coeficiente de correlación de rangos de Spearman (SRCC). La Tabla 1 reporta los
parámetros más influyentes sobre la DBO.

**Tabla 1. Sensibilidad global (SRCC) sobre la DBO — parámetros dominantes.**

| Parámetro | SRCC | Proceso |
|---|---:|---|
| `dbo5_bypass` | +0.66 | Carga orgánica del bypass sin tratamiento |
| `caudal_bypass` | +0.50 | Caudal del bypass (masa descargada) |
| `caudal_la_vega` | −0.35 | Dilución por el afluente R. La Vega |
| `dbo5_veolia` | +0.20 | Carga del efluente tratado (PTAR principal) |
| `kaaa` | −0.14 | Reaireación superficial |
| `alpha_1` | +0.10 | Coeficiente hidráulico (velocidad, residencia) |

**Discusión.** El resultado es físicamente coherente: la DBO del tramo está gobernada por
el balance entre **aporte de carga orgánica** (signo positivo de `dbo5_bypass`,
`caudal_bypass` y `dbo5_veolia`) y **dilución** (signo negativo de `caudal_la_vega`). El
bypass sin tratamiento, mayor descarga del tramo, domina ampliamente la respuesta. Las
tasas cinéticas y los parámetros hidráulicos aparecen con influencia secundaria pero no
despreciable. Este análisis sustentó la reducción de dimensionalidad del problema: de los
36 parámetros iniciales se retuvieron **16 predictores** (por criterio estadístico de SRCC
y criterio físico de relevancia de procesos), a los que se añadió la coordenada
longitudinal `x_km` para dotar al metamodelo de resolución espacial. El mapa de calor de
sensibilidad espacial (`heatmap_srcc.png`) confirma además que estas influencias varían a
lo largo del tramo, intensificándose aguas abajo de cada descarga.

## 2. Base de datos de escenarios

Con las 16 variables seleccionadas se generó, mediante muestreo LHS y ejecución
automatizada de QUAL2K, una base de **30 000 simulaciones** independientes. Cada simulación
se discretizó en **52 puntos longitudinales** a lo largo de los 34.56 km del tramo,
produciendo un total de **1 560 000 registros**. La DBO simulada cubre un rango de 0.03 a
456.55 mg/L (media de 49.49 mg/L), lo que refleja escenarios que van desde agua limpia
hasta condiciones de alta carga orgánica. La partición para el entrenamiento se realizó
**por simulación** (`sim_id`) en fracciones 70/10/20 % (entrenamiento/validación/prueba),
para evitar fuga de información espacial entre puntos de un mismo escenario; el conjunto de
prueba quedó conformado por 6 000 simulaciones (312 000 registros).

## 3. Desempeño de los metamodelos

Se entrenaron seis metamodelos no lineales y un baseline de regresión lineal. Los modelos
de árboles y boosting se ajustaron sobre la transformación `log(1+y)` de la DBO y sus
hiperparámetros se optimizaron con Optuna (20 *trials*, protocolo de partición única con
detención temprana). La Tabla 2 resume el desempeño sobre el conjunto de prueba, en mg/L.

**Tabla 2. Desempeño predictivo sobre el conjunto de prueba (mg/L).**

| Modelo | R² | RMSE | MAE | Sesgo |
|---|---:|---:|---:|---:|
| Red neuronal (MLP) | 0.9991 | 1.87 | 1.25 | +0.03 |
| CatBoost | 0.9970 | 3.46 | 1.51 | −0.19 |
| LightGBM | 0.9947 | 4.59 | 2.23 | −0.23 |
| XGBoost | 0.9937 | 4.99 | 2.28 | −0.91 |
| Random Forest | 0.9308 | 16.57 | 8.51 | −3.44 |
| Regresión lineal (baseline) | 0.4944 | 44.80 | 33.90 | −0.10 |

**Discusión.** La red neuronal multicapa alcanzó el mejor desempeño, explicando el 99.91 %
de la varianza de la DBO con un RMSE de 1.87 mg/L sobre una variable de rango superior a
450 mg/L. Los modelos de boosting la siguieron de cerca (R² entre 0.994 y 0.997), mientras
que Random Forest quedó notablemente por debajo (R² = 0.931). Dos rasgos son relevantes.
Primero, el **sesgo**: la red neuronal es prácticamente insesgada (+0.03 mg/L), frente al
sesgo negativo de los modelos de árboles (hasta −3.44 mg/L en Random Forest). Los ensambles
de árboles predicen por promediado y no extrapolan hacia los picos extremos de DBO aguas
abajo del bypass, por lo que **subestiman** sistemáticamente; la pérdida ponderada por
magnitud de la red neuronal corrige ese efecto. Segundo, la **estabilidad**: la brecha
entre entrenamiento y prueba fue mínima en todos los casos (p. ej., RMSE de la red neuronal
de 1.73 a 1.87 mg/L), descartando sobreajuste y confirmando que la limitación de Random
Forest es de sesgo de modelo, no de generalización. Los gráficos observado-predicho
(`figuras_nn/obs_vs_pred.png`) muestran la nube de puntos alineada con la diagonal en todo
el rango de DBO.

## 4. Justificación de la no linealidad

El baseline de regresión lineal alcanzó un R² de apenas 0.49 (RMSE de 44.80 mg/L), dejando
sin explicar la mitad de la varianza. El diagnóstico de sus supuestos clásicos (Tabla 3)
confirma su inadecuación estructural.

**Tabla 3. Diagnóstico de supuestos del modelo lineal.**

| Supuesto | Prueba | Resultado | Estado |
|---|---|---|---|
| Linealidad | RESET (Ramsey) | F = 840.5 (p < 0.05) | Violado |
| Independencia | Durbin-Watson | DW = 0.18 | Violado |
| Homocedasticidad | Breusch-Pagan | LM = 2361.9 (p < 0.05) | Violado |
| Normalidad | Anderson-Darling / KS | A² = 5233.6 ; D = 0.042 | Violado |
| Multicolinealidad | VIF | VIF_máx = 1.00 | Aceptable |

**Discusión.** Todos los supuestos estructurales se incumplen: la prueba RESET confirma
estructura no lineal no capturada y el bajo Durbin-Watson refleja la autocorrelación entre
puntos longitudinales de una misma simulación. En contraste, la multicolinealidad es nula
(VIF ≈ 1.0), consecuencia directa del muestreo LHS, que genera predictores casi
ortogonales. El bajo desempeño del baseline y la violación sistemática de los supuestos
justifican empíricamente el uso de metamodelos no lineales.

## 5. Explicabilidad

La importancia de variables se calculó por permutación para cada metamodelo. La Tabla 4
presenta el ranking, representativo del patrón común a los seis modelos.

**Tabla 4. Importancia por permutación (predictores principales).**

| Predictor | Importancia relativa | Correspondencia con sensibilidad (§1) |
|---|---:|---|
| `x_km` | 1.47 | Gradiente espacial (no evaluado por SRCC) |
| `dbo5_bypass` | 0.47 | 1.º en SRCC (+0.66) |
| `caudal_bypass` | 0.29 | 2.º en SRCC (+0.50) |
| `caudal_la_vega` | 0.26 | 3.º en SRCC (−0.35) |
| `alpha_1` | 0.05 | 6.º en SRCC (+0.10) |
| `dbo5_veolia` | 0.03 | 4.º en SRCC (+0.20) |
| `dbo5_honda` | ≈ 0.00 | Despreciable en SRCC |

**Discusión.** Este es el resultado que articula la coherencia física del metamodelo. La
jerarquía de predictores hallada por permutación **reproduce la del análisis de
sensibilidad del modelo mecanicista**: los factores más influyentes son la carga del bypass
y la dilución de La Vega, en el mismo orden en ambos análisis, y los aportes despreciables
(como `dbo5_honda`) lo son en ambos. Además, esta jerarquía es **consistente entre las seis
familias de metamodelos** (árboles, boosting y red neuronal), lo que indica que refleja una
propiedad del sistema modelado y no un artefacto algorítmico. En consecuencia, el
metamodelo no solo reproduce el valor de la DBO, sino también su estructura causal, lo que
respalda su uso como sustituto interpretable de QUAL2K.

## 6. Eficiencia computacional y herramienta de predicción

La red neuronal seleccionada se comparó contra QUAL2K en tiempo de cómputo sobre 100
escenarios nuevos, con la inferencia forzada a CPU (Tabla 5).

**Tabla 5. Comparación de eficiencia computacional.**

| Enfoque | Tiempo por escenario | 50 000 corridas |
|---|---:|---:|
| QUAL2K (mecanicista) | 3.56 ± 0.61 s | 49.43 h |
| Metamodelo (red neuronal) | 0.0020 ± 0.0006 s | 1.7 min |

**Discusión.** El metamodelo resultó **≈1877 veces más rápido** que QUAL2K. Una campaña de
50 000 escenarios —cercana a dos días de cómputo con el modelo mecanicista— se resuelve con
el metamodelo en menos de dos minutos. Además de la predicción puntual, el modelo reporta
intervalos de predicción calibrados por *conformal prediction* (±3.59 mg/L al 95 % de
cobertura), sin supuestos distribucionales sobre los residuos. Esta capacidad se materializó
en una herramienta interactiva (Streamlit) que carga el modelo entrenado —sin ejecutar
QUAL2K— y permite explorar perfiles de DBO a lo largo del tramo con incertidumbre asociada,
cumpliendo el objetivo del trabajo: habilitar el análisis rápido de alternativas de gestión.

## 7. Selección del metamodelo final

Integrando desempeño, sesgo, estabilidad, explicabilidad y eficiencia, se seleccionó la
**red neuronal multicapa** como metamodelo final. Combina el mejor R² (0.9991) y el menor
sesgo (+0.03 mg/L), una importancia de variables coherente con la física del sistema,
inferencia casi instantánea (≈1877× más rápida que QUAL2K), intervalos de predicción
calibrados y una huella de serialización reducida (~22 KB, frente a 34–41 MB de los
ensambles de árboles), lo que facilita su despliegue en la herramienta interactiva.
