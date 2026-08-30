# Modelos

## Escalera de complejidad

La experimentación parte de referencias fáciles de defender y añade complejidad sólo si validation la
justifica:

1. Media y mediana global.
2. Medianas por producto, cesta, frecuencia y madurez.
3. Ridge como referencia lineal regularizada.
4. ExtraTrees y HistGradientBoosting como árboles de scikit-learn.
5. XGBoost y LightGBM como gradient boosting eficiente.
6. CatBoost con variables categóricas nativas.

Esta escalera permite distinguir la señal contractual básica de la mejora atribuible a interacciones no
lineales y composición de cesta.

## Familias y encoding

| Familia | Encoding | Ventaja en este problema | Riesgo principal |
| --- | --- | --- | --- |
| Medianas agrupadas | Reglas explícitas | Interpretabilidad y baseline de negocio | Poca granularidad |
| Ridge | One-hot | Referencia lineal estable | No captura interacciones complejas |
| ExtraTrees | One-hot | No linealidad sin boosting | Coste y dimensionalidad |
| HistGradientBoosting | Ordinal | Rápido y sólido en numéricas | Orden artificial en categóricas |
| XGBoost | Ordinal | Boosting flexible y eficiente | Encoding menos natural para categorías |
| LightGBM | Nativo | Rapidez y categorías nativas | Sensibilidad a hiperparámetros |
| CatBoost | Nativo | Buen tratamiento de categorías e interacciones | Mayor coste que LightGBM |

## Conjuntos de features

Los bloques de ablación separan contrato, producto, cesta, mercado, fecha y variables comerciales. Las
recetas principales son:

| Bloque | Nº features | Propósito |
| --- | ---: | --- |
| `contractual` | 18 | Términos que controlan posibilidad y calendario de autocall |
| `product` | 3 | Tipo de producto, cesta y bucket de madurez |
| `basket` | 137 | Composición, sectores, extremos y proxies `worst_of` |
| `market` | 22 | Volatilidad point-in-time y spreads |
| `all_without_commercial` | 183 | Receta amplia sin identidades comerciales |
| `all_without_noise` | 173 | Receta amplia tras ablación de diez candidatas inestables |
| `compact_core` | 27 | Alternativa mantenible de señal económica principal |
| `single_core` | 24 | Receta específica para un subyacente |
| `worst_of_core` | 166 | Receta amplia para cestas `worst_of` |

El modelo final utiliza `all_without_noise`: 159 variables numéricas y 14 categóricas. El número alto
procede principalmente de 14 indicadores de ticker y 91 indicadores de parejas. Estas variables
capturan composición histórica, pero no sustituyen una matriz de correlación. La configuración completa
del modelo ganador se encuentra en [`model_metadata.json`](/artifacts/model_metadata.json).

## Estrategia global frente a segmentada

Ambos segmentos tienen volumen suficiente. Por ello, se entrenaron modelos específicos para `single`
y `worst_of`, y se compararon con el modelo global bajo el mismo corte temporal.

La estrategia segmentada mejora ligeramente la media rolling, pero empeora la validation de 2022. Esta
mejora no justifica elegirla, ya que el router tendría que duplicar artefactos, calibración,
observabilidad y rutas de fallo sin evidencia de una mejora material. Por ello, se sirve el modelo
global.

## Modelo final

Tras seleccionar CatBoost como mejor familia en el benchmark, Optuna optimizó sus hiperparámetros. El
modelo ganador es `catboost_tuned__all_without_noise`, que utiliza el bloque `all_without_noise` y
categorías nativas. Sus hiperparámetros son 737 iteraciones, profundidad 6, learning rate 0,07675,
`l2_leaf_reg=3,8025`, `random_strength=1,4557` y `bagging_temperature=0,2483`. La configuración queda
registrada en [`model_metadata.json`](/artifacts/model_metadata.json).
