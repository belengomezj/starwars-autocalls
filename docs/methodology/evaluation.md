# Evaluación

## Métrica principal

La métrica principal para seleccionar el modelo es el MAE (error absoluto medio). Aunque también se tienen en cuenta métricas secundarias que aportan información adicional sobre los errores extremos, la variabilidad y el rendimiento
típico del modelo.

## Métricas secundarias

| Métrica | Qué aporta | Riesgo de interpretación |
| --- | --- | --- |
| RMSE | Penaliza con mayor fuerza errores grandes | Puede quedar dominado por la cola |
| R² | Proporción de varianza explicada frente a la media | Un valor alto no garantiza errores pequeños en meses |
| MedAE | Error típico robusto | Oculta el comportamiento de la cola |
| MAE por segmento | Equidad de rendimiento entre cohortes | Segmentos pequeños tienen mayor incertidumbre |
| Tiempo de ajuste y predicción | Coste operativo | Depende del hardware y la carga local |

## Criterio de promoción

Los candidatos del holdout de desarrollo se ordenan por `validation_mae`. Los candidatos rolling se
ordenan por `rolling_mae_mean`, conservando desviación y máximo como medidas de estabilidad. Las
métricas de test no participan en rankings, selección de features, tuning ni elección de estrategia.

La comparación global frente a segmentada utiliza la diferencia emparejada
`delta = MAE global − MAE segmentada`.

- Si delta es positivo, el modelo segmentado tiene menor error.
- Si es negativo, el global funciona mejor.

Se estima un intervalo percentil al 95 % mediante
5.000 remuestreos por bloques de trimestre sobre los errores absolutos de validation. Sólo se acepta
una estrategia más compleja (segmentada) si la diferencia es estadísticamente distinguible y alcanza al menos 0,25
meses de mejora práctica.

## Incertidumbre predictiva

El artefacto guarda un intervalo de predicción de ±7,2563 meses, calculado con los errores de validation. Cubre el 90,05 % de los casos de calibración, pero debe revisarse y ajustarse con datos futuros.

## Evaluación operativa

Además de la calidad estadística se registran:

- tamaño del artefacto: 2,4 MB;
- ajuste final sobre desarrollo: 12,443 segundos;
- predicción de 2.411 filas de test: 0,112 segundos;
- throughput batch observado: 21.549 filas por segundo;
- checksum, hashes de datos, `uv.lock`, dependencias y commit.

Estas medidas son comparables dentro de la misma máquina y ejecución; no son un benchmark de
producción bajo concurrencia.
