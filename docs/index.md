# Star Wars Autocalls

Star Wars Autocalls estima `avg_duration_months`, la duración media de un producto
autocallable hasta cancelación anticipada o vencimiento. La solución cubre validación de datos,
integración point-in-time, ingeniería de variables, experimentación temporal, seguimiento con MLflow,
entrenamiento reproducible y una API de inferencia.

![Mesa de estructuración cuantitativa del Banco Imperial](assets/images/imperial-quant-desk.png)

## Resultado vigente

| Elemento | Resultado |
| --- | --- |
| Estrategia servida | Modelo global |
| Modelo | `catboost_tuned__all_without_noise` |
| Desarrollo | 11.385 RFQs ejecutadas de 2016 a 2022 |
| Validación | 1.578 RFQs ejecutadas en 2022 |
| MAE de validación | 3,0832 meses |
| RMSE de validación | 4,4367 meses |
| R² de validación | 0,9577 |
| Mediana del error absoluto de validación | 2,0274 meses |
| Conjunto de prueba final | 2.411 RFQs ejecutadas de 2023 a junio de 2024 |
| MAE de test | 4,0418 meses |
| RMSE de test | 5,6649 meses |
| R² de test | 0,9373 |
| Mediana del error absoluto de test | 2,7324 meses |
| Artefacto | `artifacts/model.joblib` con checksum SHA-256 |


## Recorrido recomendado

1. Consulta la [definición del problema](problem.md) y las asunciones.
2. Revisa el [dataset](data/overview.md), el [EDA](data/eda.md) y el
   [preprocesamiento](data/preprocessing.md).
3. Lee el protocolo de [evaluación](methodology/evaluation.md) y
   [validación](methodology/validation.md).
4. Examina los [experimentos](experiments/model-comparison.md) y la selección del
   [modelo final](results/final-model.md).
5. Reproduce el flujo desde [instalación](usage/installation.md) y prueba la
   [inferencia](usage/inference.md).
