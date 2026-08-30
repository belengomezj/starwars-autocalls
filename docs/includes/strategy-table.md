| Estrategia | Modelo | Features | MAE validación | MAE rolling | Máximo rolling | Servida |
| --- | --- | --- | --- | --- | --- | --- |
| global | `catboost_tuned__all_without_noise` | `all_without_noise` | 3.0832 | 3.9886 | 5.8141 | Sí |
| segmented_by_basket_type | `single:catboost_tuned__single_core + worst_of:catboost_tuned__worst_of_core` | `single:single_core + worst_of:worst_of_core` | 3.1170 | 3.9707 | 5.7357 | No |
