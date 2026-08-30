| Ámbito | Modelo | Features | MAE base | MAE ajustado | Mejora | Trials | Ajuste (s) | Iteraciones / estimadores | Profundidad | Learning rate | Hojas | Mín. muestras | Bins | L1 | L2 | Subsample | Colsample | Random strength | Bagging temperature |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Global | `catboost_tuned__all_without_noise` | `all_without_noise` | 3.2684 | 3.0832 | 0.1852 | 22 | ≈303.1 | 737 | 6 | 0.07675 | — | — | — | — | 3.80249 | — | — | 1.45571 | 0.24831 |
| Global | `catboost_tuned__all_without_commercial` | `all_without_commercial` | 3.2811 | 3.1022 | 0.1789 | 21 | ≈314.8 | 882 | 6 | 0.04428 | — | — | — | — | 28.36003 | — | — | 1.42186 | 0.01727 |
| Global | `lightgbm_tuned__all_without_noise` | `all_without_noise` | 3.3215 | 3.2398 | 0.0818 | 25 | ≈203.6 | 884 | — | 0.03522 | 100 | 96 | — | 0.000001 | 0.000011 | 0.71377 | 0.74975 | — | — |
| Global | `hist_gradient_boosting_tuned__all_without_noise` | `all_without_noise` | 3.3385 | 3.2543 | 0.0842 | 25 | ≈183.2 | 554 | — | 0.03736 | 46 | 54 | 180 | — | 1.78367 | — | — | — | — |
| Global estable | `catboost_tuned__global_stable_no_sector` | `global_stable_no_sector` | 3.5220 | 3.3518 | 0.1702 | 19 | 29.248 | 803 | 8 | 0.07184 | — | — | — | — | 19.19892 | — | — | 0.06017 | 0.75850 |
| Global estable | `catboost_tuned__global_stable_tail` | `global_stable_tail` | 3.5271 | 3.3797 | 0.1474 | 20 | 18.730 | 797 | 7 | 0.02848 | — | — | — | — | 9.70912 | — | — | 0.77078 | 0.14677 |
| single | `catboost_tuned__single_core` | `single_core` | 3.6631 | 3.4613 | 0.2017 | 20 | ≈66.1 | 566 | 7 | 0.05718 | — | — | — | — | 0.98121 | — | — | 2.62566 | 0.60861 |
| single | `lightgbm_tuned__single_core` | `single_core` | 3.5192 | 3.5005 | 0.0187 | 20 | ≈29.8 | 541 | — | 0.04939 | 88 | 11 | — | 0.001529 | 0.000004 | 0.67343 | 0.83809 | — | — |
| worst_of | `catboost_tuned__worst_of_core` | `worst_of_core` | 3.0572 | 2.9409 | 0.1163 | 20 | ≈189.1 | 881 | 6 | 0.04561 | — | — | — | — | 29.71420 | — | — | 1.63380 | 0.00467 |
| worst_of | `hist_gradient_boosting_tuned__worst_of_core` | `worst_of_core` | 3.0686 | 3.0107 | 0.0579 | 20 | ≈79.4 | 278 | — | 0.05679 | 30 | 23 | 92 | — | 0.000034 | — | — | — | — |
