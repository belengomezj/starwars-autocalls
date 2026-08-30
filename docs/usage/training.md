# Entrenamiento

## Flujo recomendado

Los comandos muestran progreso Rich por modelo, fold o trial, junto con MAE y tiempo de ajuste. Deben
ejecutarse desde la raíz.

Para ejecutar el recorrido completo, excepto los modelos baseline:

```bash
uv run starwars-autocalls full-suite
```

Para revisar sus etapas sin ejecutarlas:

```bash
uv run starwars-autocalls full-suite --dry-run
```

La suite es costosa y se detiene en el primer fallo.

### 1. Validar y auditar

```bash
uv run starwars-autocalls validate-data
uv run starwars-autocalls eda
uv run starwars-autocalls split-audit
uv run starwars-autocalls feature-audit
```

Estas operaciones no entrenan un modelo servido. Generan HTML, CSV y JSON bajo `reports/`.

### 2. Comparar candidatos

```bash
uv run starwars-autocalls benchmark
uv run starwars-autocalls rolling-benchmark
uv run starwars-autocalls segmented-benchmark
uv run starwars-autocalls segmented-rolling-benchmark
```

Cada benchmark entrena por definición. Rolling vuelve a ajustar por fold para preservar el corte
temporal. Puede limitarse el coste con nombres separados por comas:

```bash
uv run starwars-autocalls benchmark \
  --models catboost_native__all_without_noise,lightgbm_native__all_without_noise
```

### 3. Ajustar hiperparámetros

```bash
uv run starwars-autocalls tune \
  --n-trials 25 \
  --top-n 4 \
  --timeout-seconds-per-model 300

uv run starwars-autocalls tune-segmented \
  --n-trials 20 \
  --timeout-seconds-per-model 300
```

Para la rama de estabilidad y la selección segmentada:

```bash
uv run starwars-autocalls global-stable-experiment --n-trials 20 --top-n 2
uv run starwars-autocalls select-segmented-models \
  --n-trials 20 \
  --top-n-per-segment 2 \
  --ranking-metric validation_mae
```

`tune-segmented` ajusta los candidatos indicados. `select-segmented-models` lee primero el benchmark,
selecciona los mejores por segmento y ajusta sólo esos; se solapan si se ejecuta el recorrido exhaustivo.

### 4. Comparar estrategias

```bash
uv run starwars-autocalls compare-serving-strategies
uv run starwars-autocalls experiment-summary
```

El primer comando reentrena global, segmentos y folds porque necesita residuos emparejados bajo un
protocolo homogéneo. No accede a test. `experiment-summary` sólo consolida CSV existentes y nunca
entrena.

### 5. Entrenar el artefacto

Para reentrenar el modelo final seleccionado:

```bash
uv run starwars-autocalls train
```

La especificación y los hiperparámetros están congelados en `config/final_model.json`.

Para repetir la selección experimental de estrategia:

```bash
uv run starwars-autocalls train --use-strategy-selection
```

Para consumir directamente el mejor tuning global:

```bash
uv run starwars-autocalls train --use-tuned-best
```

Las opciones experimentales requieren su fichero de selección previo. No deben concatenarse dos comandos sin un
separador de shell. Por ejemplo, ejecuta primero `tune` y después `train --use-tuned-best` en una línea
nueva.

También puede fijarse una especificación exacta:

```bash
uv run starwars-autocalls train \
  --model-name hist_gradient_boosting__all_without_noise
```

`train` ajusta una vez sobre train más validation, abre test tras congelar la decisión y escribe:

```text
artifacts/model.joblib
artifacts/model.joblib.sha256
artifacts/model_metadata.json
reports/model_evaluation/final_test_metrics.csv
reports/model_evaluation/operational_metrics.json
```

### 6. Evaluar sin reentrenar

```bash
uv run starwars-autocalls evaluate
```

`evaluate` carga el artefacto actual y recalcula test. No selecciona ni modifica el modelo.

## Qué comandos reentrenan

| Comando | ¿Entrena? | Reutilización |
| --- | --- | --- |
| `benchmark` | Sí | Escribe ranking y predicciones de validation |
| `rolling-benchmark` | Sí, por fold | Puede leer nombres del benchmark |
| `segmented-benchmark` | Sí | Escribe ranking por segmento |
| `segmented-rolling-benchmark` | Sí, por segmento y fold | No reutiliza modelos fitted |
| `tune`, `tune-segmented` | Sí, por trial | Lee rankings existentes si no se pasan modelos |
| `error-analysis` | No | Consume predicciones persistidas del benchmark |
| `compare-serving-strategies` | Sí | Necesita residuos y folds comparables |
| `explain` | Sí | Refit diagnóstico para SHAP; no cambia el artefacto |
| `experiment-summary` | No | Consolida resultados |
| `train` | Sí | Refit final sobre desarrollo |
| `evaluate`, `predict`, `serve` | No | Cargan `model.joblib` |

Los experimentos se registran bajo `starwars-autocalls`. El arranque de MLflow, local o con Compose, se
describe una sola vez en [Instalación](installation.md#servicios-con-uv-run).
