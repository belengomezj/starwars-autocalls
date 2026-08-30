# Inferencia

## Inferencia directa

El repositorio incluye un payload válido:

```bash
uv run starwars-autocalls predict --input sample_payload.json
```

El comando usa el mismo esquema, builder y artefacto que la API, sin HTTP.

## Inferencia por HTTP

Con FastAPI activo según la guía de [Instalación](installation.md#servicios-con-uv-run), están
disponibles estos endpoints:

| Método y ruta | Función | Respuesta ante fallo |
| --- | --- | --- |
| `GET /health` | Liveness y versión del paquete | No necesita artefacto |
| `GET /ready` | Verifica artefacto y checksum | `503` si no está listo |
| `GET /model-info` | Expone metadata reproducible | `503` si no puede cargarla |
| `POST /predict` | Predice duración e intervalo | `422` contrato inválido; `503` sin modelo |
| `POST /predict-batch` | Predice entre 1 y 1.000 RFQs | `422` si alguna RFQ es inválida; `503` sin modelo |

Prueba local:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @sample_payload.json
```

Para predecir varias RFQs en una única llamada, el endpoint batch recibe los mismos objetos dentro de
`requests`:

```json
{
  "requests": [
    {
      "product_type": "Kessel Run Snowball",
      "underlyings": "KYBR|CORL",
      "basket_type": "worst_of",
      "autocall_barrier_pct": 1.0,
      "protection_barrier_pct": 0.6,
      "no_call_period_months": 6,
      "observation_frequency": "1M",
      "quoted_implied_vol": 0.25,
      "notional_credits": 250000,
      "requested_date": "2024-06-28",
      "start_date": "2024-06-28",
      "end_date": "2030-06-28"
    }
  ]
}
```

La respuesta conserva el orden de entrada:

```json
{
  "predictions": [
    {
      "predicted_avg_duration_months": 34.2248,
      "model_version": "0.1.0",
      "model_name": "catboost_tuned__all_without_noise",
      "serving_strategy": "global",
      "prediction_interval_lower_months": 26.9685,
      "prediction_interval_upper_months": 41.4811,
      "interval_nominal_coverage": 0.9,
      "out_of_distribution": false,
      "warnings": [],
      "market_data_as_of": "2024-06-28",
      "max_market_data_age_days": 0
    }
  ]
}
```

El lote es atómico: si cualquiera de sus contratos o datos de mercado es inválido, no devuelve
resultados parciales. Streamlit utiliza este endpoint para procesar en una sola petición todas las
filas del JSON o CSV cargado.

Swagger está en `http://127.0.0.1:8000/docs`.

## Contrato de entrada

```json
{
  "product_type": "Kessel Run Snowball",
  "underlyings": "KYBR|CORL",
  "basket_type": "worst_of",
  "autocall_barrier_pct": 1.0,
  "protection_barrier_pct": 0.6,
  "no_call_period_months": 6,
  "observation_frequency": "1M",
  "quoted_implied_vol": 0.25,
  "notional_credits": 250000,
  "counterparty": "UNKNOWN",
  "trader_id": "UNKNOWN",
  "requested_date": "2024-06-28",
  "start_date": "2024-06-28",
  "end_date": "2030-06-28"
}
```

Las fechas de inicio y fin forman parte del contrato asumido. `single` exige un ticker y `worst_of` al
menos dos. Los tickers deben pertenecer a los 14 conocidos y disponer de volatilidad no posterior a la
fecha solicitada.

También se aceptan filas con el formato histórico completo de `rfqs.csv`. En ese caso, `rfq_id`,
`executed` y `avg_duration_months` se descartan antes de construir las features: son campos de
identificación o posteriores a la solicitud y no deben intervenir en la predicción. Cualquier otra
columna desconocida se rechaza con `422`, de modo que un error de esquema no pase inadvertido.

## Contrato de salida

```json
{
  "predicted_avg_duration_months": 34.2248,
  "model_version": "0.1.0",
  "model_name": "catboost_tuned__all_without_noise",
  "serving_strategy": "global",
  "prediction_interval_lower_months": 26.9685,
  "prediction_interval_upper_months": 41.4811,
  "interval_nominal_coverage": 0.9,
  "out_of_distribution": false,
  "warnings": [],
  "market_data_as_of": "2024-06-28",
  "max_market_data_age_days": 0
}
```

La predicción y el intervalo están acotados al contrato. `out_of_distribution` sólo cubre rangos
numéricos auditados; no es un detector completo de drift multivariante.

## Imagen Docker

```bash
docker build -t starwars-autocalls .
docker run --rm -p 8000:8000 starwars-autocalls
```

La imagen por defecto corresponde al target `api`: copia el artefacto versionado y no entrena durante
el arranque. El arranque mediante Compose está centralizado en
[Instalación](installation.md#servicios-con-docker-compose).
