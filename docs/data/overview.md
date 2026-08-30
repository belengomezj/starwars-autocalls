# Dataset

## Fuentes

| Fichero | Grano | Filas | Uso |
| --- | --- | ---: | --- |
| `data/raw/rfqs.csv` | Una RFQ | 25.000 | Contrato, producto, cesta, datos comerciales y target |
| `data/raw/daily_volatility.csv` | Fecha y subyacente | 44.744 | Volatilidad realizada point-in-time |
| `data/raw/underlyings_reference.csv` | Un subyacente | 14 | Sector y volatilidad estructural |


## Esquema de RFQs

| Grupo | Columnas | Disponibilidad en inferencia |
| --- | --- | --- |
| Identidad | `rfq_id` | Generada por el servicio |
| Producto y cesta | `product_type`, `basket_type`, `underlyings` | Sí |
| Contrato | `autocall_barrier_pct`, `protection_barrier_pct`, `no_call_period_months`, `observation_frequency`, `start_date`, `end_date` | Sí |
| Mercado cotizado | `quoted_implied_vol` | Sí |
| Comercial | `notional_credits`, `counterparty`, `trader_id` | Sí, aunque contraparte y trader no se usan en el modelo final |
| Tiempo | `requested_date` | Sí |
| Resultado | `executed`, `avg_duration_months` | No son features |


## Poblaciones

| Población | Criterio | Filas |
| --- | --- | ---: |
| Total | Todas las RFQs | 25.000 |
| Supervisada completa | `executed=True` y target informado | 13.796 |
| Desarrollo | Supervisada, `requested_date <= 2022-12-31` | 11.385 |
| Test final | Supervisada, `requested_date >= 2023-01-01` | 2.411 |


## Calidad e invariantes

La validación comprueba tipos, rangos, unicidad de referencia y coherencia entre fuentes. Además:

- `single` debe contener exactamente un subyacente.
- `worst_of` debe contener al menos dos.
- Una cesta no admite tickers duplicados.
- `end_date` debe ser posterior a `start_date`.
- `requested_date` no puede ser posterior al inicio contractual en la API.
- El periodo de no-call no puede exceder la madurez.
- Cada ticker debe existir en referencia y disponer de mercado anterior a la RFQ.

## Riesgo de selección muestral

Se compararon las RFQs ejecutadas y no ejecutadas mediante las variables disponibles. No se encontraron diferencias estadísticamente sólidas después de corregir por haber realizado muchas comparaciones. Esto reduce la evidencia de sesgo observable, pero no demuestra que no exista.
