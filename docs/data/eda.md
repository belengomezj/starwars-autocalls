# Análisis exploratorio

## Alcance del análisis

El análisis comienza con perfiles automáticos de las fuentes en crudo, continúa con un EDA
complementario sobre relaciones específicas del problema y finaliza con una observación de los datos
bajo el split temporal, la validación *rolling* con ventana expansiva y las posibles segmentaciones
del modelo.

| Informe | Datos analizados | Enfoque | Documento |
| --- | --- | --- | --- |
| **Skrub** | Las tres tablas en crudo, sin filtros | Estructura, tipos, valores ausentes, cardinalidad y distribuciones | [RFQs](/reports/data_analysis/eda/supplemental/skrub_rfqs_table_report.html), [volatilidad](/reports/data_analysis/eda/supplemental/skrub_daily_volatility_table_report.html) y [referencia](/reports/data_analysis/eda/supplemental/skrub_underlyings_reference_table_report.html) |
| **Sweetviz** | RFQs con `executed=True` y `avg_duration_months` como target | Distribuciones y asociaciones automáticas con el target | [Informe Sweetviz](/reports/data_analysis/eda/supplemental/sweetviz_rfqs_report.html) |
| **EDA complementario** | RFQs, volatilidad, referencia y features unidas *point-in-time* | Relaciones que los informes automáticos no cubren | [EDA complementario](/reports/data_analysis/eda/eda_report.html) |
| **Auditoría de splits** | RFQs de desarrollo distribuidas entre train, validation y folds temporales expansivos | Cobertura temporal y categórica, drift y viabilidad de modelos segmentados | [Auditoría de splits](/reports/data_analysis/split_audit/split_audit.html) |

## Calidad de las fuentes

No se observan problemas críticos en ninguna de las fuentes.

La principal ausencia afecta a 11.204 valores de `avg_duration_months`, todos correspondientes a RFQs
con `executed=False`. No se imputan: el conjunto supervisado se limita a las RFQs ejecutadas, para las
que el target está disponible.

### Normalización

La construcción de features normaliza dos campos:

- `observation_frequency`: unifica sus representaciones en `daily`, `1m`, `2m`, `3m`, `6m` o `12m`,
  y las convierte a un intervalo numérico en meses.
- `underlyings`: separa los componentes delimitados por `|`, elimina espacios y genera una composición
  canónica independiente de su orden.

`product_type` y `basket_type` conservan sus categorías originales.

## Hallazgos principales

### Target y estructura del producto

Las 11.385 RFQs de desarrollo presentan una duración media de 39,83 meses y una mediana de 35,37
meses. Los productos `worst_of` duran más en promedio que los productos `single` —45,46 frente a
28,68 meses— y `Wretched Hive Digital` presenta la duración media más elevada, con 59,71 meses.

Sweetviz identifica las asociaciones descriptivas más fuertes del target con `product_type`,
`basket_type` y `observation_frequency`. El detalle se encuentra en el
[informe Sweetviz](/reports/data_analysis/eda/supplemental/sweetviz_rfqs_report.html) y en la
[auditoría de splits](/reports/data_analysis/split_audit/split_audit.html).

### Coherencia con la madurez nominal

En 302 RFQs, el 2,19 % de la muestra supervisada, `avg_duration_months` supera la madurez contractual
derivada de `end_date - start_date`. Sólo 28 casos la superan en más de seis meses y dos en más de doce
meses.

No existe información suficiente sobre el simulador externo para clasificar estos casos como errores,
por lo que se conservan. El gráfico frente a la madurez y el detalle por umbral se incluyen en el
[EDA complementario](/reports/data_analysis/eda/eda_report.html), en la sección «Coherencia con la
madurez nominal». Allí se muestran el gráfico de `nominal_maturity_months` frente a
`avg_duration_months` y la tabla de observaciones por meses de exceso, con el número de RFQs y su
porcentaje sobre la muestra supervisada.

### Relaciones contractuales y de mercado

La madurez nominal presenta la correlación numérica más alta con el target (`0,52`), seguida por la
madurez disponible después del periodo de *no-call* (`0,52`). También destacan las variables que
representan presión y dispersión de riesgo en productos `worst_of`.

La volatilidad realizada se incorpora utilizando únicamente la última observación disponible hasta
`requested_date`. Las relaciones temporales, la composición de las cestas y la comparación entre
volatilidad implícita y realizada se desarrollan en el
[EDA complementario](/reports/data_analysis/eda/eda_report.html).

### Cobertura y estabilidad temporal

Validation no contiene productos, tipos de cesta ni subyacentes individuales ausentes en train. Sin
embargo, aparecen 112 composiciones completas de cesta nuevas, que afectan a 119 RFQs, el 7,54 % de
validation.

Los controles de drift numérico y categórico no detectan cambios relevantes entre train y validation.
La estabilidad también se comprueba mediante cinco folds temporales con ventana expansiva. La
distribución por split, la cobertura categórica y la viabilidad de modelos segmentados se desarrollan
en la [auditoría de splits](/reports/data_analysis/split_audit/split_audit.html).

## Sesgo de selección por ejecución

Se comparan las variables observadas entre RFQs ejecutadas y no ejecutadas mediante
Kolmogorov–Smirnov para las numéricas y chi-cuadrado para las categóricas. Tras corregir los diez
contrastes mediante Bonferroni, no se detectan diferencias robustas.

`notional_credits` sólo resulta significativo sin corrección (`p=0,013`; `KS=0,020`). El análisis no
permite descartar sesgo asociado a variables no observadas ni al propio target, que no está disponible
para `executed=False`. El detalle se encuentra en el
[EDA complementario](/reports/data_analysis/eda/eda_report.html).

## Reproducción

El EDA complementario se genera con:

```bash
uv run starwars-autocalls eda
```

Para incluir los informes automáticos de Skrub y Sweetviz:

```bash
uv run starwars-autocalls eda --library-reports
```

La auditoría temporal y de splits se genera por separado:

```bash
uv run starwars-autocalls split-audit
```
