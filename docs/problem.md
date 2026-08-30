# Definición del problema y objetivos

La literatura sobre autocallables aborda habitualmente su valoración y dinámica mediante modelos de pricing y simulación. Sin embargo, **este proyecto no pretende valorar el producto ni reproducir su mecanismo de simulación**.

El target disponible, `avg_duration_months`, **se interpreta como la duración media obtenida mediante una simulación externa**. No se observan las trayectorias individuales ni se dispone de información suficiente sobre el motor que las genera. Por tanto, técnicas orientadas a reproducir directamente la dinámica del producto, como Monte Carlo, hazard o survival analysis, no son aplicables al problema observado.

En consecuencia, el problema se formula como una **regresión supervisada a nivel de RFQ**.

## Objetivos

Desarrollar las pautas marcadas por los creadores de la prueba técnica.

## Asunciones principales

Esta formulación se apoya en los siguientes supuestos:

- `avg_duration_months` se considera el target de referencia proporcionado por el simulador externo, aunque se desconoce su metodología interna.
- `start_date` y `end_date` son términos contractuales conocidos en el momento de cotización y pueden utilizarse para derivar la madurez nominal sin introducir información futura.
- Para cada RFQ sólo se utiliza información de mercado disponible hasta `requested_date`.
- **El modelo se entrena sobre RFQs ejecutadas** y, por tanto, estima duración condicionada a ejecución; **no modela la probabilidad de ejecución de una RFQ**.
- Se conservan **302 RFQs cuyo `avg_duration_months` supera la duración contractual derivada como `end_date - start_date`**. Dado que se desconoce la lógica del simulador, no existe evidencia suficiente para clasificarlas como errores. Además, su exclusión no modifica el rendimiento de los modelos.
