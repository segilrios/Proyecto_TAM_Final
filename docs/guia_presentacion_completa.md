# Guía de presentación y defensa

## Mensaje central

El proyecto no intenta predecir el oro como una serie aislada. El objetivo es explicar y anticipar la dirección del oro colombiano combinando:

- contexto local,
- contexto global,
- enfermedad holandesa,
- periodos presidenciales,
- regímenes UMAP/KMeans,
- modelos supervisados.

## Explicación rápida

1. La base tiene fechas diarias, pero el oro tiene frecuencia efectiva cercana a mensual.
2. Por eso se usa el horizonte `h21`.
3. Se construyen regímenes `ABS` y `CHG`.
4. Se entrena una familia de modelos tabulares.
5. Se valida con walk-forward.
6. Se selecciona el mejor modelo por balanced accuracy.
7. Se interpreta con variables, segmentos, errores y simulación.

## Frase recomendada

> El aporte principal no es solo obtener una predicción, sino construir una arquitectura interpretable donde el oro colombiano se entiende como una función de régimen macrofinanciero, contexto global, variables locales y shocks de corto plazo.

## Preguntas frecuentes

### ¿Qué es ABS?

`ABS` es la vista de valores absolutos o niveles. Captura el estado macroeconómico.

### ¿Qué es CHG?

`CHG` es la vista de cambios. Captura shocks y transiciones.

### ¿Por qué h21?

Porque el oro no cambia de forma efectiva todos los días y 21 observaciones aproximan un mes operativo.

### ¿Por qué balanced accuracy?

Porque evalúa de forma equilibrada la clase `sube` y la clase `no sube`.

### ¿Por qué no red neuronal profunda?

Porque el problema es tabular, con ingeniería de variables y necesidad de interpretabilidad. Los modelos de boosting son más adecuados en este escenario.
