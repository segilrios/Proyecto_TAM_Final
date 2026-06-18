# Datos del proyecto

## raw/

Contiene las bases de entrada o bases fuente ya homogeneizadas usadas para el proyecto:

- `base_colombia_normalizada.csv`: base Colombia tratada.
- `global_market_context_homogenized.csv`: contexto global homogeneizado.
- `dutch_disease.csv`: enfermedad holandesa.
- `presidential_periods.csv`: periodos presidenciales.
- `gold_series.csv`: serie del oro usada como objetivo.

## processed/

Contiene resultados derivados del pipeline:

- `predictions/`: predicciones y salidas por modelo.
- `metrics/`: métricas de clasificación, magnitud y segmentos.
- `regimes/`: UMAP, KMeans, perfiles de clusters y biplots.
- `catalogs/`: catálogo de features y configuraciones.
- `audit/`: auditoría, cobertura y tratamiento de datos.
- `strategies/`: simulaciones y backtests exploratorios.

No se duplican modelos ni figuras en esta carpeta. Los modelos están en `models/` y las figuras en `outputs/figures/`.
