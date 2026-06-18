# Oro Colombia Regime ML

Repositorio final depurado para el proyecto de predicción direccional del oro en Colombia usando regímenes macrofinancieros, enfermedad holandesa y modelos de aprendizaje de máquinas.

## Resumen

El proyecto analiza el oro colombiano como una serie temporal influida por:

- variables locales de Colombia,
- contexto global de mercados,
- commodities,
- TRM,
- política monetaria,
- enfermedad holandesa,
- periodos presidenciales,
- regímenes macrofinancieros UMAP/KMeans.

La tarea principal es predecir si el oro sube o no sube en el horizonte `h21`, aproximadamente un mes operativo.

## Resultado principal

El mejor modelo reportado fue:

```text
HistGradientBoostingClassifier
feature_set: colombia_plus_global_full
horizon: h21
weight_strategy: balanced
regime_version: umap_k6_7_structural
```

Métricas aproximadas:

```text
Balanced accuracy media: 87.34 %
Accuracy media: 88.77 %
ROC-AUC media: 0.948
```

## Estructura

```text
.
├── README.md
├── requirements.txt
├── MANIFEST.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── scripts/
├── dashboard/
├── outputs/
├── reports/
└── docs/
```

## Uso rápido

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Validar estructura:

```bash
python scripts/00_check_project.py
```

Ejecutar prueba rápida sin reentrenar:

```bash
python scripts/02_quick_test.py
```

Abrir dashboard:

```bash
python scripts/04_run_dashboard_server.py
```

## Cuaderno final

Solo se incluye un cuaderno final depurado:

```text
notebooks/proyecto_oro_colombia_final.ipynb
```

El cuaderno tiene modo rápido y modo completo:

```python
RUN_PROFILE = "quick"  # prueba rápida
RUN_PROFILE = "full"   # corrida completa
```

## Scripts

El código principal exportado está en:

```text
scripts/01_pipeline_completo.py
```

Además hay scripts para validación, inspección del modelo, prueba rápida y servidor del dashboard.

## Dashboard

El dashboard final está en:

```text
dashboard/dashboard_interactivo_final.html
```

Incluye exploración de modelos, regímenes, errores, explicabilidad, auditoría y simulación.

## Datos

- `data/raw/`: bases fuente/homogeneizadas.
- `data/processed/`: resultados del pipeline.
- `models/`: modelo final entrenado.
- `outputs/figures/`: figuras finales.

## Advertencia

Este proyecto es académico. El simulador de inversión es exploratorio y no constituye recomendación financiera.
