# Scripts

Esta carpeta contiene código ejecutable para validar, reproducir e inspeccionar el proyecto.

## Scripts principales

```bash
python scripts/00_check_project.py
```

Valida la estructura mínima del proyecto.

```bash
python scripts/02_quick_test.py
```

Ejecuta una prueba rápida sin reentrenar: carga métricas, predicciones y simula Buy & Hold vs modelo.

```bash
python scripts/03_model_inference_demo.py
```

Carga `best_model_bundle.joblib` e inspecciona el artefacto final.

```bash
python scripts/04_run_dashboard_server.py
```

Levanta un servidor local para abrir el dashboard.

```bash
python scripts/05_generar_diccionario_datos.py
```

Genera un diccionario automático de CSVs.

## Pipeline completo

```bash
RUN_PROFILE=quick python scripts/01_pipeline_completo.py
RUN_PROFILE=full  python scripts/01_pipeline_completo.py
```

`01_pipeline_completo.py` fue exportado desde el cuaderno final.
Para ejecución interactiva se recomienda usar `notebooks/proyecto_oro_colombia_final.ipynb`.
