#!/usr/bin/env python
"""Valida que la estructura mínima del proyecto esté completa."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "notebooks/proyecto_oro_colombia_final.ipynb",
    "dashboard/dashboard_interactivo_final.html",
    "models/best_model_bundle.joblib",
    "models/best_model_config.json",
    "data/raw/base_colombia_normalizada.csv",
    "data/raw/global_market_context_homogenized.csv",
    "data/raw/dutch_disease.csv",
    "data/processed/predictions/predictions.csv",
    "data/processed/metrics/classification_summary.csv",
]
missing = [p for p in REQUIRED if not (ROOT / p).exists()]
if missing:
    print("Faltan archivos:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)

cfg = json.loads((ROOT / "models/best_model_config.json").read_text(encoding="utf-8"))
summary = pd.read_csv(ROOT / "data/processed/metrics/classification_summary.csv")
preds = pd.read_csv(ROOT / "data/processed/predictions/predictions.csv")

print("OK: estructura mínima completa.")
print("Mejor modelo:", cfg.get("model", cfg.get("model_name", "N/D")))
print("Filas classification_summary:", len(summary))
print("Filas predictions:", len(preds))
print("Rango predicciones:", preds["fecha"].min(), "→", preds["fecha"].max())
