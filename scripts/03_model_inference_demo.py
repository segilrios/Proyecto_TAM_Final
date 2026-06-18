#!/usr/bin/env python
"""Inspección del artefacto best_model_bundle.joblib.

Este script no reentrena. Solo carga el bundle final, muestra sus llaves
principales y, si es posible, genera una inferencia de ejemplo con la última
fila disponible de la base procesada.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models/best_model_bundle.joblib"
CFG = ROOT / "models/best_model_config.json"
RAW = ROOT / "data/raw/base_colombia_normalizada.csv"

bundle = joblib.load(BUNDLE)
cfg = json.loads(CFG.read_text(encoding="utf-8"))
print("Configuración:")
for k, v in cfg.items():
    if isinstance(v, (str, int, float, bool)) or v is None:
        print(f" - {k}: {v}")

print("\nLlaves del bundle:", list(bundle.keys()) if isinstance(bundle, dict) else type(bundle))

if not isinstance(bundle, dict):
    raise SystemExit("El bundle no es dict; no se puede inspeccionar de forma genérica.")

model = bundle.get("model") or bundle.get("best_model") or bundle.get("estimator")
features = bundle.get("features") or bundle.get("feature_names") or bundle.get("selected_features")
imputer = bundle.get("imputer") or bundle.get("preprocessor")

print("Modelo:", type(model).__name__ if model is not None else "No encontrado")
print("Número de features:", len(features) if features is not None else "No encontrado")

if model is not None and features is not None and RAW.exists():
    df = pd.read_csv(RAW)
    available = [f for f in features if f in df.columns]
    if len(available) == len(features):
        x = df[available].tail(1).apply(pd.to_numeric, errors="coerce")
        if imputer is not None:
            x_in = imputer.transform(x)
        else:
            x_in = x.fillna(x.median(numeric_only=True)).values
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_in)[0, 1]
            print("Probabilidad de subida ejemplo:", round(float(proba), 4))
        else:
            pred = model.predict(x_in)[0]
            print("Predicción ejemplo:", pred)
    else:
        print("No se pudo hacer inferencia directa: faltan columnas respecto al bundle.")
        print("Features disponibles:", len(available), "/", len(features))
