#!/usr/bin/env python
"""Prueba rápida del proyecto sin reentrenar.

Carga métricas, predicciones y modelo final; ejecuta una simulación básica
Buy & Hold vs señal del modelo long/cash.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/processed/metrics/classification_summary.csv"
PREDS = ROOT / "data/processed/predictions/predictions.csv"
MODEL_CFG = ROOT / "models/best_model_config.json"

def fmt_pct(x): return f"{100*x:,.2f}%"
def fmt_money(x): return f"${x:,.0f}"

summary = pd.read_csv(SUMMARY)
preds = pd.read_csv(PREDS)
cfg = json.loads(MODEL_CFG.read_text(encoding="utf-8"))

print("\n=== RESUMEN DEL MODELO FINAL ===")
best = summary.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
for k in ["model", "feature_set", "horizon", "train_window_years", "weight_strategy"]:
    if k in best:
        print(f"{k}: {best[k]}")
for k in ["mean_balanced_accuracy", "mean_accuracy", "mean_roc_auc", "mean_threshold"]:
    if k in best:
        print(f"{k}: {best[k]:.4f}")

print("\n=== SIMULACIÓN RÁPIDA ===")
p = preds.copy()
p["fecha"] = pd.to_datetime(p["fecha"])
price_col = "precio_oro"
if price_col not in p.columns:
    raise SystemExit("No se encontró columna precio_oro en predictions.csv")

p = p.dropna(subset=[price_col]).sort_values("fecha").reset_index(drop=True)
p["ret"] = p[price_col].pct_change().fillna(0)

if "pred_up_calibrated" in p.columns:
    signal = p["pred_up_calibrated"].shift(1).fillna(0).astype(float)
elif "proba_up" in p.columns and "threshold" in p.columns:
    signal = (p["proba_up"].shift(1) >= p["threshold"].shift(1)).fillna(False).astype(float)
else:
    signal = pd.Series(1, index=p.index)

cap0 = 1_000_000
buy_hold = cap0 * (1 + p["ret"]).cumprod()
model_lc = cap0 * (1 + signal * p["ret"]).cumprod()

def max_dd(curve):
    peak = curve.cummax()
    return (curve / peak - 1).min()

print("Rango:", p["fecha"].min().date(), "→", p["fecha"].max().date())
print("Capital inicial:", fmt_money(cap0))
print("Buy & hold final:", fmt_money(buy_hold.iloc[-1]), "| retorno:", fmt_pct(buy_hold.iloc[-1]/cap0 - 1), "| max DD:", fmt_pct(max_dd(buy_hold)))
print("Modelo long/cash final:", fmt_money(model_lc.iloc[-1]), "| retorno:", fmt_pct(model_lc.iloc[-1]/cap0 - 1), "| max DD:", fmt_pct(max_dd(model_lc)))

print("\nOK: prueba rápida completada.")
