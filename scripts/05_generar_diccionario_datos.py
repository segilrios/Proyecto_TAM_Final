#!/usr/bin/env python
"""Genera un diccionario simple de datos con filas, columnas y nombres."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for p in sorted((ROOT / "data").rglob("*.csv")):
    try:
        df = pd.read_csv(p, nrows=5)
        # Count lines cheaply
        with p.open("rb") as f:
            n = sum(1 for _ in f) - 1
        rows.append({
            "archivo": str(p.relative_to(ROOT)),
            "filas_aprox": max(n, 0),
            "columnas": len(df.columns),
            "nombres_columnas": ", ".join(map(str, df.columns[:30])) + (" ..." if len(df.columns) > 30 else "")
        })
    except Exception as e:
        rows.append({"archivo": str(p.relative_to(ROOT)), "filas_aprox": None, "columnas": None, "nombres_columnas": f"ERROR: {e}"})
out = pd.DataFrame(rows)
out_path = ROOT / "docs" / "data_dictionary_auto.csv"
out.to_csv(out_path, index=False)
print("Diccionario generado:", out_path)
print(out[["archivo", "filas_aprox", "columnas"]].to_string(index=False))
