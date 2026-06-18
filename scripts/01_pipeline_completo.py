"""Pipeline completo exportado desde notebooks/proyecto_oro_colombia_final.ipynb.

Uso recomendado:
  RUN_PROFILE=quick python scripts/01_pipeline_completo.py
  RUN_PROFILE=full  python scripts/01_pipeline_completo.py

Nota: el cuaderno sigue siendo la fuente principal para ejecución interactiva.
"""


# %% [markdown]
# # Proyecto final · Oro Colombia, regímenes macrofinancieros y ML


# %% [markdown]
# ## Cómo usar este cuaderno en Kaggle


# %% [markdown]
# ## 0. Configuración inicial


# %% Cell 3
# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

import os, sys, json, math, warnings, zipfile, subprocess, re
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ----------------------------
# Modo de ejecución
# ----------------------------
RUN_PROFILE = os.getenv("RUN_PROFILE", "quick").lower()  # "quick" o "full"
FAST_MODE = (RUN_PROFILE == "quick")          # True = corrida corta de prueba
USE_YFINANCE = True        # Descarga contexto global si no existe
USE_UMAP = True            # Requiere umap-learn
USE_XGBOOST = True         # Requiere xgboost
SAVE_MODEL = True          # Guarda el mejor modelo entrenado al final
XGB_USE_GPU = False        # Cambiar a True solo si Kaggle tiene GPU activa y XGBoost soporta CUDA
RANDOM_STATE = 42

# ----------------------------
# Regímenes económicos
# ----------------------------
# Corridas profundas previas indicaron que 6/7 clusters eran los más útiles
# para interpretar régimen económico y enfermedad holandesa.
REGIME_VERSION = "umap_k6_7_structural"
REGIME_CLUSTER_GRID_ABS = [6, 7]
REGIME_CLUSTER_GRID_CHG = [6, 7]
REGIME_SELECTION_SCORE = "structural_silhouette"
FORCE_REBUILD_REGIMES_6_7 = True

# ----------------------------
# Cache / reanudación
# ----------------------------
# Si subes un dashboard_data_checkpoint.zip anterior como input, puede reutilizar
# datos tratados y contexto global, pero NO mezcla resultados de modelos de otro régimen.
USE_PREVIOUS_OUTPUTS_AS_CACHE = False
CLEAR_OUTPUT_DIR = True

# ----------------------------
# Criterios de búsqueda y validación
# ----------------------------
MODEL_SELECTION_METRIC = "mean_balanced_accuracy"
THRESHOLD_METRIC = "balanced_accuracy"
THRESHOLD_GRID = None  # None = grilla fina automática entre cuantiles de probabilidad
MAX_FEATURES_COMPACT = 1500
MAX_FEATURES_FULL = 2200

# Estrategias de ponderación. Se comparan dentro de la misma validación temporal.
WEIGHT_STRATEGIES = ["none", "balanced", "cluster_x4", "macrofull_x15", "recency_4y"]

# ----------------------------
# Horizonte principal
# ----------------------------
HORIZONS = [21]
MAIN_HORIZON = 21

# ----------------------------
# Directorios
# ----------------------------
ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT_CANDIDATES = [
    Path("/kaggle/input"),
    Path("/mnt/data"),
    ROOT,
]
OUT = ROOT / "dashboard_data"
FIG = OUT / "figures"
MODELS = OUT / "models"

# ============================================================
# LIMPIEZA DESDE CERO
# ============================================================
if CLEAR_OUTPUT_DIR and OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print("ROOT:", ROOT)
print("OUT:", OUT)

# ============================================================
# CACHE: extraer outputs previos si existen como input
# ============================================================
if USE_PREVIOUS_OUTPUTS_AS_CACHE:
    cache_zips = []
    for base in [Path("/kaggle/input"), Path("/mnt/data"), ROOT]:
        if base.exists():
            cache_zips += list(base.rglob("dashboard_data_checkpoint.zip"))
            cache_zips += list(base.rglob("dashboard_data_bundle.zip"))
            cache_zips += list(base.rglob("dashboard_data.zip"))
    cache_zips = [p for p in cache_zips if p.is_file()]
    if cache_zips:
        cache_zip = sorted(cache_zips, key=lambda p: p.stat().st_size, reverse=True)[0]
        print("Cache detectado:", cache_zip)
        try:
            with zipfile.ZipFile(cache_zip, "r") as z:
                z.extractall(ROOT)
            print("Cache extraído en:", ROOT)
        except Exception as e:
            print("No se pudo extraer cache:", e)

print("REGIME_VERSION:", REGIME_VERSION)
print("REGIME_CLUSTER_GRID_ABS:", REGIME_CLUSTER_GRID_ABS)
print("REGIME_CLUSTER_GRID_CHG:", REGIME_CLUSTER_GRID_CHG)


# %% Cell 4
# ============================================================
# 0.1 Instalación opcional de dependencias
# ============================================================

def pip_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception as e:
        print(f"No se pudo instalar {pkg}: {e}")

# En Kaggle normalmente sklearn/pandas/numpy ya están.
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    pip_install("pandas numpy matplotlib openpyxl")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

try:
    import sklearn
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, roc_auc_score,
        precision_score, recall_score, f1_score, matthews_corrcoef,
        confusion_matrix, classification_report, mean_absolute_error,
        mean_squared_error
    )
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
except Exception:
    pip_install("scikit-learn")
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, roc_auc_score,
        precision_score, recall_score, f1_score, matthews_corrcoef,
        confusion_matrix, mean_absolute_error, mean_squared_error
    )
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance

if USE_XGBOOST:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception:
        pip_install("xgboost")
        from xgboost import XGBClassifier, XGBRegressor

if USE_UMAP:
    try:
        import umap
    except Exception:
        pip_install("umap-learn")
        import umap

if USE_YFINANCE:
    try:
        import yfinance as yf
    except Exception:
        pip_install("yfinance")
        import yfinance as yf

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
print("Dependencias listas.")


# %% [markdown]
# ## 1. Utilidades de carga


# %% Cell 6
# ============================================================
# 1. UTILIDADES DE CARGA
# ============================================================

def find_file(patterns, candidates=INPUT_CANDIDATES):
    if isinstance(patterns, str):
        patterns = [patterns]
    hits = []
    for base in candidates:
        if not base.exists():
            continue
        for pat in patterns:
            hits.extend(base.rglob(pat))
    # Preferir archivos fuera de checkpoints y con tamaño > 0
    hits = [p for p in hits if p.is_file() and p.stat().st_size > 0 and ".ipynb_checkpoints" not in str(p)]
    if not hits:
        return None
    hits = sorted(hits, key=lambda p: (len(str(p)), -p.stat().st_size))
    return hits[0]

def normalize_columns(df):
    import unicodedata
    def norm(c):
        c = str(c).strip()
        c = c.replace("\n", " ").replace("\r", " ")
        c = re.sub(r"\s+", "_", c)
        c = c.replace("%", "pct")
        c = c.replace("/", "_")
        c = c.replace("-", "_")
        c = c.replace(".", "_")
        c = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii")
        c = re.sub(r"[^0-9a-zA-Z_]+", "", c)
        c = re.sub(r"_+", "_", c).strip("_")
        return c.lower()
    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out

def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=False)

def pct(x):
    return 100*x

def save_csv(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print("Guardado:", path, df.shape)
    return path

def business_or_daily_reindex(df, date_col="fecha"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(date_col)
    return df



def sanitize_numeric_frame(X):
    """Reemplaza infinitos por NaN y recorta valores extremos para evitar errores de sklearn/xgboost."""
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        # Recorte amplio: evita overflow sin cambiar la escala normal de las variables.
        X[num_cols] = X[num_cols].clip(lower=-1e12, upper=1e12)
    return X

def sanitize_numeric_series(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.clip(lower=-1e12, upper=1e12)

print("Utilidades listas.")


# %% [markdown]
# ## 2. Carga de la base Colombia


# %% Cell 8
# ============================================================
# 2. CARGA BASE COLOMBIA
# ============================================================

base_path = find_file(["BD_Energía_Colombia.xlsx", "BD_Energia_Colombia.xlsx", "*.xlsx"])
print("Base Colombia detectada:", base_path)

if base_path is None:
    raise FileNotFoundError("No se encontró BD_Energía_Colombia.xlsx. Súbelo al entorno o a /kaggle/input.")

xls = pd.ExcelFile(base_path)
print("Hojas:", xls.sheet_names)

# Tomar la primera hoja no vacía
base_raw = None
for sh in xls.sheet_names:
    tmp = pd.read_excel(base_path, sheet_name=sh)
    if tmp.shape[0] > 10 and tmp.shape[1] > 2:
        base_raw = tmp
        print("Hoja usada:", sh, tmp.shape)
        break

if base_raw is None:
    raise ValueError("No se pudo leer una hoja válida del Excel.")

base = normalize_columns(base_raw)

# Detectar fecha
date_candidates = [c for c in base.columns if c in ["fecha", "date", "dia", "time"] or "fecha" in c]
if not date_candidates:
    # probar primera columna
    date_candidates = [base.columns[0]]
date_col = date_candidates[0]
base["fecha"] = pd.to_datetime(base[date_col], errors="coerce")
base = base.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha")

# Renombrar columnas comunes a nombres canónicos
rename_map = {}
for c in base.columns:
    lc = c.lower()
    if "oro" in lc and "precio" in lc:
        rename_map[c] = "precio_oro"
    elif lc == "oro":
        rename_map[c] = "precio_oro"
    elif "trm" in lc:
        rename_map[c] = "trm"
    elif "tipm" in lc:
        rename_map[c] = "tipm"
    elif "dtf" in lc:
        rename_map[c] = "dtf"
    elif "brent" in lc:
        rename_map[c] = "precio_brent"
    elif "cafe" in lc or "caf" in lc:
        rename_map[c] = "precio_cafe_centusd"
    elif "demanda" in lc and "ener" in lc:
        rename_map[c] = "demanda_energetica"
    elif "inflacion" in lc and "alimento" in lc:
        rename_map[c] = "inflacion_sin_alimentos"
    elif "bolsa" in lc and "ener" in lc:
        rename_map[c] = "precio_bolsa_nacional_energetica"
    elif "bancolombia" in lc:
        rename_map[c] = "bancolombia_price_usd"

base = base.rename(columns=rename_map)
base = base.loc[:, ~base.columns.duplicated()]

if "precio_oro" not in base.columns:
    print("Columnas disponibles:", list(base.columns))
    raise ValueError("No se detectó columna de precio de oro. Renómbrala como precio_oro.")

# Convertir numéricas
for c in base.columns:
    if c != "fecha":
        base[c] = pd.to_numeric(base[c], errors="coerce")

base = base.sort_values("fecha").reset_index(drop=True)
print(base.shape, base["fecha"].min(), base["fecha"].max())
print("Valores únicos oro:", base["precio_oro"].nunique())

save_csv(base[["fecha"] + [c for c in base.columns if c != "fecha"]], "base_colombia_normalizada.csv")
base.head()


# %% [markdown]
# ## 3. Cobertura y frecuencia efectiva


# %% Cell 10
# ============================================================
# 3. COBERTURA Y FRECUENCIA EFECTIVA
# ============================================================

def infer_freq_from_unique_ratio(n_unique, n):
    rep = 1 - n_unique / max(n, 1)
    if rep > 0.95:
        return "mensual→diaria (ff)"
    if rep > 0.80:
        return "semanal/trimestral→diaria (ff)"
    if rep > 0.25:
        return "diaria con repetidos"
    return "diaria"

coverage_rows = []
for c in base.columns:
    if c == "fecha":
        continue
    n = base[c].notna().sum()
    nu = base[c].nunique(dropna=True)
    rep = 1 - nu / max(n, 1)
    coverage_rows.append({
        "var": c,
        "grupo": "Colombia",
        "n_no_nulos": int(n),
        "n_unicos": int(nu),
        "pct_repetido": round(100*rep, 2),
        "freq": infer_freq_from_unique_ratio(nu, n),
        "fecha_ini": base.loc[base[c].notna(), "fecha"].min(),
        "fecha_fin": base.loc[base[c].notna(), "fecha"].max(),
    })

coverage = pd.DataFrame(coverage_rows)
save_csv(coverage, "coverage.csv")

gold_series_cols = ["fecha", "precio_oro"]
for c in ["trm", "precio_brent"]:
    if c in base.columns:
        gold_series_cols.append(c)

gold_series = base[gold_series_cols].copy()
save_csv(gold_series, "gold_series.csv")

coverage.sort_values("pct_repetido", ascending=False).head(12)


# %% [markdown]
# ## 4. Enfermedad holandesa


# %% Cell 12
# ============================================================
# 4. ENFERMEDAD HOLANDESA
# ============================================================

dd_path = find_file(["*enfermedad*holandesa*.csv", "*dutch*disease*.csv", "*enfermedad*holandesa*.xlsx"])
print("Archivo enfermedad holandesa detectado:", dd_path)

dd = None
if dd_path is not None:
    if dd_path.suffix.lower() == ".csv":
        dd = pd.read_csv(dd_path)
    else:
        dd = pd.read_excel(dd_path)
    dd = normalize_columns(dd)
    # Detectar year
    year_col = None
    for c in dd.columns:
        if c in ["year", "anio", "ano"] or "year" in c or "anio" in c:
            year_col = c
            break
    if year_col is None:
        # probar fecha
        date_cols = [c for c in dd.columns if "fecha" in c or "date" in c]
        if date_cols:
            dd["year"] = pd.to_datetime(dd[date_cols[0]], errors="coerce").dt.year
        else:
            dd["year"] = dd.iloc[:,0]
    else:
        dd["year"] = pd.to_numeric(dd[year_col], errors="coerce").astype("Int64")

    # Canónicos
    for c in list(dd.columns):
        lc = c.lower()
        if "dd_score" in lc or ("score" in lc and "dd" in lc):
            dd = dd.rename(columns={c: "dd_score"})
        if "clas" in lc or "categoria" in lc:
            dd = dd.rename(columns={c: "dd_clasificacion"})

    if "dd_score" not in dd.columns:
        score_cols = [c for c in dd.columns if "score" in c]
        if score_cols:
            dd = dd.rename(columns={score_cols[0]: "dd_score"})
    if "dd_clasificacion" not in dd.columns:
        dd["dd_clasificacion"] = "Sin clasificación"

    dd = dd.dropna(subset=["year"]).copy()
    dd["year"] = dd["year"].astype(int)
    print(dd.shape)
else:
    # Intentar desde base
    dd_cols = [c for c in base.columns if c.startswith("dd_") or "dd_lag" in c or "holand" in c]
    if dd_cols:
        tmp = base[["fecha"] + dd_cols].copy()
        tmp["year"] = tmp["fecha"].dt.year
        dd = tmp.groupby("year").last().reset_index()
    else:
        dd = pd.DataFrame(columns=["year", "dd_score", "dd_clasificacion"])

# Crear lags y unir a base
if not dd.empty:
    dd = dd.sort_values("year").drop_duplicates("year")
    if "dd_score" in dd.columns:
        dd["dd_lag1_dd_score"] = dd["dd_score"].shift(1)
    else:
        dd["dd_lag1_dd_score"] = np.nan
    if "dd_clasificacion" in dd.columns:
        dd["dd_lag1_dd_clasificacion"] = dd["dd_clasificacion"].shift(1)
    else:
        dd["dd_lag1_dd_clasificacion"] = "Sin clasificación"
    save_csv(dd, "dutch_disease.csv")

    base["year"] = base["fecha"].dt.year
    base = base.merge(dd[["year", "dd_lag1_dd_score", "dd_lag1_dd_clasificacion"]], on="year", how="left")
    base["dd_lag1_dd_score"] = pd.to_numeric(base["dd_lag1_dd_score"], errors="coerce")
else:
    print("No hay enfermedad holandesa disponible. Se continúa sin DD.")

base.head()


# %% [markdown]
# ## 5. Contexto global


# %% Cell 14
# ============================================================
# 5. CONTEXTO GLOBAL
# ============================================================

def load_available_global_context():
    patterns = [
        "global_market_context_homogenized.csv",
        "global_market_context.csv",
        "*global*context*.csv",
        "*contexto*global*.csv",
    ]
    gpath = find_file(patterns)
    if gpath is None:
        return None
    print("Contexto global detectado:", gpath)
    return pd.read_csv(gpath)

global_ctx = load_available_global_context()

if global_ctx is None and USE_YFINANCE:
    print("Descargando contexto global desde yfinance...")
    tickers = ["GLD","IAU","SLV","GC=F","SI=F","CL=F","HG=F","PL=F","PA=F","DX-Y.NYB","^VIX","SPY","QQQ","^TNX","^IRX"]
    start = base["fecha"].min().strftime("%Y-%m-%d")
    end = (base["fecha"].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    raw_parts = []
    failures = []
    for t in tickers:
        try:
            data = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            if data is None or data.empty:
                failures.append({"ticker": t, "error": "empty"})
                continue
            data.columns = [str(c[0] if isinstance(c, tuple) else c).lower().replace(" ", "_") for c in data.columns]
            data = data.reset_index().rename(columns={"Date":"fecha", "date":"fecha"})
            if "fecha" not in data.columns:
                data = data.rename(columns={data.columns[0]:"fecha"})
            keep = [c for c in ["fecha", "open", "high", "low", "close", "adj_close", "volume"] if c in data.columns]
            data = data[keep].copy()
            prefix = re.sub(r"[^0-9a-zA-Z]+", "_", t).strip("_").lower()
            data = data.rename(columns={c: f"{prefix}_{c}" for c in data.columns if c != "fecha"})
            raw_parts.append(data)
        except Exception as e:
            failures.append({"ticker": t, "error": str(e)[:200]})
    if raw_parts:
        global_ctx = raw_parts[0]
        for part in raw_parts[1:]:
            global_ctx = global_ctx.merge(part, on="fecha", how="outer")
        global_ctx["fecha"] = pd.to_datetime(global_ctx["fecha"])
        global_ctx = global_ctx.sort_values("fecha")
        pd.DataFrame(failures).to_csv(OUT / "yfinance_download_failures.csv", index=False)
    else:
        global_ctx = pd.DataFrame({"fecha": base["fecha"]})

if global_ctx is None:
    print("No se encontró ni se descargó contexto global. Se continúa con variables Colombia.")
    global_ctx = pd.DataFrame({"fecha": base["fecha"]})

global_ctx = normalize_columns(global_ctx)
if "fecha" not in global_ctx.columns:
    date_cols = [c for c in global_ctx.columns if "date" in c or "fecha" in c]
    if date_cols:
        global_ctx = global_ctx.rename(columns={date_cols[0]:"fecha"})
global_ctx["fecha"] = pd.to_datetime(global_ctx["fecha"], errors="coerce")
global_ctx = global_ctx.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha")

# Reindexación al calendario local.
# Regla anti-fuga: solo forward-fill. No se usa backfill para rellenar datos futuros hacia el pasado.
calendar = pd.DataFrame({"fecha": base["fecha"].sort_values().unique()})
global_ctx = calendar.merge(global_ctx, on="fecha", how="left")
global_value_cols = [c for c in global_ctx.columns if c != "fecha"]
global_ctx[global_value_cols] = global_ctx[global_value_cols].ffill()

save_csv(global_ctx, "global_market_context_homogenized.csv")
print("Global ctx:", global_ctx.shape)
global_ctx.head()


# %% [markdown]
# ## 6. Ingeniería de variables


# %% Cell 16
# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def add_time_features(df):
    df = df.copy()
    df["year"] = df["fecha"].dt.year
    df["month"] = df["fecha"].dt.month
    df["quarter"] = df["fecha"].dt.quarter
    df["dayofyear"] = df["fecha"].dt.dayofyear
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df

def add_series_features(df, cols, prefix_group="", lags=(1,5,21,63), windows=(21,63,126)):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # Log solo si positivo
        if (s.dropna() > 0).mean() > 0.98:
            log_s = np.log(s.where(s > 0))
            df[f"{c}_log"] = log_s
            df[f"{c}_logret_1"] = log_s.diff()
            df[f"{c}_logret_5"] = log_s.diff(5)
            df[f"{c}_logret_21"] = log_s.diff(21)
        df[f"{c}_diff_1"] = s.diff()
        df[f"{c}_pct_21"] = s.pct_change(21)
        for L in lags:
            df[f"{c}_lag{L}"] = s.shift(L)
        for W in windows:
            df[f"{c}_roll_mean{W}"] = s.rolling(W, min_periods=max(5, W//4)).mean()
            df[f"{c}_roll_std{W}"] = s.rolling(W, min_periods=max(5, W//4)).std()
            df[f"{c}_dist_max{W}"] = s / s.rolling(W, min_periods=max(5, W//4)).max() - 1
            df[f"{c}_dist_min{W}"] = s / s.rolling(W, min_periods=max(5, W//4)).min() - 1
    return df

# Base de modelado
df = base.copy()
df = add_time_features(df)

colombia_base_cols = [
    "demanda_energetica", "trm", "tipm", "precio_brent", "precio_oro",
    "precio_cafe_centusd", "dtf", "bancolombia_price_usd",
    "inflacion_sin_alimentos", "precio_bolsa_nacional_energetica",
    "dd_lag1_dd_score"
]
colombia_base_cols = [c for c in colombia_base_cols if c in df.columns]

df = add_series_features(df, colombia_base_cols, lags=(1,5,21,63), windows=(21,63,126))

# Política monetaria
if "tipm" in df.columns:
    df["tipm_change_30"] = df["tipm"].diff(30)
    df["policy_stance"] = np.select(
        [df["tipm_change_30"] > 0.05, df["tipm_change_30"] < -0.05],
        ["tightening", "easing"],
        default="neutral"
    )
else:
    df["policy_stance"] = "unknown"

# Variables dummy categóricas moderadas
for cat in ["policy_stance", "dd_lag1_dd_clasificacion"]:
    if cat in df.columns:
        dummies = pd.get_dummies(df[cat].astype(str), prefix=cat, dummy_na=False)
        df = pd.concat([df, dummies], axis=1)

# Contexto global + features globales
global_cols_raw = [c for c in global_ctx.columns if c != "fecha"]
gfeat = global_ctx.copy()
gfeat = add_series_features(gfeat, global_cols_raw, lags=(1,5,21,63), windows=(21,63,126))

# Unir global features
global_feature_cols = [c for c in gfeat.columns if c != "fecha"]
df = df.merge(gfeat[["fecha"] + global_feature_cols], on="fecha", how="left", suffixes=("", "_globaldup"))



# Limpieza global de infinitos generados por pct_change, ratios o divisiones por valores cercanos a cero.
numeric_cols_df = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[numeric_cols_df].to_numpy()).sum(axis=0)
sanitization_report = pd.DataFrame({
    "column": numeric_cols_df,
    "n_inf_before": inf_counts
})
df[numeric_cols_df] = df[numeric_cols_df].replace([np.inf, -np.inf], np.nan)
df[numeric_cols_df] = df[numeric_cols_df].clip(lower=-1e12, upper=1e12)
sanitization_report["n_inf_after"] = np.isinf(df[numeric_cols_df].to_numpy()).sum(axis=0)
sanitization_report = sanitization_report[sanitization_report["n_inf_before"] > 0]
save_csv(sanitization_report, "feature_sanitization_report.csv")

print("df final:", df.shape)
print("features Colombia base:", len(colombia_base_cols))
print("features global:", len(global_feature_cols))


# %% [markdown]
# ## 7. Targets h21/h30


# %% Cell 18
# ============================================================
# 7. TARGETS
# ============================================================

for h in HORIZONS:
    df[f"future_price_h{h}"] = df["precio_oro"].shift(-h)
    df[f"future_logret_h{h}"] = np.log(df[f"future_price_h{h}"] / df["precio_oro"])
    df[f"target_up_h{h}"] = (df[f"future_logret_h{h}"] > 0).astype(float)
    df.loc[df[f"future_logret_h{h}"].isna(), f"target_up_h{h}"] = np.nan

target_diag = []
for h in HORIZONS:
    yret = df[f"future_logret_h{h}"].dropna()
    y = df.loc[yret.index, f"target_up_h{h}"]
    target_diag.append({
        "horizon": h,
        "n": len(y),
        "positive_share": float(y.mean()),
        "non_positive_share": float(1-y.mean()),
        "zero_return_share": float((yret == 0).mean()),
        "mean_return": float(yret.mean()),
        "std_return": float(yret.std()),
    })
target_diag = pd.DataFrame(target_diag)
save_csv(target_diag, "target_diagnostics.csv")
target_diag


# %% [markdown]
# ## 8. Regímenes UMAP/KMeans optimizados


# %% Cell 20
# ============================================================
# 8. REGÍMENES UMAP/KMEANS · FAMILIA 6/7 CLUSTERS
# ============================================================

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
)

regime_cols = [
    "demanda_energetica", "trm", "tipm", "precio_brent", "precio_oro",
    "precio_cafe_centusd", "dtf", "bancolombia_price_usd",
    "inflacion_sin_alimentos", "precio_bolsa_nacional_energetica",
    "tipm_change_30", "dd_lag1_dd_score"
]
regime_cols = [c for c in regime_cols if c in df.columns]
print("Columnas régimen:", regime_cols)

def prepare_regime_matrix(data, cols):
    X = data[cols].copy()
    X = sanitize_numeric_frame(X)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=cols)
    Xs = StandardScaler().fit_transform(X)
    return X, Xs

def metadata_alignment_scores(labels, data):
    """Mide alineación estructural con enfermedad holandesa y policy stance.
    No usa el target futuro. Sirve para elegir entre k=6 y k=7 cuando la silueta es parecida.
    """
    out = {}
    lab = pd.Series(labels).astype(str)
    if "dd_lag1_dd_clasificacion" in data.columns:
        dd = data["dd_lag1_dd_clasificacion"].astype(str).fillna("NA")
        try:
            out["nmi_dutch_disease"] = normalized_mutual_info_score(dd, lab)
        except Exception:
            out["nmi_dutch_disease"] = np.nan
    else:
        out["nmi_dutch_disease"] = np.nan

    if "policy_stance" in data.columns:
        ps = data["policy_stance"].astype(str).fillna("NA")
        try:
            out["nmi_policy_stance"] = normalized_mutual_info_score(ps, lab)
        except Exception:
            out["nmi_policy_stance"] = np.nan
    else:
        out["nmi_policy_stance"] = np.nan

    return out

def optimize_umap_kmeans_6_7(Xs, meta_df, label_prefix="abs", random_state=RANDOM_STATE):
    if FAST_MODE:
        n_neighbors_grid = [30]
        min_dist_grid = [0.05, 0.10]
    else:
        n_neighbors_grid = [15, 30, 45, 75]
        min_dist_grid = [0.00, 0.05, 0.10, 0.25]

    if label_prefix == "abs":
        n_clusters_grid = REGIME_CLUSTER_GRID_ABS
    else:
        n_clusters_grid = REGIME_CLUSTER_GRID_CHG

    metrics_grid = ["euclidean"]

    n = Xs.shape[0]
    sample_size = min(3000, n)
    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(rng.choice(np.arange(n), size=sample_size, replace=False)) if n > sample_size else np.arange(n)

    rows = []
    best = None

    for metric in metrics_grid:
        for nn in n_neighbors_grid:
            for mdist in min_dist_grid:
                if USE_UMAP and Xs.shape[1] >= 3:
                    reducer = umap.UMAP(
                        n_neighbors=nn,
                        min_dist=mdist,
                        n_components=2,
                        metric=metric,
                        random_state=random_state
                    )
                    emb = reducer.fit_transform(Xs)
                    method = "UMAP"
                else:
                    from sklearn.decomposition import PCA
                    reducer = None
                    emb = PCA(n_components=2, random_state=random_state).fit_transform(Xs)
                    method = "PCA"

                for k in n_clusters_grid:
                    km = KMeans(n_clusters=k, n_init=30, random_state=random_state)
                    labels = km.fit_predict(emb)

                    emb_s = emb[sample_idx]
                    lab_s = labels[sample_idx]
                    if len(np.unique(lab_s)) < 2:
                        sil = np.nan
                        cal = np.nan
                        db = np.nan
                    else:
                        sil = silhouette_score(emb_s, lab_s)
                        cal = calinski_harabasz_score(emb_s, lab_s)
                        db = davies_bouldin_score(emb_s, lab_s)

                    align = metadata_alignment_scores(labels, meta_df)
                    nmi_dd = align.get("nmi_dutch_disease", np.nan)
                    nmi_policy = align.get("nmi_policy_stance", np.nan)

                    # Score compuesto: prioriza silueta, pero incorpora alineación estructural
                    # con enfermedad holandesa y postura monetaria sin usar el target.
                    structural_score = (
                        (sil if pd.notna(sil) else -999)
                        + 0.15 * (nmi_dd if pd.notna(nmi_dd) else 0)
                        + 0.10 * (nmi_policy if pd.notna(nmi_policy) else 0)
                    )

                    row = {
                        "embedding": label_prefix,
                        "regime_version": REGIME_VERSION,
                        "method": method,
                        "metric": metric,
                        "n_neighbors": nn,
                        "min_dist": mdist,
                        "n_clusters": k,
                        "silhouette": sil,
                        "calinski_harabasz": cal,
                        "davies_bouldin": db,
                        "nmi_dutch_disease": nmi_dd,
                        "nmi_policy_stance": nmi_policy,
                        "structural_silhouette_score": structural_score,
                        "sample_size": len(sample_idx),
                    }
                    rows.append(row)

                    score = structural_score if REGIME_SELECTION_SCORE == "structural_silhouette" else (sil if pd.notna(sil) else -999)
                    if best is None or score > best["selection_score"]:
                        best = {
                            **row,
                            "selection_score": score,
                            "embedding_values": emb,
                            "cluster_labels": labels,
                            "reducer": reducer,
                            "kmeans": km,
                        }

    results = pd.DataFrame(rows).sort_values(
        ["structural_silhouette_score", "silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    return results, best

# ----------------------------
# 8.1 UMAP de niveles absolutos
# ----------------------------
reg = df[["fecha"] + regime_cols].copy()
X_abs, Xs_abs = prepare_regime_matrix(reg, regime_cols)

opt_abs, best_abs = optimize_umap_kmeans_6_7(
    Xs_abs,
    df,
    label_prefix="abs",
    random_state=RANDOM_STATE
)
save_csv(opt_abs, "umap_optimization_abs.csv")

emb_abs = best_abs["embedding_values"]
cluster_abs = best_abs["cluster_labels"]

print("Mejor UMAP/KMeans ABS 6/7:")
display(opt_abs.head(10))

# ----------------------------
# 8.2 UMAP de cambios
# ----------------------------
chg_cols = []
for c in regime_cols:
    reg[f"{c}_chg"] = pd.to_numeric(reg[c], errors="coerce").diff(21)
    chg_cols.append(f"{c}_chg")

X_chg, Xs_chg = prepare_regime_matrix(reg, chg_cols)

opt_chg, best_chg = optimize_umap_kmeans_6_7(
    Xs_chg,
    df,
    label_prefix="chg",
    random_state=RANDOM_STATE + 1
)
save_csv(opt_chg, "umap_optimization_chg.csv")

emb_chg = best_chg["embedding_values"]
cluster_chg = best_chg["cluster_labels"]

print("Mejor UMAP/KMeans CHG 6/7:")
display(opt_chg.head(10))

# ----------------------------
# 8.3 Guardar regímenes finales
# ----------------------------
regimes = pd.DataFrame({
    "fecha": df["fecha"],
    "umap_abs_1": emb_abs[:, 0],
    "umap_abs_2": emb_abs[:, 1],
    "cluster_abs": cluster_abs,
    "umap_chg_1": emb_chg[:, 0],
    "umap_chg_2": emb_chg[:, 1],
    "cluster_chg": cluster_chg,
})
save_csv(regimes, "umap_regimes.csv")

regime_design = pd.DataFrame([
    {
        "regime_version": REGIME_VERSION,
        "embedding": "abs",
        "selected_n_clusters": best_abs["n_clusters"],
        "selected_n_neighbors": best_abs["n_neighbors"],
        "selected_min_dist": best_abs["min_dist"],
        "silhouette": best_abs["silhouette"],
        "nmi_dutch_disease": best_abs["nmi_dutch_disease"],
        "nmi_policy_stance": best_abs["nmi_policy_stance"],
        "selection_score": best_abs["selection_score"],
    },
    {
        "regime_version": REGIME_VERSION,
        "embedding": "chg",
        "selected_n_clusters": best_chg["n_clusters"],
        "selected_n_neighbors": best_chg["n_neighbors"],
        "selected_min_dist": best_chg["min_dist"],
        "silhouette": best_chg["silhouette"],
        "nmi_dutch_disease": best_chg["nmi_dutch_disease"],
        "nmi_policy_stance": best_chg["nmi_policy_stance"],
        "selection_score": best_chg["selection_score"],
    },
])
save_csv(regime_design, "regime_design_6_7.csv")

# Unir al dataframe principal
df = df.drop(columns=[c for c in ["cluster_abs", "cluster_chg"] if c in df.columns], errors="ignore")
df = df.merge(regimes[["fecha", "cluster_abs", "cluster_chg"]], on="fecha", how="left")

# Dummies de clusters
for cat in ["cluster_abs", "cluster_chg"]:
    dummies = pd.get_dummies(df[cat].astype("Int64").astype(str), prefix=cat)
    df = df.drop(columns=[c for c in df.columns if c.startswith(cat + "_")], errors="ignore")
    df = pd.concat([df, dummies], axis=1)

# ----------------------------
# 8.4 Figuras
# ----------------------------
def plot_umap_optimization(opt, out_name, title):
    if opt.empty:
        return
    top = opt.head(20).copy()
    top["config"] = (
        "k=" + top["n_clusters"].astype(str)
        + " | nn=" + top["n_neighbors"].astype(str)
        + " | md=" + top["min_dist"].astype(str)
    )
    top = top.sort_values("structural_silhouette_score", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top["config"], top["structural_silhouette_score"])
    plt.xlabel("Structural silhouette score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / out_name, dpi=180)
    plt.close()

def plot_umap_embedding(regimes_df, x_col, y_col, cluster_col, out_name, title):
    r = regimes_df.dropna(subset=[x_col, y_col, cluster_col]).copy()
    if len(r) > 5000:
        r = r.sample(5000, random_state=RANDOM_STATE).sort_values("fecha")
    plt.figure(figsize=(8, 6))
    for cl, g in r.groupby(cluster_col):
        plt.scatter(g[x_col], g[y_col], s=6, alpha=0.55, label=f"{cluster_col}={cl}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / out_name, dpi=180)
    plt.close()

plot_umap_optimization(opt_abs, "umap_optimization_abs_silhouette.png", "Optimización UMAP/KMeans ABS · familia 6/7")
plot_umap_optimization(opt_chg, "umap_optimization_chg_silhouette.png", "Optimización UMAP/KMeans CHG · familia 6/7")
plot_umap_embedding(regimes, "umap_abs_1", "umap_abs_2", "cluster_abs", "umap_regimes_abs.png", "Regímenes UMAP ABS · 6/7")
plot_umap_embedding(regimes, "umap_chg_1", "umap_chg_2", "cluster_chg", "umap_regimes_chg.png", "Regímenes UMAP CHG · 6/7")

print("Regímenes UMAP/KMeans 6/7 listos.")
regimes.head()


# %% [markdown]
# ## 9. Feature sets del dashboard


# %% Cell 22
# ============================================================
# 9. FEATURE SETS
# ============================================================

# Evitar columnas objetivo/futuras/no features
forbidden_patterns = [
    "future_", "target_", "fecha", "date",
]
forbidden_exact = set(["precio_oro"])  # el precio actual crudo se puede dejar fuera para evitar trivialidad; se incluyen lags/logrets

all_numeric_cols = []
for c in df.columns:
    if c in forbidden_exact:
        continue
    if any(c.startswith(p) for p in forbidden_patterns):
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        all_numeric_cols.append(c)

# Identificación de globales por nombres/tickers
global_markers = [
    "gld", "iau", "slv", "gc_f", "si_f", "cl_f", "hg_f", "pl_f", "pa_f",
    "dx_y_nyb", "vix", "spy", "qqq", "tnx", "irx"
]
def is_global(c):
    return any(m in c.lower() for m in global_markers)

global_features = [c for c in all_numeric_cols if is_global(c)]
colombia_features = [c for c in all_numeric_cols if not is_global(c)]

# Remover columnas con demasiados NaN y columnas constantes
def clean_feature_list(cols, min_nonnull=0.55):
    out = []
    n = len(df)
    for c in cols:
        if c not in df.columns:
            continue
        s = sanitize_numeric_series(df[c])
        finite_share = np.isfinite(s.dropna()).mean() if s.notna().any() else 0
        if s.notna().mean() < min_nonnull:
            continue
        if finite_share < 1.0:
            # Ya se reemplazan infinitos por NaN; esta condición deja trazabilidad conceptual.
            pass
        if s.nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return out

colombia_features = clean_feature_list(colombia_features, min_nonnull=0.70)
global_features = clean_feature_list(global_features, min_nonnull=0.40)

# Compact global: seleccionar transformaciones principales y evitar demasiadas columnas por ticker
compact_keywords = [
    "close", "adj_close", "logret_1", "logret_5", "logret_21",
    "pct_21", "roll_std21", "roll_mean21", "dist_max63", "dist_min63", "lag1", "lag5", "lag21"
]
global_compact = [c for c in global_features if any(k in c for k in compact_keywords)]

# Si queda demasiado grande, limitar por varianza / cobertura
if len(global_compact) > 250:
    variances = df[global_compact].var(numeric_only=True).sort_values(ascending=False)
    global_compact = list(variances.head(250).index)

# Auditoría
gold_direct_markers = ["gld", "iau", "gc_f"]
metal_markers = ["gld", "iau", "gc_f", "slv", "si_f", "pl_f", "pa_f", "hg_f"]

def exclude_markers(cols, markers):
    return [c for c in cols if not any(m in c.lower() for m in markers)]

macro_global_markers = ["dx_y_nyb", "vix", "spy", "qqq", "tnx", "irx", "cl_f"]
global_macro_only = [c for c in global_features if any(m in c.lower() for m in macro_global_markers)]

feature_sets = {
    "colombia_only": colombia_features,
    "colombia_plus_global_compact": colombia_features + global_compact,
    "colombia_plus_global_full": colombia_features + global_features,
    "global_no_gold_direct": colombia_features + exclude_markers(global_compact, gold_direct_markers),
    "global_no_metals": colombia_features + exclude_markers(global_compact, metal_markers),
    "global_macro_only": colombia_features + global_macro_only,
}

if FAST_MODE:
    feature_sets = {
        k: v[:min(len(v), 250)] for k, v in feature_sets.items()
    }

for k, v in feature_sets.items():
    print(k, len(v))

# Guardar conteo por grupo
pd.DataFrame([{"feature_set": k, "n_features": len(v), "features": "|".join(v[:100])} for k,v in feature_sets.items()]).to_csv(OUT / "feature_sets_catalog.csv", index=False)


# %% [markdown]
# ## 10. Modelos del dashboard


# %% Cell 24
# ============================================================
# 10. MODELOS
# ============================================================

def make_model(name):
    if name == "XGBClassifier":
        params = dict(
            n_estimators=320 if not FAST_MODE else 80,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=2.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        # GPU opcional. Si falla por versión de XGBoost/CUDA, dejar XGB_USE_GPU=False.
        if globals().get("XGB_USE_GPU", False):
            params.update({"tree_method": "hist", "device": "cuda"})
        return XGBClassifier(**params)
    if name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            max_iter=300 if not FAST_MODE else 80,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE
        )
    if name == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=260 if not FAST_MODE else 100,
            max_depth=None,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None
        )
    if name == "ExtraTreesClassifier":
        return ExtraTreesClassifier(
            n_estimators=280 if not FAST_MODE else 120,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None
        )
    if name == "LogisticRegression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE))
        ])
    raise ValueError(name)

model_names = ["XGBClassifier", "HistGradientBoostingClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "LogisticRegression"]

model_names


# %% [markdown]
# ## 11. Split temporal y calibración


# %% Cell 26
# ============================================================
# 11. MÉTRICAS Y CALIBRACIÓN
# ============================================================

def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    # fallback decision function
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1/(1+np.exp(-z))
    return model.predict(X).astype(float)

def best_threshold(y_true, proba, grid=None):
    if grid is None:
        grid = adaptive_threshold_grid(proba) if 'adaptive_threshold_grid' in globals() else np.linspace(0.15, 0.85, 71)
    best_t, best_b = 0.5, -1
    for t in grid:
        pred = (proba >= t).astype(int)
        b = balanced_accuracy_score(y_true, pred)
        if b > best_b:
            best_b = b
            best_t = t
    return float(best_t), float(best_b)

def metrics_row(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_up": precision_score(y_true, pred, zero_division=0),
        "recall_up": recall_score(y_true, pred, zero_division=0),
        "f1_up": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 and len(np.unique(y_true)) > 1 else np.nan,
        "threshold": threshold,
        "n_test": len(y_true),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    except Exception:
        out["roc_auc"] = np.nan
    cm = confusion_matrix(y_true, pred, labels=[0,1])
    out.update({"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])})
    out["recall_not_up"] = out["tn"] / max(out["tn"] + out["fp"], 1)
    return out

def select_top_features_by_train(X_train, y_train, features, max_features=1200):
    # Reduce dimensión para modelos pesados: varianza + correlación con y aproximada
    if len(features) <= max_features:
        return features
    X = pd.DataFrame(X_train, columns=features)
    X = sanitize_numeric_frame(X)
    # imputación rápida
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    scores = {}
    y = pd.Series(y_train).reset_index(drop=True)
    for c in features:
        s = sanitize_numeric_series(X[c])
        if s.std() == 0 or s.notna().sum() < 20:
            scores[c] = 0
        else:
            valid = s.notna().values & pd.notna(y).values
            if valid.sum() < 20:
                scores[c] = 0
            else:
                corr = np.corrcoef(s.loc[valid], y.loc[valid])[0,1]
                scores[c] = abs(corr) if np.isfinite(corr) else 0
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:max_features]

def compute_sample_weight_strategy(train_df, y, strategy="none"):
    """Calcula pesos de entrenamiento usando solo información del bloque de entrenamiento."""
    y = np.asarray(y).astype(int)
    n = len(y)
    w = np.ones(n, dtype=float)
    if strategy is None or strategy == "none":
        return w

    if strategy == "balanced":
        p = y.mean()
        w[y == 1] *= 0.5 / max(p, 1e-6)
        w[y == 0] *= 0.5 / max(1-p, 1e-6)
        return w / np.mean(w)

    if strategy == "cluster_x4" and "cluster_abs" in train_df.columns:
        grp = train_df["cluster_abs"].astype("Int64").astype(str).values
        vc = pd.Series(grp).value_counts(normalize=True)
        wg = pd.Series(grp).map(lambda g: 1 / max(vc.get(g, 1), 1e-6)).values
        wg = wg / np.mean(wg)
        # limitar para que no domine excesivamente
        wg = np.clip(wg, 0.35, 4.0)
        return wg / np.mean(wg)

    if strategy == "macrofull_x15":
        parts = []
        for col in ["cluster_abs", "cluster_chg", "policy_stance", "dd_lag1_dd_clasificacion"]:
            if col in train_df.columns:
                grp = train_df[col].astype(str).values
                vc = pd.Series(grp).value_counts(normalize=True)
                wg = pd.Series(grp).map(lambda g: 1 / max(vc.get(g, 1), 1e-6)).values
                parts.append(wg / np.mean(wg))
        if parts:
            wg = np.mean(np.vstack(parts), axis=0)
            wg = np.clip(wg, 0.35, 15.0)
            # también balancea la clase
            p = y.mean()
            wc = np.ones(n)
            wc[y == 1] *= 0.5 / max(p, 1e-6)
            wc[y == 0] *= 0.5 / max(1-p, 1e-6)
            w = wg * wc
            return w / np.mean(w)
        return compute_sample_weight_strategy(train_df, y, "balanced")

    if strategy == "recency_4y":
        if "fecha" in train_df.columns:
            dates = pd.to_datetime(train_df["fecha"])
            maxd = dates.max()
            age_days = (maxd - dates).dt.days.clip(lower=0)
            # peso exponencial: semivida aproximada de 4 años
            half_life = 365.25 * 4
            w = np.exp(-np.log(2) * age_days / half_life).values
            w = np.clip(w, 0.20, 1.0)
            return w / np.mean(w)
        return np.ones(n)

    return np.ones(n)

def fit_with_optional_weights(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except Exception:
        # Pipeline con clasificador final llamado clf
        try:
            return model.fit(X, y, clf__sample_weight=sample_weight)
        except Exception:
            return model.fit(X, y)

def adaptive_threshold_grid(proba):
    proba = np.asarray(proba)
    proba = proba[np.isfinite(proba)]
    if len(proba) < 10:
        return np.linspace(0.15, 0.85, 71)
    q1, q99 = np.quantile(proba, [0.01, 0.99])
    lo = max(0.05, min(0.30, q1))
    hi = min(0.95, max(0.70, q99))
    return np.linspace(lo, hi, 101)


# %% [markdown]
# ## 12. Validación walk-forward y búsqueda del mejor modelo


# %% [markdown]
# ## Nota sobre la optimización de la grilla


# %% Cell 29
# ============================================================
# 12. WALK-FORWARD
# ============================================================

def make_walkforward_blocks(data, start_test="2020-05-01", block_months=6):
    dates = pd.to_datetime(data["fecha"])
    start = pd.Timestamp(start_test)
    end_all = dates.max()
    blocks = []
    cur = start
    while cur < end_all:
        end = cur + pd.DateOffset(months=block_months)
        if end > end_all:
            end = end_all
        blocks.append((cur, end))
        cur = end
    return blocks

def run_walkforward_for_config(data, horizon, feature_set_name, features, model_name, train_window_years=None, max_features=1200, weight_strategy='none'):
    target = f"target_up_h{horizon}"
    ret_col = f"future_logret_h{horizon}"
    d = data.dropna(subset=[target, ret_col]).copy()
    d[target] = d[target].astype(int)
    blocks = make_walkforward_blocks(d, start_test="2020-05-01", block_months=6 if not FAST_MODE else 12)
    all_preds = []
    results = []
    selected_features_rows = []

    for bi, (test_start, test_end) in enumerate(blocks):
        train_mask = d["fecha"] < test_start
        if train_window_years is not None:
            train_mask &= d["fecha"] >= (test_start - pd.DateOffset(years=train_window_years))
        test_mask = (d["fecha"] >= test_start) & (d["fecha"] < test_end)

        train = d.loc[train_mask].copy()
        test = d.loc[test_mask].copy()
        if len(train) < 800 or len(test) < 20:
            continue

        # Calibración: últimos 20 % o 1 año de entrenamiento
        cal_start = max(train["fecha"].quantile(0.80), test_start - pd.DateOffset(years=1))
        fit = train[train["fecha"] < cal_start].copy()
        cal = train[train["fecha"] >= cal_start].copy()
        if len(fit) < 500 or len(cal) < 50:
            fit = train.iloc[:int(len(train)*0.8)].copy()
            cal = train.iloc[int(len(train)*0.8):].copy()

        # Selección de features con solo entrenamiento
        candidate_features = [f for f in features if f in d.columns]
        candidate_features = clean_feature_list(candidate_features, min_nonnull=0.0)

        X_fit_raw = sanitize_numeric_frame(fit[candidate_features])
        y_fit = fit[target].values
        selected = select_top_features_by_train(X_fit_raw, y_fit, candidate_features, max_features=max_features)

        imputer = SimpleImputer(strategy="median")
        X_fit = imputer.fit_transform(sanitize_numeric_frame(fit[selected]))
        X_cal = imputer.transform(sanitize_numeric_frame(cal[selected]))
        X_test = imputer.transform(sanitize_numeric_frame(test[selected]))

        model = make_model(model_name)
        sample_weight = compute_sample_weight_strategy(fit, y_fit, weight_strategy)
        fit_with_optional_weights(model, X_fit, y_fit, sample_weight=sample_weight)

        p_cal = get_proba(model, X_cal)
        t, cal_bacc = best_threshold(cal[target].values, p_cal, grid=THRESHOLD_GRID)
        p_test = get_proba(model, X_test)
        met = metrics_row(test[target].values, p_test, t)
        met.update({
            "block_id": bi,
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "horizon": horizon,
            "feature_set": feature_set_name,
            "model": model_name,
            "train_window_years": "all" if train_window_years is None else train_window_years,
            "weight_strategy": weight_strategy,
            "cal_bacc": cal_bacc,
            "n_features": len(selected),
            "n_global_features": sum(is_global(x) for x in selected),
        })
        results.append(met)

        pred = test[["fecha", "precio_oro", ret_col, target]].copy()
        pred = pred.rename(columns={ret_col: "real_return_h21" if horizon==21 else f"real_return_h{horizon}", target:"y_true"})
        pred["proba_up"] = p_test
        pred["threshold"] = t
        pred["pred_up_calibrated"] = (p_test >= t).astype(int)
        pred["block_id"] = bi
        pred["horizon"] = horizon
        pred["feature_set"] = feature_set_name
        pred["model"] = model_name
        pred["train_window_years"] = "all" if train_window_years is None else train_window_years
        pred["weight_strategy"] = weight_strategy
        all_preds.append(pred)

        for r, f in enumerate(selected[:250]):
            selected_features_rows.append({
                "block_id": bi, "horizon": horizon, "feature_set": feature_set_name,
                "model": model_name, "train_window_years": "all" if train_window_years is None else train_window_years,
                "weight_strategy": weight_strategy,
                "rank": r+1, "feature": f, "is_global": is_global(f)
            })

    return pd.DataFrame(results), pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(), pd.DataFrame(selected_features_rows)

# ============================================================
# 12. GRILLA OPTIMIZADA CON CONFIGURACIONES DE REFERENCIA
# ============================================================

# Esta lista reemplaza la grilla exhaustiva.
# Incluye:
# - configuraciones competitivas observadas;
# - auditorías sin oro directo / sin metales;
# - referencias no competitivas como Colombia-only;
# - algunos modelos simples para comparación.

SELECTED_MODEL_CONFIGS = [
    # --------------------------------------------------------
    # Núcleo competitivo observado en ejecuciones previas
    # --------------------------------------------------------
    {
        "tag": "competitivo_HGB_full_all_balanced",
        "horizon": 21,
        "feature_set": "colombia_plus_global_full",
        "model": "HistGradientBoostingClassifier",
        "train_window_years": None,
        "weight_strategy": "balanced",
        "max_features": MAX_FEATURES_FULL,
        "reason": "Muy competitivo en corrida previa; full global + balance de clases.",
    },
    {
        "tag": "competitivo_XGB_full_15_none",
        "horizon": 21,
        "feature_set": "colombia_plus_global_full",
        "model": "XGBClassifier",
        "train_window_years": 15,
        "weight_strategy": "none",
        "max_features": MAX_FEATURES_FULL,
        "reason": "Uno de los mejores XGB observados; ventana 15 años.",
    },
    {
        "tag": "competitivo_XGB_full_all_recency",
        "horizon": 21,
        "feature_set": "colombia_plus_global_full",
        "model": "XGBClassifier",
        "train_window_years": None,
        "weight_strategy": "recency_4y",
        "max_features": MAX_FEATURES_FULL,
        "reason": "Full global con énfasis en datos recientes.",
    },
    {
        "tag": "competitivo_XGB_full_15_cluster",
        "horizon": 21,
        "feature_set": "colombia_plus_global_full",
        "model": "XGBClassifier",
        "train_window_years": 15,
        "weight_strategy": "cluster_x4",
        "max_features": MAX_FEATURES_FULL,
        "reason": "Full global ponderando regímenes poco frecuentes.",
    },
    {
        "tag": "competitivo_HGB_full_all_none",
        "horizon": 21,
        "feature_set": "colombia_plus_global_full",
        "model": "HistGradientBoostingClassifier",
        "train_window_years": None,
        "weight_strategy": "none",
        "max_features": MAX_FEATURES_FULL,
        "reason": "Referencia fuerte sin ponderación.",
    },

    # --------------------------------------------------------
    # Configuración que antes había sido la mejor histórica
    # --------------------------------------------------------
    {
        "tag": "historico_XGB_compact_all_none",
        "horizon": 21,
        "feature_set": "colombia_plus_global_compact",
        "model": "XGBClassifier",
        "train_window_years": None,
        "weight_strategy": "none",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Configuración comparable al mejor resultado histórico reportado.",
    },
    {
        "tag": "compact_XGB_all_recency",
        "horizon": 21,
        "feature_set": "colombia_plus_global_compact",
        "model": "XGBClassifier",
        "train_window_years": None,
        "weight_strategy": "recency_4y",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Compacto global con énfasis reciente.",
    },

    # --------------------------------------------------------
    # Auditorías: evitar que el resultado dependa solo de oro internacional directo
    # --------------------------------------------------------
    {
        "tag": "auditoria_XGB_no_gold_direct_all_none",
        "horizon": 21,
        "feature_set": "global_no_gold_direct",
        "model": "XGBClassifier",
        "train_window_years": None,
        "weight_strategy": "none",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Auditoría sin GLD/IAU/GC directo.",
    },
    {
        "tag": "auditoria_HGB_no_gold_direct_all_balanced",
        "horizon": 21,
        "feature_set": "global_no_gold_direct",
        "model": "HistGradientBoostingClassifier",
        "train_window_years": None,
        "weight_strategy": "balanced",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Auditoría sin oro directo con modelo alternativo.",
    },
    {
        "tag": "auditoria_XGB_no_metals_all_none",
        "horizon": 21,
        "feature_set": "global_no_metals",
        "model": "XGBClassifier",
        "train_window_years": None,
        "weight_strategy": "none",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Auditoría sin metales; mide señal macro no metálica.",
    },
    {
        "tag": "auditoria_XGB_macro_only_all_none",
        "horizon": 21,
        "feature_set": "global_macro_only",
        "model": "XGBClassifier",
        "train_window_years": None,
        "weight_strategy": "none",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Contexto macro global sin depender de instrumentos metálicos.",
    },

    # --------------------------------------------------------
    # Referencias no competitivas / explicativas
    # --------------------------------------------------------
    {
        "tag": "referencia_colombia_only_XGB_all_recency",
        "horizon": 21,
        "feature_set": "colombia_only",
        "model": "XGBClassifier",
        "train_window_years": None,
        "weight_strategy": "recency_4y",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Referencia local: mide cuánto se logra sin contexto global.",
    },
    {
        "tag": "referencia_colombia_only_HGB_all_macrofull",
        "horizon": 21,
        "feature_set": "colombia_only",
        "model": "HistGradientBoostingClassifier",
        "train_window_years": None,
        "weight_strategy": "macrofull_x15",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Referencia local con ponderación macroestructural.",
    },
    {
        "tag": "referencia_RF_compact_15_macrofull",
        "horizon": 21,
        "feature_set": "colombia_plus_global_compact",
        "model": "RandomForestClassifier",
        "train_window_years": 15,
        "weight_strategy": "macrofull_x15",
        "max_features": MAX_FEATURES_COMPACT,
        "reason": "Referencia de bosque aleatorio; más lenta y generalmente menos competitiva.",
    },
    {
        "tag": "referencia_logit_compact_10_recency",
        "horizon": 21,
        "feature_set": "colombia_plus_global_compact",
        "model": "LogisticRegression",
        "train_window_years": 10,
        "weight_strategy": "recency_4y",
        "max_features": 900,
        "reason": "Referencia lineal/interpretable de menor capacidad.",
    },
]

selected_model_configs_df = pd.DataFrame(SELECTED_MODEL_CONFIGS)
selected_model_configs_df["regime_version"] = REGIME_VERSION
save_csv(selected_model_configs_df, "selected_model_configs.csv")
display(selected_model_configs_df[["tag", "feature_set", "model", "train_window_years", "weight_strategy", "reason"]])

# ============================================================
# CHECKPOINT / REANUDACIÓN
# ============================================================

PARTIAL_RESULTS_PATH = OUT / f"classification_results_partial_{REGIME_VERSION}.csv"
PARTIAL_PREDICTIONS_PATH = OUT / f"all_predictions_partial_{REGIME_VERSION}.csv"
PARTIAL_SELECTED_PATH = OUT / f"selected_features_by_run_partial_{REGIME_VERSION}.csv"
COMPLETED_CONFIGS_PATH = OUT / f"completed_configs_{REGIME_VERSION}.csv"

def config_key_from_dict(cfg):
    tw_label = "all" if cfg["train_window_years"] is None else str(cfg["train_window_years"])
    return f'h{cfg["horizon"]}__{cfg["feature_set"]}__{cfg["model"]}__win{tw_label}__w{cfg["weight_strategy"]}__{cfg["tag"]}'

def append_csv(df_new, path):
    if df_new is None or len(df_new) == 0:
        return
    df_new = df_new.copy()
    if path.exists():
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)

def load_partial_csv(path):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()

def write_checkpoint_zip():
    zip_tmp = OUT.parent / "dashboard_data_checkpoint.zip"
    if zip_tmp.exists():
        zip_tmp.unlink()
    with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    return zip_tmp

completed = set()
if COMPLETED_CONFIGS_PATH.exists():
    done_df = pd.read_csv(COMPLETED_CONFIGS_PATH)
    if "config_key" in done_df.columns:
        completed = set(done_df["config_key"].astype(str).tolist())

print("Configuraciones seleccionadas:", len(SELECTED_MODEL_CONFIGS))
print("Configuraciones ya completadas:", len(completed))

all_results = []
all_preds = []
all_selected = []

for cfg in SELECTED_MODEL_CONFIGS:
    key = config_key_from_dict(cfg)
    if key in completed:
        print("Skipping:", key)
        continue

    h = cfg["horizon"]
    fs_name = cfg["feature_set"]
    mname = cfg["model"]
    tw = cfg["train_window_years"]
    ws = cfg["weight_strategy"]
    max_feats = cfg["max_features"]
    feats = feature_sets[fs_name]

    print("Running:", cfg["tag"], "|", h, fs_name, mname, "win=", tw, "weight=", ws)
    try:
        res, preds, sel = run_walkforward_for_config(
            df, h, fs_name, feats, mname, tw,
            max_features=max_feats,
            weight_strategy=ws
        )

        if len(res):
            res = res.copy()
            preds = preds.copy()
            sel = sel.copy()

            for obj in [res, preds, sel]:
                obj["config_key"] = key
                obj["config_tag"] = cfg["tag"]
                obj["config_reason"] = cfg["reason"]

            all_results.append(res)
            all_preds.append(preds)
            all_selected.append(sel)

            append_csv(res, PARTIAL_RESULTS_PATH)
            append_csv(preds, PARTIAL_PREDICTIONS_PATH)
            append_csv(sel, PARTIAL_SELECTED_PATH)

            pd.DataFrame([{
                "config_key": key,
                "config_tag": cfg["tag"],
                "reason": cfg["reason"],
                "horizon": h,
                "feature_set": fs_name,
                "model": mname,
                "train_window_years": "all" if tw is None else tw,
                "weight_strategy": ws,
                "mean_balanced_accuracy": float(res["balanced_accuracy"].mean()),
                "mean_accuracy": float(res["accuracy"].mean()),
                "mean_auc": float(res["roc_auc"].mean()),
                "finished_at": datetime.now().isoformat(),
            }]).to_csv(
                COMPLETED_CONFIGS_PATH,
                mode="a",
                header=not COMPLETED_CONFIGS_PATH.exists(),
                index=False
            )
            # Copia genérica para dashboard/inspección rápida
            try:
                pd.read_csv(COMPLETED_CONFIGS_PATH).to_csv(OUT / "completed_configs.csv", index=False)
            except Exception:
                pass

            completed.add(key)
            print("  ok", res["balanced_accuracy"].mean())

            # En esta versión optimizada se guarda checkpoint después de CADA configuración.
            zpath = write_checkpoint_zip()
            print("  checkpoint:", zpath)
        else:
            print("  sin resultados")

    except Exception as e:
        print("  ERROR:", e)

# Consolidar desde los CSV parciales, no solo desde memoria.
classification_results = load_partial_csv(PARTIAL_RESULTS_PATH)
all_predictions = load_partial_csv(PARTIAL_PREDICTIONS_PATH)
selected_features = load_partial_csv(PARTIAL_SELECTED_PATH)

# Guardar nombres finales compatibles con el resto del cuaderno.
save_csv(classification_results, "classification_results.csv")
save_csv(all_predictions, "all_predictions.csv")
save_csv(selected_features, "selected_features_by_run.csv")

# Copias genéricas de parciales para facilitar recuperación manual.
classification_results.to_csv(OUT / "classification_results_partial.csv", index=False)
all_predictions.to_csv(OUT / "all_predictions_partial.csv", index=False)
selected_features.to_csv(OUT / "selected_features_by_run_partial.csv", index=False)

write_checkpoint_zip()
classification_results.head()


# %% [markdown]
# ## 12A. Celda de rescate manual


# %% Cell 31
# ============================================================
# 12A. RESCATE MANUAL DE RESULTADOS PARCIALES
# ============================================================

def rescue_partial_model_outputs():
    partial_results_path = OUT / f"classification_results_partial_{REGIME_VERSION}.csv"
    partial_predictions_path = OUT / f"all_predictions_partial_{REGIME_VERSION}.csv"
    partial_selected_path = OUT / f"selected_features_by_run_partial_{REGIME_VERSION}.csv"

    rescued = []

    if "all_results" in globals() and isinstance(all_results, list) and len(all_results):
        tmp = pd.concat(all_results, ignore_index=True)
        tmp.to_csv(partial_results_path, index=False)
        tmp.to_csv(OUT / "classification_results.csv", index=False)
        rescued.append(("classification_results", tmp.shape))

    if "all_preds" in globals() and isinstance(all_preds, list) and len(all_preds):
        tmp = pd.concat(all_preds, ignore_index=True)
        tmp.to_csv(partial_predictions_path, index=False)
        tmp.to_csv(OUT / "all_predictions.csv", index=False)
        rescued.append(("all_predictions", tmp.shape))

    if "all_selected" in globals() and isinstance(all_selected, list) and len(all_selected):
        tmp = pd.concat(all_selected, ignore_index=True)
        tmp.to_csv(partial_selected_path, index=False)
        tmp.to_csv(OUT / "selected_features_by_run.csv", index=False)
        rescued.append(("selected_features", tmp.shape))

    # reconstruir completed_configs desde classification_results si existe
    cr_path = OUT / "classification_results.csv"
    if cr_path.exists():
        cr = pd.read_csv(cr_path)
        group_cols = ["horizon", "feature_set", "model", "train_window_years", "weight_strategy"]
        if all(c in cr.columns for c in group_cols):
            done = cr[group_cols].drop_duplicates().copy()
            done["config_key"] = (
                "h" + done["horizon"].astype(str)
                + "__" + done["feature_set"].astype(str)
                + "__" + done["model"].astype(str)
                + "__win" + done["train_window_years"].astype(str)
                + "__w" + done["weight_strategy"].astype(str)
            )
            done.to_csv(OUT / "completed_configs.csv", index=False)
            rescued.append(("completed_configs", done.shape))

    zip_tmp = OUT.parent / "dashboard_data_checkpoint.zip"
    if zip_tmp.exists():
        zip_tmp.unlink()
    with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))

    print("Rescatado:", rescued)
    print("ZIP checkpoint:", zip_tmp)
    return zip_tmp

checkpoint_zip = rescue_partial_model_outputs()


# %% [markdown]
# ## 12B. Reglas determinísticas de comparación


# %% Cell 33
# ============================================================
# 12B. BASELINES DETERMINÍSTICOS
# ============================================================

def deterministic_signal(data, rule_name):
    d = data.copy()
    price = pd.to_numeric(d["precio_oro"], errors="coerce")
    if rule_name == "Momentum_21":
        return (np.log(price / price.shift(21)) > 0).astype(float)
    if rule_name == "Momentum_63":
        return (np.log(price / price.shift(63)) > 0).astype(float)
    if rule_name == "MA_21_63":
        return (price.rolling(21, min_periods=10).mean() > price.rolling(63, min_periods=20).mean()).astype(float)
    if rule_name == "MA_63_126":
        return (price.rolling(63, min_periods=20).mean() > price.rolling(126, min_periods=40).mean()).astype(float)
    if rule_name == "TRM_Momentum_21" and "trm" in d.columns:
        trm = pd.to_numeric(d["trm"], errors="coerce")
        return (np.log(trm / trm.shift(21)) > 0).astype(float)
    if rule_name == "Brent_Momentum_21" and "precio_brent" in d.columns:
        brent = pd.to_numeric(d["precio_brent"], errors="coerce")
        return (np.log(brent / brent.shift(21)) > 0).astype(float)
    return pd.Series(np.nan, index=d.index)

def run_deterministic_baselines(data, horizons):
    rules = ["Momentum_21", "Momentum_63", "MA_21_63", "MA_63_126", "TRM_Momentum_21", "Brent_Momentum_21"]
    res_rows, pred_rows = [], []
    for h in horizons:
        target = f"target_up_h{h}"
        ret_col = f"future_logret_h{h}"
        d = data.dropna(subset=[target, ret_col]).copy()
        blocks = make_walkforward_blocks(d, start_test="2020-05-01", block_months=6 if not FAST_MODE else 12)
        for rule in rules:
            sig = deterministic_signal(d, rule)
            for bi, (test_start, test_end) in enumerate(blocks):
                test = d[(d["fecha"] >= test_start) & (d["fecha"] < test_end)].copy()
                if len(test) < 20:
                    continue
                idx = test.index
                pred = sig.loc[idx].copy()
                valid = pred.notna() & test[target].notna()
                if valid.sum() < 20:
                    continue
                y_true = test.loc[valid, target].astype(int).values
                y_pred = pred.loc[valid].astype(int).values
                proba = y_pred.astype(float)
                met = metrics_row(y_true, proba, 0.5)
                met.update({
                    "block_id": bi,
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "horizon": h,
                    "feature_set": "deterministic_baseline",
                    "model": rule,
                    "train_window_years": "deterministic",
                    "weight_strategy": "deterministic",
                    "cal_bacc": np.nan,
                    "n_features": 1,
                    "n_global_features": 0,
                })
                res_rows.append(met)

                pr = test.loc[valid, ["fecha", "precio_oro", ret_col, target]].copy()
                pr = pr.rename(columns={ret_col: "real_return_h21" if h == 21 else f"real_return_h{h}", target: "y_true"})
                pr["proba_up"] = proba
                pr["threshold"] = 0.5
                pr["pred_up_calibrated"] = y_pred
                pr["block_id"] = bi
                pr["horizon"] = h
                pr["feature_set"] = "deterministic_baseline"
                pr["model"] = rule
                pr["train_window_years"] = "deterministic"
                pr["weight_strategy"] = "deterministic"
                pred_rows.append(pr)
    return pd.DataFrame(res_rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

det_results, det_predictions = run_deterministic_baselines(df, HORIZONS)
save_csv(det_results, "deterministic_baseline_results.csv")

if not det_results.empty:
    det_summary = det_results.groupby(["horizon", "feature_set", "model", "train_window_years", "weight_strategy"]).agg(
        mean_accuracy=("accuracy","mean"),
        std_accuracy=("accuracy","std"),
        mean_balanced_accuracy=("balanced_accuracy","mean"),
        std_balanced_accuracy=("balanced_accuracy","std"),
        mean_roc_auc=("roc_auc","mean"),
        mean_precision_up=("precision_up","mean"),
        mean_recall_up=("recall_up","mean"),
        mean_recall_not_up=("recall_not_up","mean"),
        mean_f1_up=("f1_up","mean"),
        mean_mcc=("mcc","mean"),
        mean_threshold=("threshold","mean"),
        mean_n_features=("n_features","mean"),
        mean_n_global_features=("n_global_features","mean"),
        total_tp=("tp","sum"),
        total_tn=("tn","sum"),
        total_fp=("fp","sum"),
        total_fn=("fn","sum"),
        n_blocks=("block_id","nunique"),
        n_test_total=("n_test","sum"),
    ).reset_index()
else:
    det_summary = pd.DataFrame()

save_csv(det_summary, "deterministic_baseline_summary.csv")

# Integrar al flujo principal para que el comparador del dashboard los incluya.
if not det_results.empty:
    classification_results = pd.concat([classification_results, det_results], ignore_index=True)
    save_csv(classification_results, "classification_results.csv")
if not det_predictions.empty:
    all_predictions = pd.concat([all_predictions, det_predictions], ignore_index=True)
    save_csv(all_predictions, "all_predictions.csv")

det_summary.sort_values(["horizon", "mean_balanced_accuracy"], ascending=[True, False]).head(20)


# %% [markdown]
# ## 13. Resumen comparativo de modelos


# %% Cell 35
# ============================================================
# 13. SUMMARY DE CLASIFICACIÓN
# ============================================================

if classification_results.empty:
    raise ValueError("No hay resultados de clasificación. Revisa pasos anteriores.")

group_cols = ["horizon", "feature_set", "model", "train_window_years", "weight_strategy"]
summary = classification_results.groupby(group_cols).agg(
    mean_accuracy=("accuracy","mean"),
    std_accuracy=("accuracy","std"),
    mean_balanced_accuracy=("balanced_accuracy","mean"),
    std_balanced_accuracy=("balanced_accuracy","std"),
    mean_roc_auc=("roc_auc","mean"),
    mean_precision_up=("precision_up","mean"),
    mean_recall_up=("recall_up","mean"),
    mean_recall_not_up=("recall_not_up","mean"),
    mean_f1_up=("f1_up","mean"),
    mean_mcc=("mcc","mean"),
    mean_threshold=("threshold","mean"),
    mean_n_features=("n_features","mean"),
    mean_n_global_features=("n_global_features","mean"),
    total_tp=("tp","sum"),
    total_tn=("tn","sum"),
    total_fp=("fp","sum"),
    total_fn=("fn","sum"),
    n_blocks=("block_id","nunique"),
    n_test_total=("n_test","sum"),
).reset_index()

summary = summary.sort_values(["horizon", MODEL_SELECTION_METRIC, "mean_roc_auc"], ascending=[True, False, False])
save_csv(summary, "classification_summary.csv")
summary.head(20)


# %% [markdown]
# ## 14. Mejor modelo y archivos del dashboard


# %% Cell 37
# ============================================================
# 14. MEJOR MODELO Y PREDICCIONES
# ============================================================

best_row = summary[summary["horizon"] == MAIN_HORIZON].iloc[0].copy()
print("Mejor configuración:")
display(best_row.to_frame().T)

def mask_same_config(table, row):
    m = pd.Series(True, index=table.index)
    if "horizon" in table.columns:
        m &= table["horizon"].astype(int).eq(int(row["horizon"]))
    if "feature_set" in table.columns:
        m &= table["feature_set"].astype(str).eq(str(row["feature_set"]))
    if "model" in table.columns:
        m &= table["model"].astype(str).eq(str(row["model"]))
    if "train_window_years" in table.columns:
        m &= table["train_window_years"].astype(str).eq(str(row["train_window_years"]))
    if "weight_strategy" in table.columns and "weight_strategy" in row.index:
        m &= table["weight_strategy"].astype(str).eq(str(row.get("weight_strategy", "none")))
    return m

best_mask = mask_same_config(all_predictions, best_row)
best_preds = all_predictions.loc[best_mask].copy().sort_values("fecha")

if best_preds.empty:
    raise ValueError("No se encontraron predicciones para la mejor configuración.")

# Normalizar nombre retorno principal para dashboard
if "real_return_h21" not in best_preds.columns:
    ret_cols = [c for c in best_preds.columns if c.startswith("real_return")]
    if ret_cols:
        best_preds["real_return_h21"] = best_preds[ret_cols[0]]

# Asegurar predicción calibrada
if "pred_up_calibrated" not in best_preds.columns:
    best_preds["pred_up_calibrated"] = (best_preds["proba_up"] >= best_preds["threshold"]).astype(int)

# Retorno esperado preliminar: se reemplaza luego si la capa de magnitud corre bien.
vol_h21 = df["future_logret_h21"].std() if "future_logret_h21" in df.columns else best_preds["real_return_h21"].std()
best_preds["pred_return_direct"] = (best_preds["proba_up"] - 0.5) * 2 * vol_h21
best_preds["expected_price_h21"] = best_preds["precio_oro"] * np.exp(best_preds["pred_return_direct"])

save_csv(best_preds, "predictions.csv")
save_csv(best_preds, "best_predictions.csv")

cm = confusion_matrix(best_preds["y_true"].astype(int), best_preds["pred_up_calibrated"].astype(int), labels=[0, 1])
cm_df = pd.DataFrame(cm, index=["real_no_sube", "real_sube"], columns=["pred_no_sube", "pred_sube"]).reset_index().rename(columns={"index": "real"})
save_csv(cm_df, "confusion_matrix.csv")

print(cm_df)


# %% [markdown]
# ## 15. Magnitud auxiliar


# %% [markdown]
# ## Corrección de fecha en la capa de magnitud


# %% Cell 40
# ============================================================
# 15. MAGNITUD AUXILIAR
# ============================================================

def run_magnitude_for_best(data, best_row, feature_sets):
    h = int(best_row["horizon"])
    fs_name = best_row["feature_set"]
    feats = feature_sets[fs_name]
    ret_col = f"future_logret_h{h}"
    target = f"target_up_h{h}"
    d = data.dropna(subset=[ret_col, target]).copy()
    blocks = make_walkforward_blocks(d, start_test="2020-05-01", block_months=6 if not FAST_MODE else 12)

    rows, preds_all = [], []
    for bi, (test_start, test_end) in enumerate(blocks):
        train = d[d["fecha"] < test_start].copy()
        test = d[(d["fecha"] >= test_start) & (d["fecha"] < test_end)].copy()
        if len(train) < 800 or len(test) < 20:
            continue
        candidate_features = [f for f in feats if f in d.columns]
        selected = select_top_features_by_train(train[candidate_features], train[target].astype(int), candidate_features, max_features=1000)

        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(sanitize_numeric_frame(train[selected]))
        X_test = imp.transform(sanitize_numeric_frame(test[selected]))
        y_train = train[ret_col].values
        y_test = test[ret_col].values

        if USE_XGBOOST:
            reg = XGBRegressor(
                n_estimators=250 if not FAST_MODE else 80,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            reg = RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=4)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)

        rows.append({
            "block_id": bi,
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "feature_set": fs_name,
            "model": "XGBRegressor" if USE_XGBOOST else "RandomForestRegressor",
            "magnitude_method": "direct_return_regression",
            "rmse_return": math.sqrt(mean_squared_error(y_test, pred)),
            "mae_return": mean_absolute_error(y_test, pred),
            "corr_return": np.corrcoef(y_test, pred)[0,1] if len(y_test) > 1 else np.nan,
            "direction_bacc_from_return": balanced_accuracy_score((y_test>0).astype(int), (pred>0).astype(int)),
            "direction_acc_from_return": accuracy_score((y_test>0).astype(int), (pred>0).astype(int)),
            "n_test": len(test),
            "n_features": len(selected),
        })
        p = test[["fecha", "precio_oro", ret_col]].copy()
        p = p.rename(columns={ret_col: "real_return_h21"})
        p["pred_return_direct"] = pred
        p["expected_price_h21"] = p["precio_oro"] * np.exp(pred)
        p["block_id"] = bi
        preds_all.append(p)
    return pd.DataFrame(rows), pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()

mag_results, mag_preds = run_magnitude_for_best(df, best_row, feature_sets)
if not mag_results.empty:
    mag_summary = mag_results.groupby(["feature_set", "model", "magnitude_method"]).agg(
        mean_rmse_return=("rmse_return","mean"),
        mean_mae_return=("mae_return","mean"),
        mean_corr_return=("corr_return","mean"),
        mean_direction_bacc_from_return=("direction_bacc_from_return","mean"),
        mean_direction_acc_from_return=("direction_acc_from_return","mean"),
        n_blocks=("block_id","nunique"),
        n_test_total=("n_test","sum")
    ).reset_index()
else:
    mag_summary = pd.DataFrame()

save_csv(mag_results, "magnitude_results.csv")
save_csv(mag_summary, "magnitude_summary.csv")
save_csv(mag_preds, "magnitude_predictions.csv")

# Actualizar predictions.csv con retorno predicho real si está disponible
if not mag_preds.empty:
    best_preds = best_preds.copy()
    mag_preds = mag_preds.copy()
    best_preds["fecha"] = pd.to_datetime(best_preds["fecha"], errors="coerce")
    mag_preds["fecha"] = pd.to_datetime(mag_preds["fecha"], errors="coerce")
    mag_merge = (
        mag_preds[["fecha", "pred_return_direct", "expected_price_h21"]]
        .dropna(subset=["fecha"])
        .drop_duplicates(subset=["fecha"], keep="last")
    )
    bp = (
        best_preds
        .drop(columns=["pred_return_direct", "expected_price_h21"], errors="ignore")
        .merge(mag_merge, on="fecha", how="left", validate="many_to_one")
    )
    bp["pred_return_direct"] = bp["pred_return_direct"].fillna((bp["proba_up"] - 0.5) * 2 * vol_h21)
    bp["expected_price_h21"] = bp["expected_price_h21"].fillna(bp["precio_oro"] * np.exp(bp["pred_return_direct"]))
    save_csv(bp, "predictions.csv")
    best_preds = bp.copy()

mag_summary


# %% [markdown]
# ## 16. Análisis por segmentos


# %% Cell 42
# ============================================================
# 16. SEGMENTOS
# ============================================================

seg_base = best_preds.merge(
    df[["fecha", "cluster_abs", "cluster_chg", "policy_stance", "dd_lag1_dd_clasificacion", "year"]].drop_duplicates("fecha"),
    on="fecha", how="left"
)

segment_cols = ["cluster_abs", "cluster_chg", "policy_stance", "dd_lag1_dd_clasificacion", "year"]
seg_rows = []
for scol in segment_cols:
    if scol not in seg_base.columns:
        continue
    for seg, g in seg_base.groupby(scol, dropna=False):
        if len(g) < 20:
            continue
        try:
            auc = roc_auc_score(g["y_true"], g["proba_up"]) if g["y_true"].nunique() > 1 else np.nan
        except Exception:
            auc = np.nan
        m = metrics_row(g["y_true"].astype(int), g["proba_up"], g["threshold"].mean())
        seg_rows.append({
            "segment_col": scol,
            "segment": str(seg),
            "n": len(g),
            "positive_share": g["y_true"].mean(),
            "accuracy": accuracy_score(g["y_true"], g["pred_up_calibrated"]),
            "balanced_accuracy": balanced_accuracy_score(g["y_true"], g["pred_up_calibrated"]) if g["y_true"].nunique() > 1 else np.nan,
            "recall_up": recall_score(g["y_true"], g["pred_up_calibrated"], zero_division=0),
            "recall_not_up": recall_score(1-g["y_true"], 1-g["pred_up_calibrated"], zero_division=0),
            "roc_auc": auc,
        })

segment_metrics = pd.DataFrame(seg_rows).sort_values("balanced_accuracy", ascending=False)
save_csv(segment_metrics, "segment_metrics.csv")
segment_metrics.head(20)


# %% [markdown]
# ## 17. Simulador de inversión


# %% Cell 44
# ============================================================
# 17. BACKTEST DE ESTRATEGIAS
# ============================================================

def simulate_strategies(preds, capital_initial=1_000_000, start_date="2022-08-07", end_date=None, cost=0.0, slippage=0.0):
    p = preds.copy()
    p["fecha"] = pd.to_datetime(p["fecha"])
    p = p.sort_values("fecha")
    if end_date is None:
        end_date = p["fecha"].max()
    p = p[(p["fecha"] >= pd.Timestamp(start_date)) & (p["fecha"] <= pd.Timestamp(end_date))].copy()

    # Retorno diario real desde precio oro
    p["ret1"] = p["precio_oro"].pct_change().shift(-1)
    p = p.iloc[:-1].copy()

    # Exposiciones
    p["buy_hold"] = 1.0
    p["clf_daily_long_cash"] = (p["proba_up"] >= p["threshold"]).astype(float)

    # h21 rebalance
    cur_lc, cur_ls = 0.0, -1.0
    exp_lc, exp_ls = [], []
    for i, row in p.iterrows():
        # cada 21 observaciones de predicción
        if len(exp_lc) % 21 == 0:
            cur_lc = 1.0 if row["pred_up_calibrated"] == 1 else 0.0
            cur_ls = 1.0 if row["pred_up_calibrated"] == 1 else -1.0
        exp_lc.append(cur_lc)
        exp_ls.append(cur_ls)
    p["h21_long_cash"] = exp_lc
    p["h21_long_short"] = exp_ls

    # Histeresis
    exp, cur = [], 0.0
    for prob in p["proba_up"]:
        if prob >= 0.60:
            cur = 1.0
        elif prob <= 0.35:
            cur = 0.0
        exp.append(cur)
    p["hysteresis_60_35"] = exp

    # Retorno predicho
    p["predret_long_cash"] = (p["pred_return_direct"] > 0).astype(float)
    p["clf_daily_long_short"] = np.where(p["proba_up"] >= p["threshold"], 1.0, -1.0)
    p["predret_long_short"] = np.where(p["pred_return_direct"] > 0, 1.0, -1.0)

    strategies = ["buy_hold","h21_long_cash","clf_daily_long_cash","hysteresis_60_35","predret_long_cash",
                  "h21_long_short","clf_daily_long_short","predret_long_short"]

    out_rows = []
    summary_rows = []
    years = (p["fecha"].max() - p["fecha"].min()).days / 365.25

    for s in strategies:
        exposure = p[s].fillna(0).values
        ret = p["ret1"].fillna(0).values
        turnover = np.abs(np.diff(np.r_[0, exposure]))
        net_ret = exposure * ret - turnover * cost - turnover * slippage
        capital = capital_initial * np.cumprod(1 + net_ret)
        running_max = np.maximum.accumulate(capital)
        dd = capital / running_max - 1

        tmp = p[["fecha","precio_oro","proba_up","threshold","pred_return_direct"]].copy()
        tmp["strategy"] = s
        tmp["exposure"] = exposure
        tmp["ret1"] = ret
        tmp["turnover"] = turnover
        tmp["net_return"] = net_ret
        tmp["capital"] = capital
        tmp["drawdown"] = dd
        out_rows.append(tmp)

        total_return = capital[-1] / capital_initial - 1 if len(capital) else np.nan
        cagr = (capital[-1] / capital_initial) ** (1 / years) - 1 if years > 0 and len(capital) else np.nan
        summary_rows.append({
            "strategy": s,
            "capital_initial": capital_initial,
            "capital_final": capital[-1] if len(capital) else np.nan,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": dd.min() if len(dd) else np.nan,
            "hit_rate": (net_ret > 0).mean(),
            "time_in_market": np.mean(np.abs(exposure)),
            "n_position_changes": int((turnover > 0).sum()),
            "cost": cost,
            "slippage": slippage,
            "start_date": p["fecha"].min().date().isoformat(),
            "end_date": p["fecha"].max().date().isoformat(),
        })
    return pd.concat(out_rows, ignore_index=True), pd.DataFrame(summary_rows)

strategy_ts, strategy_summary = simulate_strategies(best_preds, cost=0.0, slippage=0.0)
save_csv(strategy_ts, "strategy_backtest_timeseries.csv")
save_csv(strategy_summary, "strategy_backtest_summary.csv")

strategy_summary.sort_values("capital_final", ascending=False)


# %% Cell 45
# ============================================================
# 17.1 ESTRATEGIAS POR RÉGIMEN DE MERCADO 6M
# ============================================================

def market_regime_analysis(strategy_ts):
    # usar buy_hold para clasificar ventanas
    bh = strategy_ts[strategy_ts["strategy"]=="buy_hold"].copy()
    bh = bh.sort_values("fecha")
    rows = []
    cur = bh["fecha"].min().normalize()
    end_all = bh["fecha"].max()
    strategies = strategy_ts["strategy"].unique()

    while cur < end_all:
        end = cur + pd.DateOffset(months=6)
        g_bh = bh[(bh["fecha"] >= cur) & (bh["fecha"] < end)].copy()
        if len(g_bh) < 20:
            cur = end
            continue
        bh_ret = (1 + g_bh["ret1"]).prod() - 1
        if bh_ret > 0.10:
            regime = "alcista_fuerte"
        elif bh_ret > 0.05:
            regime = "alcista_moderado"
        elif bh_ret >= -0.05:
            regime = "lateral"
        else:
            regime = "bajista"

        row = {
            "inicio": g_bh["fecha"].min().date().isoformat(),
            "fin": g_bh["fecha"].max().date().isoformat(),
            "regimen_mercado": regime,
            "oro_buyhold_return": bh_ret,
            "n_dias": len(g_bh),
        }
        for s in strategies:
            g = strategy_ts[(strategy_ts["strategy"]==s) & (strategy_ts["fecha"] >= cur) & (strategy_ts["fecha"] < end)]
            row[s] = (1 + g["net_return"]).prod() - 1 if len(g) else np.nan
        rows.append(row)
        cur = end

    per = pd.DataFrame(rows)
    agg = per.groupby("regimen_mercado").agg({**{"oro_buyhold_return":"mean"}, **{s:"mean" for s in strategies}}).reset_index()
    counts = per.groupby("regimen_mercado").size().reset_index(name="n_periodos")
    agg = counts.merge(agg, on="regimen_mercado", how="left")
    return per, agg

market_periods, market_summary = market_regime_analysis(strategy_ts)
save_csv(market_periods, "market_regime_strategy_periods.csv")
save_csv(market_summary, "market_regime_strategy_summary.csv")
market_summary


# %% [markdown]
# ## 18. Auditoría anti-fuga


# %% Cell 47
# ============================================================
# 18. AUDITORÍA ANTI-FUGA
# ============================================================

main = summary[(summary["horizon"]==MAIN_HORIZON)].copy()
best_bacc = float(best_row["mean_balanced_accuracy"])

audit_sets = ["colombia_only", "colombia_plus_global_compact", "global_no_gold_direct", "global_no_metals", "global_macro_only"]
audit_rows = []
for fs in audit_sets:
    sub = main[main["feature_set"] == fs].sort_values("mean_balanced_accuracy", ascending=False)
    if sub.empty:
        continue
    r = sub.iloc[0]
    delta = float(r["mean_balanced_accuracy"] - best_bacc)
    if fs == "colombia_plus_global_compact":
        status = "referencia"
    elif r["mean_balanced_accuracy"] >= best_bacc - 0.03:
        status = "verde_se_sostiene"
    elif r["mean_balanced_accuracy"] >= best_bacc - 0.10:
        status = "amarillo_baja_moderada"
    else:
        status = "rojo_dependencia_alta"
    audit_rows.append({
        "audit_case": fs,
        "best_model": r["model"],
        "train_window_years": r["train_window_years"],
        "mean_accuracy": r["mean_accuracy"],
        "mean_balanced_accuracy": r["mean_balanced_accuracy"],
        "mean_roc_auc": r["mean_roc_auc"],
        "delta_bacc_vs_best": delta,
        "status": status,
        "interpretacion": {
            "colombia_only": "Base local sin contexto global.",
            "colombia_plus_global_compact": "Modelo global compacto original.",
            "global_no_gold_direct": "Excluye GLD, IAU y GC=F.",
            "global_no_metals": "Excluye oro, plata y metales cercanos.",
            "global_macro_only": "Conserva solo macro global: dólar, VIX, tasas, equity, petróleo."
        }.get(fs, "")
    })
audit_summary = pd.DataFrame(audit_rows)
save_csv(audit_summary, "audit_summary.csv")
audit_summary


# %% [markdown]
# ## 19. Explicabilidad


# %% Cell 49
# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

def feature_group(f):
    x = f.lower()
    if any(k in x for k in ["gld", "iau", "gc_f"]):
        return "oro_internacional"
    if any(k in x for k in ["slv", "si_f", "pl_f", "pa_f", "hg_f"]):
        return "metales"
    if "dx_y_nyb" in x:
        return "dolar"
    if "vix" in x:
        return "riesgo"
    if any(k in x for k in ["tnx", "irx"]):
        return "tasas"
    if any(k in x for k in ["spy", "qqq"]):
        return "renta_variable"
    if "cl_f" in x or "brent" in x:
        return "energia"
    if "cluster" in x:
        return "regimen"
    if "dd_" in x or "holand" in x:
        return "enfermedad_holandesa"
    return "colombia"

try:
    h = MAIN_HORIZON
    fs_name = best_row["feature_set"]
    model_name = best_row["model"]
    feats = feature_sets[fs_name]
    target = f"target_up_h{h}"
    d = df.dropna(subset=[target]).copy()
    test_start = pd.Timestamp("2024-01-01")
    train = d[d["fecha"] < test_start].copy()
    test = d[d["fecha"] >= test_start].copy()
    if len(test) < 100:
        test_start = d["fecha"].quantile(0.80)
        train = d[d["fecha"] < test_start].copy()
        test = d[d["fecha"] >= test_start].copy()
    feats = [f for f in feats if f in d.columns]
    selected = select_top_features_by_train(train[feats], train[target].astype(int), feats, max_features=500)
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(sanitize_numeric_frame(train[selected]))
    X_test = imp.transform(sanitize_numeric_frame(test[selected]))
    y_train = train[target].astype(int).values
    y_test = test[target].astype(int).values
    model = make_model(model_name)
    model.fit(X_train, y_train)

    # Permutation importance
    pi = permutation_importance(
        model, X_test, y_test,
        scoring="balanced_accuracy",
        n_repeats=5 if not FAST_MODE else 2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    fi = pd.DataFrame({
        "feature": selected,
        "importance": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance", ascending=False)
    fi["rank"] = np.arange(1, len(fi)+1)
    fi["group"] = fi["feature"].map(feature_group)
    fi["model"] = model_name
    fi["feature_set"] = fs_name
    save_csv(fi, "feature_importance.csv")
    display(fi.head(30))
except Exception as e:
    print("No se pudo calcular permutation importance:", e)
    fi = pd.DataFrame(columns=["feature","importance","importance_std","rank","group","model","feature_set"])
    save_csv(fi, "feature_importance.csv")


# %% [markdown]
# ## 20. Guardar el mejor modelo completo


# %% Cell 51
# ============================================================
# 20. GUARDAR MODELO FINAL
# ============================================================

if SAVE_MODEL:
    import joblib

    h = int(best_row["horizon"])
    fs_name = str(best_row["feature_set"])
    model_name = str(best_row["model"])
    target = f"target_up_h{h}"
    ret_col = f"future_logret_h{h}"

    d = df.dropna(subset=[target, ret_col]).copy()
    feats = [f for f in feature_sets[fs_name] if f in d.columns]

    max_feats_final = MAX_FEATURES_FULL if fs_name == "colombia_plus_global_full" else MAX_FEATURES_COMPACT
    selected = select_top_features_by_train(
        sanitize_numeric_frame(d[feats]),
        d[target].astype(int).values,
        feats,
        max_features=max_feats_final
    )

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(sanitize_numeric_frame(d[selected]))
    y = d[target].astype(int).values

    model = make_model(model_name)
    sample_weight = compute_sample_weight_strategy(d, y, str(best_row.get("weight_strategy", "none")))
    fit_with_optional_weights(model, X, y, sample_weight=sample_weight)

    bundle = {
        "model": model,
        "imputer": imputer,
        "features": selected,
        "config": best_row.to_dict(),
        "mean_threshold": float(best_row.get("mean_threshold", best_preds["threshold"].mean())),
        "regime_version": REGIME_VERSION,
        "regime_design": pd.read_csv(OUT / "regime_design_6_7.csv").to_dict(orient="records") if (OUT / "regime_design_6_7.csv").exists() else [],
        "created_at": datetime.now().isoformat(),
    }

    model_path = MODELS / "best_model_bundle.joblib"
    joblib.dump(bundle, model_path)
    print("Modelo guardado:", model_path)

    # Para XGBoost también se guarda el booster en formato JSON.
    if model_name == "XGBClassifier":
        try:
            booster_path = MODELS / "best_xgb_booster.json"
            model.get_booster().save_model(str(booster_path))
            print("Booster XGBoost guardado:", booster_path)
        except Exception as e:
            print("No se pudo guardar booster XGBoost separado:", e)

    config_json = {
        "horizon": int(best_row["horizon"]),
        "feature_set": str(best_row["feature_set"]),
        "model": str(best_row["model"]),
        "train_window_years": str(best_row["train_window_years"]),
        "weight_strategy": str(best_row.get("weight_strategy", "none")),
        "mean_threshold": float(best_row.get("mean_threshold", best_preds["threshold"].mean())),
        "mean_balanced_accuracy": float(best_row.get("mean_balanced_accuracy", np.nan)),
        "mean_accuracy": float(best_row.get("mean_accuracy", np.nan)),
        "mean_roc_auc": float(best_row.get("mean_roc_auc", np.nan)),
        "n_features_final": len(selected),
        "regime_version": REGIME_VERSION,
        "model_file": "best_model_bundle.joblib",
    }
    with open(MODELS / "best_model_config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)

    # Catálogo de features usadas por el modelo final
    best_model_features = pd.DataFrame({
        "rank": range(1, len(selected) + 1),
        "feature": selected,
        "is_global": [is_global(x) for x in selected],
        "feature_group": [feature_group(x) if "feature_group" in globals() else ("global" if is_global(x) else "colombia") for x in selected],
    })
    save_csv(best_model_features, "best_model_features.csv")
else:
    print("SAVE_MODEL=False: no se guardó modelo final.")


# %% [markdown]
# ## 21. Gráficas finales para el dashboard


# %% Cell 53
# ============================================================
# 21. FIGURAS
# ============================================================

plt.rcParams.update({"figure.figsize": (10,5), "axes.grid": True})

# Top modelos
top = summary[summary["horizon"]==MAIN_HORIZON].head(20).copy()
top["label"] = top["feature_set"] + "\n" + top["model"] + " · win=" + top["train_window_years"].astype(str)
plt.figure(figsize=(11,7))
plt.barh(top["label"][::-1], top["mean_balanced_accuracy"][::-1]*100)
plt.xlabel("Balanced accuracy media (%)")
plt.title("Top 20 modelos walk-forward h21")
plt.tight_layout()
plt.savefig(FIG / "top20_models_bacc.png", dpi=180)
plt.close()

# Confusion matrix
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation="nearest")
plt.title("Matriz de confusión · mejor modelo")
plt.xticks([0,1], ["Pred no sube", "Pred sube"])
plt.yticks([0,1], ["Real no sube", "Real sube"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrix_best.png", dpi=180)
plt.close()

# Estrategias capital
plt.figure(figsize=(11,5))
for s, g in strategy_ts.groupby("strategy"):
    if s in ["buy_hold", "h21_long_cash", "hysteresis_60_35", "predret_long_cash", "predret_long_short"]:
        plt.plot(g["fecha"], g["capital"], label=s)
plt.legend()
plt.title("Simulación de capital por estrategia")
plt.ylabel("Capital COP")
plt.tight_layout()
plt.savefig(FIG / "strategy_capital_curves.png", dpi=180)
plt.close()

# Auditoría
if not audit_summary.empty:
    plt.figure(figsize=(9,4))
    plt.bar(audit_summary["audit_case"], audit_summary["mean_balanced_accuracy"]*100)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("BACC media (%)")
    plt.title("Auditoría anti-fuga por feature set")
    plt.tight_layout()
    plt.savefig(FIG / "audit_bacc.png", dpi=180)
    plt.close()

print("Figuras guardadas en:", FIG)


# %% [markdown]
# ## 18B. Periodos presidenciales y análisis estadístico/estocástico


# %% Cell 55
# ============================================================
# 18B. PERIODOS PRESIDENCIALES Y ANÁLISIS ESTADÍSTICO/ESTOCÁSTICO
# ============================================================

def assign_presidential_period(fecha):
    f = pd.Timestamp(fecha)
    periods = [
        ("Pastrana", "1998-08-07", "2002-08-06"),
        ("Uribe I", "2002-08-07", "2006-08-06"),
        ("Uribe II", "2006-08-07", "2010-08-06"),
        ("Santos I", "2010-08-07", "2014-08-06"),
        ("Santos II", "2014-08-07", "2018-08-06"),
        ("Duque", "2018-08-07", "2022-08-06"),
        ("Petro", "2022-08-07", "2026-08-06"),
    ]
    for name, start, end in periods:
        if pd.Timestamp(start) <= f <= pd.Timestamp(end):
            return name
    return "Fuera_rango"

df["periodo_presidencial"] = df["fecha"].apply(assign_presidential_period)

presidential_periods = pd.DataFrame([
    {"periodo_presidencial": "Pastrana", "inicio": "1998-08-07", "fin": "2002-08-06"},
    {"periodo_presidencial": "Uribe I", "inicio": "2002-08-07", "fin": "2006-08-06"},
    {"periodo_presidencial": "Uribe II", "inicio": "2006-08-07", "fin": "2010-08-06"},
    {"periodo_presidencial": "Santos I", "inicio": "2010-08-07", "fin": "2014-08-06"},
    {"periodo_presidencial": "Santos II", "inicio": "2014-08-07", "fin": "2018-08-06"},
    {"periodo_presidencial": "Duque", "inicio": "2018-08-07", "fin": "2022-08-06"},
    {"periodo_presidencial": "Petro", "inicio": "2022-08-07", "fin": "2026-08-06"},
])
save_csv(presidential_periods, "presidential_periods.csv")

# Variables de retorno para análisis estocástico
for h in HORIZONS:
    if f"future_logret_h{h}" not in df.columns:
        df[f"future_logret_h{h}"] = np.log(df["precio_oro"].shift(-h) / df["precio_oro"])

df["oro_logret_1d"] = np.log(df["precio_oro"] / df["precio_oro"].shift(1))
df["oro_logret_21d_realized"] = np.log(df["precio_oro"] / df["precio_oro"].shift(21))

def stochastic_group_summary(data, group_col, ret_col="future_logret_h21"):
    rows = []
    d = data.dropna(subset=[group_col, ret_col]).copy()
    for gname, g in d.groupby(group_col, dropna=False):
        r = pd.to_numeric(g[ret_col], errors="coerce").dropna()
        r1 = pd.to_numeric(g["oro_logret_1d"], errors="coerce").dropna()
        if len(r) < 20:
            continue
        acf1 = r.autocorr(lag=1) if len(r) > 2 else np.nan
        acf5 = r.autocorr(lag=5) if len(r) > 10 else np.nan
        rows.append({
            "grupo": group_col,
            "segmento": str(gname),
            "n": int(len(r)),
            "fecha_ini": g["fecha"].min(),
            "fecha_fin": g["fecha"].max(),
            "mean_h21_return": float(r.mean()),
            "median_h21_return": float(r.median()),
            "std_h21_return": float(r.std()),
            "min_h21_return": float(r.min()),
            "max_h21_return": float(r.max()),
            "skew_h21_return": float(r.skew()),
            "kurtosis_h21_return": float(r.kurtosis()),
            "positive_share_h21": float((r > 0).mean()),
            "zero_share_h21": float((r == 0).mean()),
            "acf1_h21": float(acf1) if pd.notna(acf1) else np.nan,
            "acf5_h21": float(acf5) if pd.notna(acf5) else np.nan,
            "daily_vol_annualized": float(r1.std() * np.sqrt(252)) if len(r1) > 10 else np.nan,
            "daily_mean_annualized": float(r1.mean() * 252) if len(r1) > 10 else np.nan,
            "sharpe_like_daily": float((r1.mean() / r1.std()) * np.sqrt(252)) if len(r1) > 10 and r1.std() > 0 else np.nan,
        })
    return pd.DataFrame(rows)

stoch_tables = []
stoch_outputs = {
    "president": ("periodo_presidencial", "stochastic_summary_by_president.csv"),
    "cluster_abs": ("cluster_abs", "stochastic_summary_by_cluster_abs.csv"),
    "cluster_chg": ("cluster_chg", "stochastic_summary_by_cluster_chg.csv"),
    "dutch_disease": ("dd_lag1_dd_clasificacion", "stochastic_summary_by_dutch_disease.csv"),
    "policy_stance": ("policy_stance", "stochastic_summary_by_policy_stance.csv"),
}

for key, (col, fname) in stoch_outputs.items():
    if col in df.columns:
        tab = stochastic_group_summary(df, col, ret_col="future_logret_h21")
        save_csv(tab, fname)
        stoch_tables.append(tab)

stochastic_summary_combined = pd.concat(stoch_tables, ignore_index=True) if stoch_tables else pd.DataFrame()
save_csv(stochastic_summary_combined, "stochastic_summary_combined.csv")

# Agregar periodo presidencial a predicciones y segmentos
if "periodo_presidencial" not in best_preds.columns:
    best_preds = best_preds.merge(df[["fecha", "periodo_presidencial"]].drop_duplicates("fecha"), on="fecha", how="left")
    save_csv(best_preds, "predictions.csv")

# Desempeño del modelo por periodo presidencial
pres_seg = best_preds.dropna(subset=["periodo_presidencial"]).copy()
perf_rows = []
for per, g in pres_seg.groupby("periodo_presidencial"):
    if len(g) < 20 or g["y_true"].nunique() < 2:
        continue
    perf_rows.append({
        "segment_col": "periodo_presidencial",
        "segment": per,
        "n": len(g),
        "positive_share": g["y_true"].mean(),
        "accuracy": accuracy_score(g["y_true"], g["pred_up_calibrated"]),
        "balanced_accuracy": balanced_accuracy_score(g["y_true"], g["pred_up_calibrated"]),
        "recall_up": recall_score(g["y_true"], g["pred_up_calibrated"], zero_division=0),
        "recall_not_up": recall_score(1-g["y_true"], 1-g["pred_up_calibrated"], zero_division=0),
        "roc_auc": roc_auc_score(g["y_true"], g["proba_up"]) if g["y_true"].nunique() > 1 else np.nan,
    })
president_model_performance = pd.DataFrame(perf_rows)
save_csv(president_model_performance, "model_performance_by_presidential_period.csv")

# Agregar estos resultados al segment_metrics general
if "segment_metrics" in globals() and not president_model_performance.empty:
    segment_metrics = pd.concat([segment_metrics, president_model_performance], ignore_index=True)
    save_csv(segment_metrics, "segment_metrics.csv")

# Estrategias por periodo presidencial
if "strategy_ts" in globals() and not strategy_ts.empty:
    st = strategy_ts.merge(df[["fecha", "periodo_presidencial"]].drop_duplicates("fecha"), on="fecha", how="left")
    strat_rows = []
    for (strategy, per), g in st.groupby(["strategy", "periodo_presidencial"]):
        if len(g) < 20:
            continue
        cap0 = g["capital"].iloc[0]
        cap1 = g["capital"].iloc[-1]
        strat_rows.append({
            "strategy": strategy,
            "periodo_presidencial": per,
            "fecha_ini": g["fecha"].min(),
            "fecha_fin": g["fecha"].max(),
            "n": len(g),
            "capital_ini_periodo": cap0,
            "capital_fin_periodo": cap1,
            "period_return": cap1 / cap0 - 1 if cap0 else np.nan,
            "mean_daily_return": g["net_return"].mean(),
            "vol_daily_return": g["net_return"].std(),
            "max_drawdown_period": g["drawdown"].min(),
            "time_in_market": g["exposure"].abs().mean(),
        })
    strategy_by_presidential_period = pd.DataFrame(strat_rows)
    save_csv(strategy_by_presidential_period, "strategy_by_presidential_period.csv")

# Figuras de análisis estocástico
try:
    plot_data = df.dropna(subset=["periodo_presidencial", "future_logret_h21"]).copy()
    order = ["Pastrana", "Uribe I", "Uribe II", "Santos I", "Santos II", "Duque", "Petro"]
    plot_data = plot_data[plot_data["periodo_presidencial"].isin(order)]
    plt.figure(figsize=(11,5))
    data_to_plot = [plot_data.loc[plot_data["periodo_presidencial"] == p, "future_logret_h21"].dropna()*100 for p in order if p in plot_data["periodo_presidencial"].unique()]
    labels = [p for p in order if p in plot_data["periodo_presidencial"].unique()]
    plt.boxplot(data_to_plot, labels=labels, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Retorno futuro h21 (%)")
    plt.title("Distribución del retorno h21 por periodo presidencial")
    plt.tight_layout()
    plt.savefig(FIG / "stochastic_return_boxplot_by_president.png", dpi=180)
    plt.close()

    vol_tab = stochastic_summary_combined.copy()
    vol_tab = vol_tab[vol_tab["grupo"].isin(["periodo_presidencial", "cluster_abs", "cluster_chg", "dd_lag1_dd_clasificacion"])]
    vol_tab["label"] = vol_tab["grupo"] + " · " + vol_tab["segmento"]
    vol_tab = vol_tab.sort_values("std_h21_return", ascending=False).head(25)
    plt.figure(figsize=(11,6))
    plt.barh(vol_tab["label"][::-1], vol_tab["std_h21_return"][::-1]*100)
    plt.xlabel("Desviación estándar retorno h21 (%)")
    plt.title("Volatilidad h21 por segmento económico")
    plt.tight_layout()
    plt.savefig(FIG / "stochastic_volatility_by_segment.png", dpi=180)
    plt.close()
except Exception as e:
    print("No se pudieron crear figuras estocásticas:", e)

print("Análisis presidencial/estocástico listo.")


# %% [markdown]
# ## 18C. Perfiles de clusters, transiciones y regímenes económicos


# %% Cell 57
# ============================================================
# 18C. PERFIL DE CLUSTERS Y TRANSICIONES
# ============================================================

def cluster_profile(data, cluster_col, vars_profile):
    vars_profile = [v for v in vars_profile if v in data.columns]
    rows = []
    for cl, g in data.groupby(cluster_col, dropna=False):
        row = {
            "cluster_col": cluster_col,
            "cluster": str(cl),
            "n": len(g),
            "fecha_ini": g["fecha"].min(),
            "fecha_fin": g["fecha"].max(),
        }
        for v in vars_profile:
            row[f"{v}_mean"] = pd.to_numeric(g[v], errors="coerce").mean()
            row[f"{v}_median"] = pd.to_numeric(g[v], errors="coerce").median()
            row[f"{v}_std"] = pd.to_numeric(g[v], errors="coerce").std()
        if "future_logret_h21" in g:
            row["mean_future_return_h21"] = g["future_logret_h21"].mean()
            row["positive_share_h21"] = (g["future_logret_h21"] > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)

vars_profile = [
    "precio_oro", "trm", "tipm", "precio_brent", "dtf",
    "inflacion_sin_alimentos", "precio_cafe_centusd",
    "demanda_energetica", "precio_bolsa_nacional_energetica",
    "dd_lag1_dd_score", "tipm_change_30"
]
if "cluster_abs" in df.columns:
    cp_abs = cluster_profile(df, "cluster_abs", vars_profile)
    save_csv(cp_abs, "cluster_profile_abs.csv")
else:
    cp_abs = pd.DataFrame()
if "cluster_chg" in df.columns:
    cp_chg = cluster_profile(df, "cluster_chg", vars_profile)
    save_csv(cp_chg, "cluster_profile_chg.csv")
else:
    cp_chg = pd.DataFrame()

def transition_matrix(data, cluster_col):
    s = data.sort_values("fecha")[cluster_col].astype("Int64").astype(str)
    trans = pd.crosstab(s.shift(1), s, normalize="index").fillna(0)
    trans.index.name = "from_cluster"
    trans.columns.name = "to_cluster"
    return trans.reset_index()

if "cluster_abs" in df.columns:
    trans_abs = transition_matrix(df, "cluster_abs")
    save_csv(trans_abs, "cluster_transition_abs.csv")
else:
    trans_abs = pd.DataFrame()
if "cluster_chg" in df.columns:
    trans_chg = transition_matrix(df, "cluster_chg")
    save_csv(trans_chg, "cluster_transition_chg.csv")
else:
    trans_chg = pd.DataFrame()

# Figuras heatmap sin depender de seaborn
def heatmap_from_profile(profile, name, title):
    if profile.empty:
        return
    mean_cols = [c for c in profile.columns if c.endswith("_mean")]
    if not mean_cols:
        return
    mat = profile.set_index("cluster")[mean_cols].copy()
    # Estandarizar columnas para lectura comparativa
    mat_z = (mat - mat.mean()) / mat.std(ddof=0)
    mat_z = mat_z.replace([np.inf, -np.inf], np.nan).fillna(0)
    plt.figure(figsize=(max(8, len(mean_cols)*0.5), 4))
    plt.imshow(mat_z.values, aspect="auto")
    plt.colorbar(label="z-score por variable")
    plt.yticks(range(len(mat_z.index)), mat_z.index)
    plt.xticks(range(len(mat_z.columns)), [c.replace("_mean","") for c in mat_z.columns], rotation=60, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180)
    plt.close()

def heatmap_transition(trans, name, title):
    if trans.empty:
        return
    mat = trans.set_index("from_cluster")
    plt.figure(figsize=(5,4))
    plt.imshow(mat.values, aspect="auto")
    plt.colorbar(label="probabilidad")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks(range(len(mat.columns)), mat.columns)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180)
    plt.close()

heatmap_from_profile(cp_abs, "cluster_profile_abs_heatmap.png", "Perfil macroeconómico por cluster_abs")
heatmap_from_profile(cp_chg, "cluster_profile_chg_heatmap.png", "Perfil macroeconómico por cluster_chg")
heatmap_transition(trans_abs, "cluster_transition_abs_heatmap.png", "Transiciones cluster_abs")
heatmap_transition(trans_chg, "cluster_transition_chg_heatmap.png", "Transiciones cluster_chg")

print("Perfiles de clusters y transiciones listos.")


# %% [markdown]
# ## 18D. Local biplot UMAP


# %% Cell 59
# ============================================================
# 18D. LOCAL BIPLOT UMAP
# ============================================================

def compute_biplot_loadings(regimes_df, data, x_col, y_col, vars_for_biplot):
    merged = regimes_df[["fecha", x_col, y_col]].merge(data[["fecha"] + vars_for_biplot], on="fecha", how="left")
    rows = []
    for v in vars_for_biplot:
        s = pd.to_numeric(merged[v], errors="coerce")
        valid = merged[[x_col, y_col]].notna().all(axis=1) & s.notna()
        if valid.sum() < 30:
            continue
        cx = np.corrcoef(merged.loc[valid, x_col], s.loc[valid])[0,1]
        cy = np.corrcoef(merged.loc[valid, y_col], s.loc[valid])[0,1]
        if not np.isfinite(cx): cx = 0
        if not np.isfinite(cy): cy = 0
        rows.append({
            "feature": v,
            "loading_x": cx,
            "loading_y": cy,
            "loading_norm": float(np.sqrt(cx**2 + cy**2))
        })
    return pd.DataFrame(rows).sort_values("loading_norm", ascending=False)

def plot_local_biplot(regimes_df, loadings, x_col, y_col, cluster_col, out_name, title, top_n=12, sample_n=2500):
    if loadings.empty or regimes_df.empty:
        return
    r = regimes_df.dropna(subset=[x_col, y_col, cluster_col]).copy()
    if len(r) > sample_n:
        r = r.sample(sample_n, random_state=RANDOM_STATE).sort_values("fecha")
    plt.figure(figsize=(8,6))
    # No se fija paleta específica; matplotlib usa ciclo por defecto.
    for cl, g in r.groupby(cluster_col):
        plt.scatter(g[x_col], g[y_col], s=5, alpha=0.45, label=f"{cluster_col}={cl}")
    top = loadings.head(top_n).copy()
    xspan = r[x_col].max() - r[x_col].min()
    yspan = r[y_col].max() - r[y_col].min()
    scale = 0.28 * min(xspan, yspan)
    x0 = r[x_col].median()
    y0 = r[y_col].median()
    for _, row in top.iterrows():
        dx = row["loading_x"] * scale
        dy = row["loading_y"] * scale
        plt.arrow(x0, y0, dx, dy, head_width=0.03*scale, length_includes_head=True, alpha=0.8)
        plt.text(x0 + dx*1.08, y0 + dy*1.08, row["feature"], fontsize=8)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(FIG / out_name, dpi=180)
    plt.close()

biplot_vars_abs = [v for v in vars_profile if v in df.columns]
biplot_vars_chg = []
for v in biplot_vars_abs:
    chg_name = f"{v}_chg21_for_biplot"
    df[chg_name] = pd.to_numeric(df[v], errors="coerce").diff(21)
    biplot_vars_chg.append(chg_name)

if "regimes" in globals() and not regimes.empty:
    load_abs = compute_biplot_loadings(regimes, df, "umap_abs_1", "umap_abs_2", biplot_vars_abs)
    save_csv(load_abs, "local_biplot_abs_loadings.csv")
    plot_local_biplot(regimes, load_abs, "umap_abs_1", "umap_abs_2", "cluster_abs",
                      "umap_abs_local_biplot.png", "Local biplot · UMAP niveles absolutos")

    load_chg = compute_biplot_loadings(regimes, df, "umap_chg_1", "umap_chg_2", biplot_vars_chg)
    save_csv(load_chg, "local_biplot_chg_loadings.csv")
    plot_local_biplot(regimes, load_chg, "umap_chg_1", "umap_chg_2", "cluster_chg",
                      "umap_chg_local_biplot.png", "Local biplot · UMAP cambios")
else:
    print("No existe objeto regimes; no se puede crear local biplot.")
    save_csv(pd.DataFrame(), "local_biplot_abs_loadings.csv")
    save_csv(pd.DataFrame(), "local_biplot_chg_loadings.csv")

print("Local biplots listos.")


# %% [markdown]
# ## 18E. Tratamiento de datos, trazabilidad y catálogo de features


# %% Cell 61
# ============================================================
# 18E. TRATAMIENTO DE DATOS Y CATÁLOGO DE FEATURES
# ============================================================

# Missingness
missingness = []
for c in df.columns:
    missingness.append({
        "column": c,
        "dtype": str(df[c].dtype),
        "n": len(df),
        "n_missing": int(df[c].isna().sum()),
        "pct_missing": float(df[c].isna().mean()),
        "n_unique": int(df[c].nunique(dropna=True)),
        "is_feature_candidate": c in all_numeric_cols if "all_numeric_cols" in globals() else False,
        "is_global": is_global(c) if "is_global" in globals() else False,
    })
missingness_summary = pd.DataFrame(missingness).sort_values("pct_missing", ascending=False)
save_csv(missingness_summary, "missingness_summary.csv")

# Catálogo de features
feature_catalog_rows = []
if "feature_sets" in globals():
    for fs, cols in feature_sets.items():
        for c in cols:
            transform = "raw"
            if "lag" in c: transform = "lag"
            elif "roll_mean" in c: transform = "rolling_mean"
            elif "roll_std" in c: transform = "rolling_std"
            elif "logret" in c: transform = "log_return"
            elif "pct" in c: transform = "pct_change"
            elif "diff" in c or "change" in c: transform = "difference"
            elif "dist_max" in c or "dist_min" in c: transform = "rolling_distance"
            elif "cluster" in c: transform = "regime_cluster"
            elif "dd_" in c: transform = "dutch_disease_lag"
            feature_catalog_rows.append({
                "feature_set": fs,
                "feature": c,
                "group": feature_group(c) if "feature_group" in globals() else ("global" if is_global(c) else "colombia"),
                "transform": transform,
                "is_global": is_global(c) if "is_global" in globals() else False,
                "used_in_best": c in (bundle["features"] if "bundle" in globals() and isinstance(bundle, dict) and "features" in bundle else []),
            })
feature_engineering_catalog = pd.DataFrame(feature_catalog_rows)
save_csv(feature_engineering_catalog, "feature_engineering_catalog.csv")

# Log de tratamiento
data_treatment_log = pd.DataFrame([
    {"paso": 1, "etapa": "Carga base Colombia", "descripcion": "Normalización de nombres, fecha y columnas numéricas desde BD_Energía_Colombia.xlsx.", "archivo_salida": "base_colombia_normalizada.csv"},
    {"paso": 2, "etapa": "Frecuencia efectiva", "descripcion": "Cálculo de valores únicos, porcentaje repetido y frecuencia real por variable.", "archivo_salida": "coverage.csv"},
    {"paso": 3, "etapa": "Enfermedad holandesa", "descripcion": "Integración anual y rezago dd_lag1 para evitar fuga temporal.", "archivo_salida": "dutch_disease.csv"},
    {"paso": 4, "etapa": "Contexto global", "descripcion": "Reutilización o descarga de Yahoo Finance; reindexación al calendario local y forward-fill sin backfill.", "archivo_salida": "global_market_context_homogenized.csv"},
    {"paso": 5, "etapa": "Feature engineering", "descripcion": "Retornos, diferencias, lags, rolling means/std, distancias a máximos/mínimos y variables calendario.", "archivo_salida": "feature_engineering_catalog.csv"},
    {"paso": 6, "etapa": "Targets", "descripcion": "Construcción de target_up_h21 y target_up_h30 como sube vs no sube.", "archivo_salida": "target_diagnostics.csv"},
    {"paso": 7, "etapa": "Regímenes", "descripcion": "UMAP/KMeans Colombia-only para niveles absolutos y cambios.", "archivo_salida": "umap_regimes.csv"},
    {"paso": 8, "etapa": "Validación", "descripcion": "Walk-forward temporal con calibración de umbral usando solo pasado.", "archivo_salida": "classification_summary.csv"},
    {"paso": 9, "etapa": "Auditoría", "descripcion": "Comparación con feature sets sin oro internacional directo, sin metales y solo macro global.", "archivo_salida": "audit_summary.csv"},
])
save_csv(data_treatment_log, "data_treatment_log.csv")

print("Tratamiento de datos documentado.")


# %% [markdown]
# ## 21B. Gráficas y salidas completas de modelos


# %% Cell 63
# ============================================================
# 21B. GRÁFICAS Y SALIDAS COMPLETAS DE MODELOS
# ============================================================

def config_mask(table, row):
    m = pd.Series(True, index=table.index)
    if "horizon" in table.columns:
        m &= table["horizon"].astype(int).eq(int(row.get("horizon", MAIN_HORIZON)))
    if "feature_set" in table.columns:
        m &= table["feature_set"].astype(str).eq(str(row["feature_set"]))
    if "model" in table.columns:
        m &= table["model"].astype(str).eq(str(row["model"]))
    if "train_window_years" in table.columns:
        m &= table["train_window_years"].astype(str).eq(str(row["train_window_years"]))
    if "weight_strategy" in table.columns and "weight_strategy" in row.index:
        m &= table["weight_strategy"].astype(str).eq(str(row.get("weight_strategy", "none")))
    return m

predictions_final = pd.read_csv(OUT / "predictions.csv", parse_dates=["fecha"])
if "real_return_h21" not in predictions_final.columns:
    ret_cols = [c for c in predictions_final.columns if c.startswith("real_return")]
    if ret_cols:
        predictions_final["real_return_h21"] = predictions_final[ret_cols[0]]

# 1) Retorno real vs estimado
if "pred_return_direct" in predictions_final.columns and "real_return_h21" in predictions_final.columns:
    p = predictions_final.dropna(subset=["real_return_h21", "pred_return_direct"]).copy()
    if len(p):
        plt.figure(figsize=(7, 6))
        plt.scatter(p["real_return_h21"] * 100, p["pred_return_direct"] * 100, s=10, alpha=0.45)
        lo = min(p["real_return_h21"].min(), p["pred_return_direct"].min()) * 100
        hi = max(p["real_return_h21"].max(), p["pred_return_direct"].max()) * 100
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.xlabel("Retorno real h21 (%)")
        plt.ylabel("Retorno estimado h21 (%)")
        plt.title("Retorno real vs retorno estimado")
        plt.tight_layout()
        plt.savefig(FIG / "real_vs_estimated_return.png", dpi=180)
        plt.close()

# 2) Precio real futuro vs precio esperado
if "expected_price_h21" in predictions_final.columns and "real_return_h21" in predictions_final.columns:
    p = predictions_final.sort_values("fecha").copy()
    p["real_future_price_h21"] = p["precio_oro"] * np.exp(p["real_return_h21"])
    plt.figure(figsize=(11, 5))
    plt.plot(p["fecha"], p["real_future_price_h21"], label="Precio futuro real h21")
    plt.plot(p["fecha"], p["expected_price_h21"], label="Precio esperado h21")
    plt.title("Precio real futuro vs precio esperado a h21")
    plt.ylabel("USD/oz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "real_vs_expected_price_h21.png", dpi=180)
    plt.close()

# 3) Probabilidad vs umbral
p = predictions_final.sort_values("fecha").copy()
plt.figure(figsize=(11, 4))
plt.plot(p["fecha"], p["proba_up"], label="P(sube)")
plt.plot(p["fecha"], p["threshold"], label="Umbral calibrado", linewidth=1)
plt.fill_between(p["fecha"], 0, 1, where=p["pred_up_calibrated"].astype(bool), alpha=0.08)
plt.ylim(0, 1)
plt.title("Probabilidad de subida vs umbral calibrado")
plt.ylabel("Probabilidad")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "probability_threshold_best_model.png", dpi=180)
plt.close()

# 4) Comparación por feature set
cs = pd.read_csv(OUT / "classification_summary.csv")
cs_h = cs[cs["horizon"] == MAIN_HORIZON].copy()
if "weight_strategy" not in cs_h.columns:
    cs_h["weight_strategy"] = "none"

best_by_fs = cs_h.sort_values("mean_balanced_accuracy", ascending=False).groupby("feature_set").head(1)
best_by_fs = best_by_fs.sort_values("mean_balanced_accuracy", ascending=True)
plt.figure(figsize=(9, 5))
plt.barh(best_by_fs["feature_set"], best_by_fs["mean_balanced_accuracy"] * 100)
plt.xlabel("Balanced accuracy media (%)")
plt.title("Mejor configuración por feature set")
plt.tight_layout()
plt.savefig(FIG / "model_comparison_by_feature_set.png", dpi=180)
plt.close()

# 5) Comparación por ventana de entrenamiento
best_by_window = cs_h.sort_values("mean_balanced_accuracy", ascending=False).groupby("train_window_years").head(1)
best_by_window = best_by_window.sort_values("mean_balanced_accuracy", ascending=True)
plt.figure(figsize=(8, 4))
plt.barh(best_by_window["train_window_years"].astype(str), best_by_window["mean_balanced_accuracy"] * 100)
plt.xlabel("Balanced accuracy media (%)")
plt.ylabel("Ventana de entrenamiento")
plt.title("Mejor desempeño por ventana temporal")
plt.tight_layout()
plt.savefig(FIG / "model_comparison_by_train_window.png", dpi=180)
plt.close()

# 6) IA vs determinísticos
comp = cs_h.copy()
comp["tipo_modelo"] = np.where(comp["feature_set"].eq("deterministic_baseline"), "determinístico", "IA/tabular")
best_type = comp.sort_values("mean_balanced_accuracy", ascending=False).groupby(["tipo_modelo", "model"]).head(1)
best_type = best_type.sort_values("mean_balanced_accuracy", ascending=False).head(20).sort_values("mean_balanced_accuracy")
plt.figure(figsize=(10, 7))
plt.barh(best_type["tipo_modelo"] + " · " + best_type["model"], best_type["mean_balanced_accuracy"] * 100)
plt.xlabel("Balanced accuracy media (%)")
plt.title("Comparación de modelos de IA y reglas determinísticas")
plt.tight_layout()
plt.savefig(FIG / "ai_vs_deterministic_bacc.png", dpi=180)
plt.close()

# 7) Métricas por bloque del mejor modelo
cr = pd.read_csv(OUT / "classification_results.csv")
br = cr[config_mask(cr, best_row)].sort_values("test_start")
if not br.empty:
    plt.figure(figsize=(11, 5))
    plt.plot(br["test_start"], br["balanced_accuracy"] * 100, marker="o", label="BACC")
    plt.plot(br["test_start"], br["roc_auc"] * 100, marker="o", label="AUC")
    plt.plot(br["test_start"], br["accuracy"] * 100, marker="o", label="Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Métrica (%)")
    plt.title("Métricas walk-forward por bloque · mejor modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "walkforward_metrics_best_model.png", dpi=180)
    plt.close()

# 8) Salidas de probabilidad de modelos destacados
allp = pd.read_csv(OUT / "all_predictions.csv", parse_dates=["fecha"])
top_configs = cs_h.sort_values("mean_balanced_accuracy", ascending=False).head(6)[["feature_set", "model", "train_window_years", "weight_strategy", "horizon"]]
sample_rows = []

plt.figure(figsize=(11, 5))
for _, row in top_configs.iterrows():
    g = allp[config_mask(allp, row)].sort_values("fecha").copy()
    if g.empty:
        continue
    g_sample = g.tail(240).copy()
    label = f"{row['feature_set']} · {row['model']} · {row['train_window_years']} · {row.get('weight_strategy', 'none')}"
    plt.plot(g_sample["fecha"], g_sample["proba_up"], linewidth=1, label=label)
    sample_rows.append(g_sample.assign(config_label=label))
plt.ylim(0, 1)
plt.ylabel("P(sube)")
plt.title("Salidas de probabilidad · modelos destacados")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "model_outputs_probability_sample.png", dpi=180)
plt.close()

model_outputs_comparison = cs_h.sort_values("mean_balanced_accuracy", ascending=False).copy()
save_csv(model_outputs_comparison, "model_outputs_comparison.csv")

if sample_rows:
    model_predictions_sample = pd.concat(sample_rows, ignore_index=True)
else:
    model_predictions_sample = pd.DataFrame()
save_csv(model_predictions_sample, "model_predictions_sample.csv")

print("Gráficas y salidas completas de modelos generadas.")


# %% [markdown]
# ## 22. Empaquetar outputs


# %% Cell 65
# ============================================================
# 22. ZIP FINAL
# ============================================================

# Manifiesto primero, para que también quede dentro del ZIP.
manifest = {
    "created_at": datetime.now().isoformat(),
    "regime_version": REGIME_VERSION,
    "best_config": best_row.to_dict() if "best_row" in globals() else {},
    "files": sorted([str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()])
}
with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

zip_out = ROOT / "dashboard_data_bundle.zip"
if zip_out.exists():
    zip_out.unlink()

with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUT.parent))

# Checkpoint final equivalente al bundle parcial.
checkpoint_out = ROOT / "dashboard_data_checkpoint.zip"
if checkpoint_out.exists():
    checkpoint_out.unlink()
with zipfile.ZipFile(checkpoint_out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUT.parent))

print("ZIP final:", zip_out)
print("Checkpoint final:", checkpoint_out)
print("Archivos generados:")
for f in sorted([str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()]):
    print("-", f)


# %% [markdown]
# # Resultado esperado para conectar con el HTML


# %% [markdown]
# # Checklist ampliado de salida


# %% [markdown]
# # Checklist final de salida del cuaderno
