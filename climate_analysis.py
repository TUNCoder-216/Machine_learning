#!/usr/bin/env python3
"""
climate_analysis.py — Climate Change Trends Analysis ML Project
===============================================================
Fully self-contained: auto-installs dependencies, fetches & caches data,
runs EDA, trains ML models, and forecasts climate variables to 2100.

Usage: python climate_analysis.py
"""

# ─── Force UTF-8 output so Unicode labels print on Windows terminals ──────────
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── STEP 0: Bootstrap — auto-install dependencies ────────────────────────────
import subprocess
import importlib

REQUIRED = {
    "pandas":      "pandas",
    "numpy":       "numpy",
    "matplotlib":  "matplotlib",
    "seaborn":     "seaborn",
    "sklearn":     "scikit-learn",
    "xgboost":     "xgboost",
    "lightgbm":    "lightgbm",
    "statsmodels": "statsmodels",
    "requests":    "requests",
}

print("=" * 62)
print("=== STEP 0: Checking / installing dependencies           ===")
print("=" * 62)

for import_name, pip_name in REQUIRED.items():
    try:
        importlib.import_module(import_name)
        print(f"  [OK] {pip_name}")
    except ImportError:
        print(f"  [--] Installing {pip_name} ...", end="", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(" done.")

# ─── All imports (safe after bootstrap) ──────────────────────────────────────
import os
import io
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")                     # non-interactive backend; saves PNGs without display

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import lightgbm as lgb
import requests

# ─── Global config ────────────────────────────────────────────────────────────
CACHE_DIR    = "./climate_cache"
RANDOM_STATE = 42
FORECAST_END = 2100
POLY_DEGREE  = 2          # degree of OLS polynomial for trend extrapolation
CV_SPLITS    = 5

os.makedirs(CACHE_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

GENERATED_FILES = []      # track output PNGs for the summary

# ─── Helper: save figure ──────────────────────────────────────────────────────
def savefig(fname):
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    GENERATED_FILES.append(fname)
    print(f"    -> Saved {fname}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING & CLEANING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("=== STEP 1: Data Loading & Cleaning                     ===")
print("=" * 62)


def fetch_raw(name: str, urls: list, cache_path: str) -> str:
    """
    Fetch raw text from the first reachable URL; cache locally.
    On total failure, load from cache (if it exists).
    """
    headers = {"User-Agent": "climate-analysis-script/1.0 (educational)"}

    for url in urls:
        try:
            print(f"  Fetching {name} from {url[:65]}...")
            resp = requests.get(url, timeout=30, headers=headers)
            resp.raise_for_status()
            with open(cache_path, "wb") as fh:
                fh.write(resp.content)
            print(f"  [OK] {name} cached -> {cache_path}")
            return resp.text
        except Exception as exc:
            print(f"  [!!] {url[:65]}  failed: {exc}")

    if os.path.exists(cache_path):
        print(f"  [~~] All URLs failed; loading {name} from local cache.")
        with open(cache_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    raise RuntimeError(
        f"\n[ERROR] Could not load '{name}'.\n"
        f"All URLs tried: {urls}\n"
        f"No cache found at: {cache_path}"
    )


# ── 1a. NASA GISTEMP v4 ───────────────────────────────────────────────────────
def load_gistemp() -> pd.DataFrame:
    """Annual global temperature anomaly (°C) from NASA GISTEMP v4."""
    raw = fetch_raw(
        "NASA_GISTEMP",
        [
            "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
            "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts.csv",
        ],
        os.path.join(CACHE_DIR, "gistemp.csv"),
    )
    lines = raw.splitlines()
    # Find the header row that starts with "Year"
    try:
        header_idx = next(
            i for i, ln in enumerate(lines) if ln.strip().startswith("Year")
        )
    except StopIteration:
        raise RuntimeError("Could not find 'Year' header in GISTEMP data.")

    cleaned = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(cleaned), na_values=["***", "****", ""])
    df.columns = df.columns.str.strip()

    # Keep Year and J-D (January–December annual mean)
    df = df[["Year", "J-D"]].copy()
    df.columns = ["year", "temp_anomaly"]
    df["year"]        = pd.to_numeric(df["year"], errors="coerce")
    df["temp_anomaly"] = pd.to_numeric(df["temp_anomaly"], errors="coerce")
    df = df.dropna().astype({"year": int}).set_index("year")
    print(f"  GISTEMP  : {len(df)} annual records  "
          f"({df.index.min()}-{df.index.max()})")
    return df


# ── 1b. NOAA Mauna Loa CO2 ───────────────────────────────────────────────────
def load_co2() -> pd.DataFrame:
    """Annual mean atmospheric CO2 (ppm) from NOAA Mauna Loa."""
    raw = fetch_raw(
        "NOAA_CO2",
        [
            "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv",
            "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        ],
        os.path.join(CACHE_DIR, "co2.csv"),
    )

    # Strip comment lines (start with '#')
    data_lines = [
        ln for ln in raw.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    cleaned = "\n".join(data_lines)

    # Try to parse as CSV; fall back to whitespace-delimited
    try:
        df = pd.read_csv(io.StringIO(cleaned), header=None)
    except Exception:
        df = pd.read_csv(io.StringIO(cleaned), sep=r"\s+", header=None)

    # Detect annual vs monthly layout by column count
    # Annual  (co2_annmean_mlo.csv): cols = [year, mean, unc]
    # Monthly (co2_mm_mlo.csv)     : cols = [year, month, decimal, average, ...]
    if df.shape[1] >= 4:
        # Monthly layout: year=col0, average=col3
        df = df.iloc[:, [0, 3]].copy()
    else:
        # Annual layout: year=col0, mean=col1
        df = df.iloc[:, [0, 1]].copy()

    df.columns = ["year", "co2_ppm"]
    df["year"]    = pd.to_numeric(df["year"], errors="coerce")
    df["co2_ppm"] = pd.to_numeric(df["co2_ppm"], errors="coerce")
    df = df.dropna()
    df = df[df["co2_ppm"] > 0]          # remove fill-value rows (-99.99 etc.)
    df["year"] = df["year"].astype(int)

    # Aggregate to annual if monthly data was fetched
    if df.groupby("year").size().max() > 1:
        df = df.groupby("year")["co2_ppm"].mean().reset_index()

    df = df.set_index("year")
    print(f"  CO2      : {len(df)} annual records  "
          f"({df.index.min()}-{df.index.max()})")
    return df


# ── 1c. CSIRO / EPA Global Mean Sea Level ────────────────────────────────────
def load_sea_level() -> pd.DataFrame:
    """
    Global mean sea level (mm) using the EPA/CSIRO Church & White dataset.
    The GitHub 'datasets' project hosts it as a clean CSV.
    Values in the source are in INCHES; we convert to mm (×25.4).
    """
    raw = fetch_raw(
        "CSIRO_SeaLevel",
        [
            "https://raw.githubusercontent.com/datasets/sea-level/main/data/epa-sea-level.csv",
            "https://datahub.io/core/sea-level-rise/r/epa-sea-level.csv",
        ],
        os.path.join(CACHE_DIR, "sea_level.csv"),
    )

    data_lines = [
        ln for ln in raw.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    cleaned = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(cleaned))
    df.columns = df.columns.str.strip()

    # Identify year and sea-level columns by name heuristic
    year_col = next(
        (c for c in df.columns if "year" in c.lower() or "time" in c.lower()),
        df.columns[0],
    )
    sl_col = next(
        (c for c in df.columns
         if ("csiro" in c.lower() and "adjusted" in c.lower())),
        None,
    )
    if sl_col is None:
        sl_col = next(
            (c for c in df.columns
             if any(kw in c.lower() for kw in ("sea", "level", "gmsl", "sl"))),
            df.columns[1],
        )

    df = df[[year_col, sl_col]].copy()
    df.columns = ["year", "sea_level_mm"]
    df["year"]         = pd.to_numeric(df["year"], errors="coerce")
    df["sea_level_mm"] = pd.to_numeric(df["sea_level_mm"], errors="coerce")
    df = df.dropna()

    # Year may be fractional (e.g. 1880.5) → floor to int
    df["year"] = df["year"].astype(float).apply(int)

    # Group sub-annual rows to annual mean
    df = df.groupby("year")["sea_level_mm"].mean().reset_index()

    # EPA dataset values are in inches — convert to mm
    median_val = df["sea_level_mm"].median()
    if abs(median_val) < 25:            # likely inches (should be ~3-4 in by 2000)
        print(f"  Sea level values appear to be in inches "
              f"(median={median_val:.2f}); converting to mm (×25.4)")
        df["sea_level_mm"] = df["sea_level_mm"] * 25.4

    df = df.set_index("year")
    print(f"  Sea Level: {len(df)} annual records  "
          f"({df.index.min()}-{df.index.max()})")
    return df


# ── Load & merge ──────────────────────────────────────────────────────────────
df_temp = load_gistemp()
df_co2  = load_co2()
df_sl   = load_sea_level()

print("\n  Merging datasets ...")
# Wide (union) frame — used for plotting individual series
df_wide = df_temp.join(df_co2, how="outer").join(df_sl, how="outer")
df_wide = df_wide.sort_index()
df_wide.index.name = "year"

# Model frame (inner join — all three variables present)
df_model = df_wide.dropna(subset=["temp_anomaly", "co2_ppm", "sea_level_mm"]).copy()

# Interpolate any remaining small gaps (≤2 years) in the model frame
df_model = df_model.interpolate(method="index", limit=2)
df_model = df_model.dropna()

print(f"  Wide  (union) : {df_wide.shape}  "
      f"years {df_wide.index.min()}-{df_wide.index.max()}")
print(f"  Model (inner) : {df_model.shape}  "
      f"years {df_model.index.min()}-{df_model.index.max()}")

if len(df_model) < 20:
    raise RuntimeError(
        f"Model dataset has only {len(df_model)} rows — too few to train. "
        "Check that all three sources loaded correctly."
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("=== STEP 2: Exploratory Data Analysis                   ===")
print("=" * 62)

print("\n--- Summary Statistics (model dataset) ---")
print(df_model.describe().to_string())

print("\n--- Missing Values (wide dataset) ---")
print(df_wide.isnull().sum().rename("missing").to_frame().to_string())

# 2a. Correlation heatmap
print("\n  Plotting correlation heatmap ...")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    df_model.corr(), annot=True, fmt=".3f", cmap="coolwarm",
    linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    annot_kws={"size": 11},
)
ax.set_title("Pearson Correlation — Temperature · CO₂ · Sea Level", fontsize=12, pad=12)
plt.tight_layout()
savefig("eda_correlation_heatmap.png")

# 2b. Time-series overview
print("  Plotting time-series ...")
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Climate Indicators — Historical Record", fontsize=13, y=1.01)

for ax, col, lbl, color, baseline in [
    (axes[0], "temp_anomaly", "Temp Anomaly (°C)",        "#d62728", 0),
    (axes[1], "co2_ppm",      "CO₂ (ppm)",                "#1f77b4", None),
    (axes[2], "sea_level_mm", "Sea Level Change (mm)",    "#2ca02c", None),
]:
    ax.plot(df_wide.index, df_wide[col], color=color, lw=1.5, alpha=0.9)
    if baseline is not None:
        ax.axhline(baseline, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(lbl, fontsize=9)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel("Year")
plt.tight_layout()
savefig("eda_timeseries.png")

# 2c. Scatter plots
print("  Plotting scatter plots ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Climate Variable Relationships (colour = year)", fontsize=11)

year_vals = df_model.index.astype(float)
cmap      = "RdYlGn"
norm      = plt.Normalize(year_vals.min(), year_vals.max())
sm_obj    = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm_obj.set_array([])

for ax, xcol, xlbl, ycol, ylbl in [
    (axes[0], "co2_ppm",      "CO₂ (ppm)",             "temp_anomaly", "Temp Anomaly (°C)"),
    (axes[1], "sea_level_mm", "Sea Level Change (mm)", "temp_anomaly", "Temp Anomaly (°C)"),
]:
    sc = ax.scatter(
        df_model[xcol], df_model[ycol],
        c=year_vals, cmap=cmap, norm=norm,
        alpha=0.8, edgecolors="k", lw=0.3, s=40,
    )
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    ax.set_title(f"{xlbl} vs {ylbl}")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sm_obj, ax=ax, label="Year")

plt.tight_layout()
savefig("eda_scatter.png")

print("  EDA complete.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("=== STEP 3: Feature Engineering                         ===")
print("=" * 62)

fe = df_model.copy()
fe.index = fe.index.astype(int)
fe["year_val"] = fe.index
fe["decade"]   = (fe.index // 10) * 10

# Lag features (1, 5, 10 years) for all three variables
LAG_COLS = ["temp_anomaly", "co2_ppm", "sea_level_mm"]
LAGS     = [1, 5, 10]
for col in LAG_COLS:
    for lag in LAGS:
        fe[f"{col}_lag{lag}"] = fe[col].shift(lag)
        print(f"  Added lag feature: {col}_lag{lag}")

# Rolling means (3, 5, 10-year windows) on CO2 and temperature
ROLL_COLS    = ["co2_ppm", "temp_anomaly"]
ROLL_WINDOWS = [3, 5, 10]
for col in ROLL_COLS:
    for w in ROLL_WINDOWS:
        fe[f"{col}_roll{w}"] = fe[col].rolling(w, min_periods=1).mean()
        print(f"  Added rolling mean: {col}_roll{w}")

# Interaction features
fe["co2_x_year"]      = fe["co2_ppm"]            * fe["year_val"]
fe["co2_x_sealevel"]  = fe["co2_ppm"]            * fe["sea_level_mm"]
fe["temp_lag1_x_co2"] = fe["temp_anomaly_lag1"]  * fe["co2_ppm"]   # no target leakage
print("  Added interaction features: co2_x_year, co2_x_sealevel, temp_lag1_x_co2")

# Drop rows that have NaN from lags (max lag = 10)
fe = fe.dropna()

TARGET   = "temp_anomaly"
FEATURES = [c for c in fe.columns if c != TARGET]
X = fe[FEATURES]
y = fe[TARGET]

print(f"\n  Feature matrix : {X.shape}  (rows x features)")
print(f"  Target         : '{TARGET}'")
print(f"  Feature list   : {', '.join(FEATURES)}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: MODEL TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("=== STEP 4: Model Training & Evaluation                 ===")
print("=" * 62)

MODELS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0,
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=RANDOM_STATE,
    ),
}

tscv    = TimeSeriesSplit(n_splits=CV_SPLITS)
results = {}

for model_name, model in MODELS.items():
    print(f"\n  Training {model_name} (TimeSeriesSplit CV, n={CV_SPLITS}) ...")
    fold_rmse, fold_mae, fold_r2 = [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        fold_mae .append(mean_absolute_error(y_val, preds))
        fold_r2  .append(r2_score(y_val, preds))
        print(f"    fold {fold_idx}: RMSE={fold_rmse[-1]:.4f}  "
              f"MAE={fold_mae[-1]:.4f}  R²={fold_r2[-1]:.4f}")

    # Final fit on the full feature set (for feature importance extraction)
    model.fit(X, y)

    results[model_name] = {
        "RMSE":  np.mean(fold_rmse),
        "MAE":   np.mean(fold_mae),
        "R²":    np.mean(fold_r2),
        "model": model,
    }
    cv_rmse = np.mean(fold_rmse)
    cv_mae  = np.mean(fold_mae)
    cv_r2   = np.mean(fold_r2)
    print(f"  => Mean CV  RMSE={cv_rmse:.4f}  MAE={cv_mae:.4f}  R²={cv_r2:.4f}")

    # Feature-importance bar chart (top 20)
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=False)
    top_n = min(20, len(importances))

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35 + 1)))
    importances.head(top_n).plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=11)
    ax.set_xlabel("Importance score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fname = f"feat_importance_{model_name.lower()}.png"
    savefig(fname)

# Identify best model (lowest mean CV RMSE)
best_name = min(results, key=lambda k: results[k]["RMSE"])
best_rmse = results[best_name]["RMSE"]

print(f"\n  ★  Best model by CV RMSE: {best_name}  (RMSE={best_rmse:.4f})")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: FORECASTING TO 2100  (Hybrid — Polynomial OLS Trend)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("=== STEP 5: Forecasting to 2100 (Polynomial OLS Trend)  ===")
print("=" * 62)
print(
    f"\n  Note: Tree models (above) cannot extrapolate beyond their training\n"
    f"  range — they would flatline. Forecasting uses statsmodels polynomial\n"
    f"  OLS (degree={POLY_DEGREE}) which CAN extrapolate, and provides\n"
    f"  prediction intervals. Best ML model ({best_name}) is used for\n"
    f"  feature attribution; OLS trend for 2100 projections.\n"
)

hist_years   = df_model.index.values.astype(float)
future_years = np.arange(df_model.index.min(), FORECAST_END + 1, dtype=float)

# Build polynomial basis once (same degree for all variables)
poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=True)
X_hist_poly = poly.fit_transform(hist_years.reshape(-1, 1))
X_fut_poly  = poly.transform(future_years.reshape(-1, 1))

forecast_results = {}
PLOT_VARS = [
    ("temp_anomaly",  "Temperature Anomaly (°C)", "#d62728"),
    ("co2_ppm",       "CO₂ (ppm)",                "#1f77b4"),
    ("sea_level_mm",  "Sea Level Change (mm)",    "#2ca02c"),
]

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle(
    f"Climate Variable Forecasts to {FORECAST_END}\n"
    f"Polynomial OLS trend (degree={POLY_DEGREE}) + 95% prediction interval\n"
    f"[ML tree models used for attribution; OLS used for extrapolation]",
    fontsize=10, y=1.02,
)

for ax, (col, ylabel, color) in zip(axes, PLOT_VARS):
    y_hist = df_model[col].values.astype(float)

    ols = sm.OLS(y_hist, X_hist_poly).fit()
    pred_obj = ols.get_prediction(X_fut_poly)
    pred_df  = pred_obj.summary_frame(alpha=0.05)   # 95% CI

    mean_fc = pred_df["mean"].values
    ci_lo   = pred_df["obs_ci_lower"].values
    ci_hi   = pred_df["obs_ci_upper"].values

    forecast_results[col] = {
        "years": future_years,
        "mean":  mean_fc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }

    # Split at end of historical record
    last_hist = hist_years.max()
    hist_mask = future_years <= last_hist
    fc_mask   = future_years >  last_hist

    ax.plot(hist_years, y_hist,
            color=color, lw=1.8, label="Historical", alpha=0.9)
    ax.plot(future_years[fc_mask], mean_fc[fc_mask],
            color=color, lw=2.2, ls="--", label=f"Forecast (poly OLS)")
    ax.fill_between(
        future_years, ci_lo, ci_hi,
        alpha=0.15, color=color, label="95% Prediction Interval",
    )
    ax.axvline(last_hist, color="gray", lw=0.9, ls=":", alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel("Year")
plt.tight_layout()
savefig("forecast_2100.png")

# Print headline forecast values
print("  Projected Values (mean + 95% prediction interval):")
print(f"  {'Variable':<30}  {'Year':>4}  {'Mean':>8}  {'95% CI':>22}")
print("  " + "-" * 70)
for col, ylabel, _ in PLOT_VARS:
    fr = forecast_results[col]
    for yr in [2030, 2050, 2075, 2100]:
        idx = np.searchsorted(fr["years"], yr)
        if idx < len(fr["years"]) and fr["years"][idx] == yr:
            mean_v = fr["mean"][idx]
            lo_v   = fr["ci_lo"][idx]
            hi_v   = fr["ci_hi"][idx]
            print(f"  {ylabel:<30}  {yr:>4}  {mean_v:>8.2f}  "
                  f"[{lo_v:>8.2f}, {hi_v:>8.2f}]")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("=== STEP 6: Final Summary                               ===")
print("=" * 62)

print("\n── DATASET COVERAGE ──────────────────────────────────────")
print(f"  NASA GISTEMP v4  : {df_temp.index.min()}–{df_temp.index.max()}"
      f"  ({len(df_temp)} years)")
print(f"  NOAA Mauna Loa   : {df_co2.index.min()}–{df_co2.index.max()}"
      f"  ({len(df_co2)} years)")
print(f"  CSIRO Sea Level  : {df_sl.index.min()}–{df_sl.index.max()}"
      f"  ({len(df_sl)} years)")
print(f"  Merged (model)   : {df_model.index.min()}–{df_model.index.max()}"
      f"  ({len(df_model)} years, {len(FEATURES)} features)")

print("\n── MODEL COMPARISON (TimeSeriesSplit, n={}) ──────────────".format(CV_SPLITS))
print(f"  {'Model':<22}  {'CV RMSE':>8}  {'CV MAE':>8}  {'CV R²':>8}")
print("  " + "-" * 52)
for mname, mres in sorted(results.items(), key=lambda x: x[1]["RMSE"]):
    star = " ★ best" if mname == best_name else ""
    print(f"  {mname:<22}  {mres['RMSE']:>8.4f}  {mres['MAE']:>8.4f}  {mres['R²']:>8.4f}{star}")

print("\n── FORECAST HEADLINES (Poly OLS, 95% prediction interval) ─")
print(f"  {'Variable':<30}  {'Year':>4}  {'Mean':>8}  {'95% CI':>22}")
print("  " + "-" * 70)
for col, ylabel, _ in PLOT_VARS:
    fr = forecast_results[col]
    for yr in [2050, 2100]:
        idx = np.searchsorted(fr["years"], yr)
        if idx < len(fr["years"]) and fr["years"][idx] == yr:
            print(f"  {ylabel:<30}  {yr:>4}  {fr['mean'][idx]:>8.2f}  "
                  f"[{fr['ci_lo'][idx]:>8.2f}, {fr['ci_hi'][idx]:>8.2f}]")
    print()

print("── OUTPUT FILES ───────────────────────────────────────────")
for fname in GENERATED_FILES:
    exists = os.path.exists(fname)
    size   = os.path.getsize(fname) if exists else 0
    mark   = "[OK]" if exists else "[!!]"
    print(f"  {mark} {fname}  ({size:,} bytes)")

print("\n── CACHE ──────────────────────────────────────────────────")
for fname in sorted(os.listdir(CACHE_DIR)):
    fpath = os.path.join(CACHE_DIR, fname)
    print(f"  [OK] {fpath}  ({os.path.getsize(fpath):,} bytes)")

print("\n" + "=" * 62)
print("  Done! Re-run offline — data is cached in ./climate_cache/")
print("=" * 62)
