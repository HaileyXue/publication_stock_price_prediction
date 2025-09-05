#!/usr/bin/env python3
"""
06_mixedlm_by_sector.py

Aggregate all sector feature files and fit a Mixed Linear Model (Gaussian)
with random intercepts and random slopes (for *all* numeric predictors) by sector.

Inputs:
  data/processed/features/features_<Sector>.csv   (auto-discovered)

Outputs:
  data/reports/models/ALL_mixedlm_by_sector_fullrandslopes.txt     # summary
  data/reports/models/ALL_mixedlm_by_sector_random_effects.csv     # per-sector REs
  data/reports/models/ALL_mixedlm_by_sector_predictions.csv        # per-row preds

Usage:
  python app/scripts/06_mixedlm_by_sector.py
  # optional:
  #   --max-rows-per-sector 3000      # only most recent N rows per sector
  #   --reml/--no-reml                 # fit with REML (default True)

Requires:
  statsmodels>=0.14, scikit-learn
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# ---------------- Paths ----------------
ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / "data"
FEATURES_DIR = DATA_DIR / "processed" / "features"
REPORTS_DIR  = DATA_DIR / "reports"
MODELS_DIR   = REPORTS_DIR / "models"
PLOTS_DIR    = REPORTS_DIR / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Candidate numeric features -------------
# We will take the intersection of these with the actual columns present.
CAND_NUM_COLS = [
    # price & returns
    "close_mean",
    # volumes
    "vol_sum", 
    # publications
    "pub_count", 
]

def discover_sectors():
    return sorted([
        p.stem.replace("features_", "")
        for p in FEATURES_DIR.glob("features_*.csv")
    ])

def load_all_features(max_rows_per_sector=None):
    frames = []
    sectors = discover_sectors()
    if not sectors:
        raise SystemExit(f"No feature files found in {FEATURES_DIR}")
    for s in sectors:
        fp = FEATURES_DIR / f"features_{s}.csv"
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        df["sector"] = s
        if max_rows_per_sector is not None and len(df) > max_rows_per_sector:
            df = df.tail(max_rows_per_sector).copy()
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

def prepare_design(df: pd.DataFrame):
    """
    - Select usable numeric predictors (present in df)
    - Standardize predictors
    - Build fixed-effects matrix X (with intercept)
    - Build random-effects matrix Z with random intercept + random slopes for all numeric predictors
    """
    # Clean numerics (inf → NaN)
    df = df.copy()
    numc_all = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numc_all] = df[numc_all].replace([np.inf, -np.inf], np.nan)

    # Target must be present
    if "ret_fwd_5d" not in df.columns:
        raise ValueError("ret_fwd_5d not found; MixedLM (Gaussian) needs continuous target")

    # Choose numeric predictors that exist (exclude the target itself from predictors)
    use_num = [c for c in CAND_NUM_COLS if c in df.columns and c != "ret_fwd_5d"]
    if not use_num:
        raise ValueError("No numeric predictors found among candidates. Check your feature files.")

    # Drop rows missing target or predictors
    keep_cols = ["sector", "date", "ret_fwd_5d"] + use_num
    df = df.dropna(subset=keep_cols).sort_values(["sector", "date"]).reset_index(drop=True)

    # Standardize predictors (helps optimizer & interpretability of random slopes)
    scaler = StandardScaler()
    df[use_num] = scaler.fit_transform(df[use_num].astype(float))

    # Fixed effects (global): intercept + all numeric predictors
    X = sm.add_constant(df[use_num])

    # Random effects (by sector): random intercept + random slopes for ALL numeric predictors
    Z = df[use_num].copy()
    Z.insert(0, "Intercept_RE", 1.0)  # random intercept column first

    y = df["ret_fwd_5d"].astype(float)
    groups = df["sector"].astype(str)

    return df, y, X, Z, groups, use_num

def fit_mixedlm_all_random_slopes(df, y, X, Z, groups, reml=True):
    """
    Fit MixedLM with:
      - Fixed effects: X (const + all numeric predictors)
      - Random effects: Z (intercept + all numeric predictors as random slopes) grouped by sector
    """
    print("[MixedLM] Fitting with random intercept + random slopes for ALL numeric predictors by sector …")
    md = sm.MixedLM(endog=y, exog=X, groups=groups, exog_re=Z)
    # Note: statsmodels MixedLM uses REML by default for Gaussian; can toggle via fit(reml=False)
    res = md.fit(method="lbfgs", maxiter=500, reml=reml, disp=False)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows-per-sector", type=int, default=None,
                    help="If set, only keep this many most recent rows per sector before fitting")
    ap.add_argument("--reml", dest="reml", action="store_true", default=True,
                    help="Fit with REML (default True)")
    ap.add_argument("--no-reml", dest="reml", action="store_false",
                    help="Fit with ML (REML=False)")
    args = ap.parse_args()

    # Load & design
    df_all = load_all_features(max_rows_per_sector=args.max_rows_per_sector)
    df_all, y, X, Z, groups, used_num_cols = prepare_design(df_all)

    # Fit
    try:
        res = fit_mixedlm_all_random_slopes(df_all, y, X, Z, groups, reml=args.reml)
    except Exception as e:
        # Fallback: if full random slope model is too heavy, try random intercept only
        print(f"[WARN] Full random-slopes fit failed ({e}). Falling back to random intercept only.")
        Z_fallback = pd.DataFrame({"Intercept_RE": np.ones(len(df_all))})
        res = sm.MixedLM(endog=y, exog=X, groups=groups, exog_re=Z_fallback).fit(
            method="lbfgs", maxiter=500, reml=args.reml, disp=False
        )

    # ---- Save outputs
    tag = "ALL_mixedlm_by_sector_fullrandslopes" if "Intercept_RE" in Z.columns and len(Z.columns) > 1 else "ALL_mixedlm_by_sector_randint"
    # 1) Summary
    summary_path = MODELS_DIR / f"{tag}.txt"
    summary_path.write_text(res.summary().as_text(), encoding="utf-8")
    print(f"[Write] {summary_path}")

    # 2) Random effects (per-sector)
    # res.random_effects is a dict: {group: Series of RE coefficients}
    re_rows = []
    for sector, coef in res.random_effects.items():
        row = {"sector": sector}
        # Ensure consistent ordering across sectors
        for col in Z.columns:
            row[f"RE[{col}]"] = float(coef.get(col, np.nan))
        re_rows.append(row)
    re_df = pd.DataFrame(re_rows).sort_values("sector")
    re_path = MODELS_DIR / f"{tag}_random_effects.csv"
    re_df.to_csv(re_path, index=False)
    print(f"[Write] {re_path}")

    # 3) Predictions for each row = fixed part + sector-specific random part
    # Fixed component
    fe = res.fe_params  # pandas Series indexed by X columns (including "const")
    fe = fe.reindex(X.columns)  # align just in case
    fixed_pred = np.asarray(X) @ np.asarray(fe)

    # Random component per sector: build a matrix of Z · b(sector)
    # res.random_effects is a dict: {sector: Series([...])}, keys should match Z columns
    Z_cols = list(Z.columns)
    re_lookup = {}
    for sector, coef in res.random_effects.items():
        # align to Z columns; missing keys -> 0
        vec = np.array([coef.get(col, 0.0) for col in Z_cols], dtype=float)
        re_lookup[sector] = vec

    # Now compute random contribution row-by-row
    rand_pred = np.zeros(len(df_all), dtype=float)
    z_mat = np.asarray(Z)
    sectors_arr = df_all["sector"].astype(str).values
    for i in range(len(df_all)):
        vec = re_lookup.get(sectors_arr[i])
        if vec is not None:
            rand_pred[i] = z_mat[i, :].dot(vec)  # Z_i · b(sector_i)
        # else no RE for this sector -> contributes 0

    preds = fixed_pred + rand_pred

    pred_df = pd.DataFrame({
        "sector": df_all["sector"].values,
        "date": df_all["date"].values,
        "ret_fwd_5d": y.values,
        "pred": preds
    })
    pred_path = MODELS_DIR / f"{tag}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[Write] {pred_path}")

    # 4) Small console recap
    print("\n[Model recap]")
    print(f"- Sectors: {df_all['sector'].nunique()}")
    print(f"- Rows:    {len(df_all)}")
    print(f"- Predictors (numeric, standardized): {used_num_cols}")
    print(f"- Random effects columns: {list(re_df.columns)}")
    print(f"- REML: {args.reml}")

if __name__ == "__main__":
    main()
