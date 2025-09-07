#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
REPORTS_DIR = DATA_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _tag(start: str, end: str) -> str:
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    return f"{s}-{e}"

def _corr_heatmap(df, cols, title, outpath):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"[WARN] no requested columns present for: {title}")
        return
    sub = df[cols].select_dtypes(include=["number"]).copy()
    if sub.empty:
        print(f"[WARN] no numeric data for: {title}")
        return

    corr = sub.corr()
    mask = corr.isna()

    cmap = mpl.colormaps['coolwarm'].copy()
    cmap.set_bad(color='#d9d9d9')  # grey for NaNs

    plt.figure(figsize=(min(1.2*len(cols)+2, 18), min(1.2*len(cols)+2, 18)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, square=True,
                cbar_kws={"shrink": .8}, vmin=-1, vmax=1, mask=mask)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

# ---------- categorical (Cramér's V) helpers ----------
def _cramers_v_from_table(ct: pd.DataFrame) -> float:
    obs = ct.to_numpy(dtype=float); n = obs.sum()
    if n <= 0: return np.nan
    row_sum = obs.sum(axis=1, keepdims=True); col_sum = obs.sum(axis=0, keepdims=True)
    expected = (row_sum @ col_sum) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((obs - expected) ** 2 / expected)
    r, k = obs.shape
    if min(r, k) < 2: return 0.0
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

def _categorical_assoc_heatmap(df, cols, title, outpath):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        print(f"[WARN] need at least 2 categorical columns for: {title}")
        return
    n = len(cols)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            ct = pd.crosstab(df[cols[i]], df[cols[j]])
            val = _cramers_v_from_table(ct) if not ct.empty else np.nan
            M[i, j] = M[j, i] = val
    assoc = pd.DataFrame(M, index=cols, columns=cols)
    plt.figure(figsize=(min(1.2*len(cols)+2, 18), min(1.2*len(cols)+2, 18)))
    sns.heatmap(assoc, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, square=True,
                cbar_kws={"shrink": .8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def _ensure_pub_growth(df: pd.DataFrame) -> pd.DataFrame:
    if "pub_growth" not in df.columns and "pub_count" in df.columns:
        g = df["pub_count"].astype(float).pct_change()
        g = g.replace([np.inf, -np.inf], np.nan).fillna(0)
        df["pub_growth"] = g
    return df

def _granger_pub_growth_vs_ret5d(df: pd.DataFrame, sector: str, tag: str, max_lag: int = 10):
    """
    Runs Granger causality tests between pub_growth (g) and ret_fwd_5d (r).
    Writes CSV of p-values across lags and a line plot PNG.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:
        print("[WARN] statsmodels not available; skipping Granger plots.")
        return

    if not {"pub_growth", "ret_fwd_5d"}.issubset(df.columns):
        print("[WARN] Granger skipped: need 'pub_growth' and 'ret_fwd_5d'")
        return

    d = df[["pub_growth", "ret_fwd_5d"]].dropna().copy()
    if len(d) < 40:
        print("[WARN] Too few rows for Granger; skipping.")
        return

    # Build arrays for both directions
    p_g2r = []
    p_r2g = []
    for lag in range(1, max_lag + 1):
        try:
            res1 = grangercausalitytests(d[["ret_fwd_5d", "pub_growth"]], maxlag=lag, verbose=False)
            p_g2r.append(res1[lag][0]["ssr_ftest"][1])
        except Exception:
            p_g2r.append(np.nan)
        try:
            res2 = grangercausalitytests(d[["pub_growth", "ret_fwd_5d"]], maxlag=lag, verbose=False)
            p_r2g.append(res2[lag][0]["ssr_ftest"][1])
        except Exception:
            p_r2g.append(np.nan)

    out_csv = PLOTS_DIR / f"{sector}_{tag}_granger_pubGrowth_vs_ret5d.csv"
    pd.DataFrame({"lag": np.arange(1, max_lag+1), "p_g_to_r": p_g2r, "p_r_to_g": p_r2g}).to_csv(out_csv, index=False)

    # Plot p-values across lags
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, max_lag+1), p_g2r, marker="o", label="pub_growth → ret_fwd_5d")
    plt.plot(np.arange(1, max_lag+1), p_r2g, marker="o", label="ret_fwd_5d → pub_growth")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1, label="α = 0.05")
    plt.xlabel("Lag"); plt.ylabel("p-value")
    plt.title(f"{sector} {tag}: Granger p-values")
    plt.legend(loc="best")
    plt.tight_layout()
    out_png = PLOTS_DIR / f"{sector}_{tag}_granger_pubGrowth_vs_ret5d.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"[Granger] wrote {out_csv.name} and {out_png.name}")

def _lead_lag_heatmap(df: pd.DataFrame, sector: str, tag: str, max_lag: int = 20):
    """
    Builds a lead–lag correlation heatmap between target ret_fwd_5d and
    drivers in ['pub_growth','pub_4w','pub_count'] across lags -max_lag..+max_lag.
    Positive lag means driver leads returns by 'lag' days.
    """
    targets = ["ret_fwd_5d"]
    drivers = [c for c in ["pub_growth", "pub_4w", "pub_count"] if c in df.columns]
    if not drivers or "ret_fwd_5d" not in df.columns:
        print("[WARN] Lead–lag skipped: need ret_fwd_5d and at least one of pub_growth/pub_4w/pub_count")
        return

    d0 = df.copy()
    # Standardize to reduce scale issues
    for c in list(set(drivers + targets)):
        if c in d0.columns:
            d0[c] = (d0[c] - d0[c].mean()) / (d0[c].std(ddof=0) or 1.0)

    lags = np.arange(-max_lag, max_lag + 1)
    mat = np.zeros((len(drivers), len(lags))) * np.nan

    for i, drv in enumerate(drivers):
        for j, lag in enumerate(lags):
            # positive lag: driver leads (shift driver forward)
            shifted = d0[drv].shift(lag * -1)  # invert sign so +lag aligns driver earlier vs returns
            corr = pd.concat([d0["ret_fwd_5d"], shifted], axis=1).dropna().corr().iloc[0,1]
            mat[i, j] = corr

    plt.figure(figsize=(min(2 + 0.4*len(lags), 18), min(2 + 0.6*len(drivers), 12)))
    sns.heatmap(pd.DataFrame(mat, index=drivers, columns=lags),
                cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=False,
                cbar_kws={"shrink": .8})
    plt.xlabel("Lag (days)  —  positive = driver leads")
    plt.ylabel("Driver")
    plt.title(f"{sector} {tag}: Lead–Lag correlation vs ret_fwd_5d")
    plt.tight_layout()
    out_png = PLOTS_DIR / f"{sector}_{tag}_leadlag_heatmap.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"[Lead-Lag] wrote {out_png.name}")

def plt_close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    ap.add_argument("--start",  required=True)  # YYYY-MM-DD
    ap.add_argument("--end",    required=True)  # YYYY-MM-DD
    args = ap.parse_args()

    tag = _tag(args.start, args.end)

    feat_p = FEATURES_DIR / f"features_{args.sector}_{tag}.csv"
    if not feat_p.exists():
        print(f"[WARN] Features not found for {args.sector} {tag}: {feat_p}")
        return
    df = pd.read_csv(feat_p, parse_dates=["date"])

    # scope to date range (redundant but safe)
    mask = (df["date"] >= pd.to_datetime(args.start)) & (df["date"] <= pd.to_datetime(args.end))
    df = df.loc[mask].copy()
    if df.empty:
        print(f"[WARN] No rows for {args.sector} in {tag}; skipping plots.")
        return

    # 1) close_mean (price)
    if "close_mean" in df.columns:
        ax = df.plot(x="date", y=["close_mean"], figsize=(12,5),
                     title=f"{args.sector} {tag}: close_mean")
        ax.figure.savefig(PLOTS_DIR / f"{args.sector}_{tag}_price.png", bbox_inches="tight"); plt_close(ax)
    else:
        print("[WARN] Missing 'close_mean' for price plot")

    # 2) close_mean & pub_count (colored)
    if {"close_mean","pub_count"}.issubset(df.columns):
        fig, ax1 = plt.subplots(figsize=(12,5))
        ax1.plot(df["date"], df["close_mean"], label="close_mean", color="tab:blue")
        ax1.set_ylabel("close_mean", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_xlabel("Date")

        ax2 = ax1.twinx()
        ax2.plot(df["date"], df["pub_count"], label="pub_count", color="tab:orange", linestyle="--")
        ax2.set_ylabel("pub_count", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax1.set_title(f"{args.sector} {tag}: close_mean vs pub_count")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{args.sector}_{tag}_levels_price_vs_pubs.png", bbox_inches="tight")
        plt.close(fig)
    else:
        print("[WARN] Levels plot skipped: need 'close_mean' and 'pub_count'")

    # Ensure pub_growth exists for later analytics
    df = _ensure_pub_growth(df)

    # 3) ret_fwd_5d vs pub_growth (colored)
    if {"ret_fwd_5d", "pub_growth"}.issubset(df.columns):
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df["date"], df["ret_fwd_5d"],
                 label="5d forward return (ret_fwd_5d)", color="#1f77b4")
        ax1.axhline(0, linestyle="--", linewidth=1, color="black")
        ax1.set_xlabel("Date"); ax1.set_ylabel("5d forward return", color="#1f77b4")
        ax1.tick_params(axis="y", colors="#1f77b4")

        ax2 = ax1.twinx()
        ax2.plot(df["date"], df["pub_growth"], linestyle="--",
                 label="Publication growth (pub_growth)", color="#ff7f0e")
        ax2.set_ylabel("Publication growth (pct change)", color="#ff7f0e")
        ax2.tick_params(axis="y", colors="#ff7f0e")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")
        ax1.set_title(f"{args.sector} {tag}: 5d return vs publication growth")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{args.sector}_{tag}_ret5d_vs_pub_growth.png", bbox_inches="tight")
        plt.close(fig)
    else:
        print("[WARN] Plot skipped: need 'ret_fwd_5d' and 'pub_growth' (or 'pub_count' to compute it)")

    # 4) ret_fwd_5d vs pub_4w (colored)
    if {"ret_fwd_5d", "pub_4w"}.issubset(df.columns):
        fig, ax1b = plt.subplots(figsize=(12, 5))
        ax1b.plot(df["date"], df["ret_fwd_5d"],
                  label="5d forward return (ret_fwd_5d)", color="#1f77b4")
        ax1b.axhline(0, linestyle="--", linewidth=1, color="black")
        ax1b.set_xlabel("Date"); ax1b.set_ylabel("5d forward return", color="#1f77b4")
        ax1b.tick_params(axis="y", colors="#1f77b4")

        ax2b = ax1b.twinx()
        ax2b.plot(df["date"], df["pub_4w"], linestyle="--",
                  label="Publications 4w (pub_4w)", color="#ff7f0e")
        ax2b.set_ylabel("Publications (4-week sum)", color="#ff7f0e")
        ax2b.tick_params(axis="y", colors="#ff7f0e")

        h1b, l1b = ax1b.get_legend_handles_labels()
        h2b, l2b = ax2b.get_legend_handles_labels()
        ax1b.legend(h1b + h2b, l1b + l2b, loc="upper left")
        ax1b.set_title(f"{args.sector} {tag}: 5d return vs 4w publications")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{args.sector}_{tag}_ret5d_vs_pub4w.png", bbox_inches="tight")
        plt.close(fig)
    else:
        print("[WARN] Plot skipped: need 'ret_fwd_5d' and 'pub_4w'")

    # Class balance bar plot
    if "y_up_5d" in df.columns:
        lbl = df["y_up_5d"].dropna().astype(int)
        counts = lbl.value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(["Class 0", "Class 1"], [counts.get(0, 0), counts.get(1, 0)])
        ax.set_title(f"{args.sector} {tag}: Class distribution (y_up_5d)")
        ax.set_ylabel("Count")
        for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
            ax.text(i, v, f"{v}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{args.sector}_{tag}_class_balance.png", bbox_inches="tight")
        plt.close(fig)
    else:
        print("[WARN] Class balance plot skipped: 'y_up_5d' not found")

    # ---- NEW: Violin plot of pub_4w by y_up_5d ----
    if {"pub_4w", "y_up_5d"}.issubset(df.columns):
        dvi = df[["pub_4w", "y_up_5d"]].dropna().copy()
        dvi["y_up_5d"] = dvi["y_up_5d"].astype(int).astype(str)
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=dvi, x="y_up_5d", y="pub_4w", inner="box")
        plt.xlabel("y_up_5d (class)"); plt.ylabel("pub_4w")
        plt.title(f"{args.sector} {tag}: pub_4w by class (violin)")
        plt.tight_layout()
        out_png = PLOTS_DIR / f"{args.sector}_{tag}_violin_pub4w_by_y.png"
        plt.savefig(out_png, bbox_inches="tight"); plt.close()
        print(f"[Violin] wrote {out_png.name}")
    else:
        print("[WARN] Violin skipped: need 'pub_4w' and 'y_up_5d'")

    # ---- NEW: Granger p-values (pub_growth ↔ ret_fwd_5d) ----
    _granger_pub_growth_vs_ret5d(df, args.sector, tag, max_lag=10)

    # ---- NEW: Lead–Lag heatmap (drivers vs ret_fwd_5d) ----
    _lead_lag_heatmap(df, args.sector, tag, max_lag=20)

    # Heatmap 1 — all
    cols_hm_full = [
        "close_mean","ret_5d",
        "volume_sum","vol_4w","vol_growth",
        "pub_count","pub_4w","pub_growth",
        "top1_4w_count","top2_4w_count","top3_4w_count","top4_4w_count","top5_4w_count",
        "top1_4w_share","top2_4w_share","top3_4w_share","top4_4w_share","top5_4w_share",
    ]
    _corr_heatmap(df, cols_hm_full, title=f"{args.sector} {tag} correlations — all",
                  outpath=PLOTS_DIR / f"{args.sector}_{tag}_corr_all.png")

    # Heatmap 2 — core
    cols_hm_core = ["ret_5d", "vol_4w","vol_growth","pub_4w", "pub_growth",
                    "top3_4w_share", "top5_4w_share"]
    _corr_heatmap(df, cols_hm_core, title=f"{args.sector} {tag} correlations — core",
                  outpath=PLOTS_DIR / f"{args.sector}_{tag}_corr_core.png")

    # Categorical associations (last)
    cols_cats = ["top1","top2","top3","top4","top5",
                 "top1_4w","top2_4w","top3_4w","top4_4w","top5_4w"]
    _categorical_assoc_heatmap(df, cols_cats,
                               title=f"{args.sector} {tag} categorical associations — top topics (Cramér's V)",
                               outpath=PLOTS_DIR / f"{args.sector}_{tag}_cat_assoc_top_topics.png")

if __name__ == "__main__":
    main()
