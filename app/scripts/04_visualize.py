#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
REPORTS_DIR = DATA_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _corr_heatmap(df, cols, title, outpath):
    # keep only requested cols that exist and are numeric
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"[WARN] no requested columns present for: {title}")
        return
    sub = df[cols].select_dtypes(include=["number"]).copy()
    if sub.empty:
        print(f"[WARN] no numeric data for: {title}")
        return
    corr = sub.corr()
    plt.figure(figsize=(min(1.2*len(cols)+2, 18), min(1.2*len(cols)+2, 18)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": .8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

# ---------- NEW: categorical (Cramér's V) heatmap helpers ----------
def _cramers_v_from_table(ct: pd.DataFrame) -> float:
    """
    Compute Cramér's V from a contingency table without SciPy.
    Returns a value in [0, 1]. If table degenerate, returns np.nan/0.
    """
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    if n <= 0:
        return np.nan
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = (row_sum @ col_sum) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((obs - expected) ** 2 / expected)
    r, k = obs.shape
    if min(r, k) < 2:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

def _categorical_assoc_heatmap(df, cols, title, outpath):
    """
    Build pairwise Cramér's V for the given categorical columns and plot a heatmap.
    """
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        print(f"[WARN] need at least 2 categorical columns for: {title}")
        return

    n = len(cols)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            # crosstab ignores NaN by default; that's fine here
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    args = ap.parse_args()

    feat_p = FEATURES_DIR / f"features_{args.sector}.csv"
    df = pd.read_csv(feat_p, parse_dates=["date"])

    # Timeseries plots
    lines = [
        (["close_mean"],            f"{args.sector}_price.png"),
        (["volume_sum","vol_4w"],   f"{args.sector}_volume.png"),
        (["pub_count","pub_4w"],    f"{args.sector}_pubs.png"),
        (["pub_z","vol_z"],         f"{args.sector}_zscores.png"),
    ]
    for cols, fname in lines:
        ax = df.plot(x="date", y=cols, figsize=(12,5), title=f"{args.sector}: {', '.join(cols)}")
        ax.figure.savefig(PLOTS_DIR / fname, bbox_inches="tight"); plt_close(ax)

    # ============================
    # Relationship plots
    # ============================

    # 1) LEVELS (absolute values) — z-scored close_mean vs z-scored pub_count on one axis
    if {"close_mean", "pub_count"}.issubset(df.columns):
        z = lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1)
        df["_close_mean_z"] = z(df["close_mean"])
        df["_pub_count_z"]  = z(df["pub_count"])

        ax = df.plot(
            x="date",
            y=["_close_mean_z", "_pub_count_z"],
            figsize=(12, 5),
            title=f"{args.sector}: Levels (z-scored) — Price vs Publications"
        )
        ax.set_ylabel("Z-score")
        ax.figure.savefig(PLOTS_DIR / f"{args.sector}_levels_price_vs_pubs_z.png", bbox_inches="tight")
        plt_close(ax)
        df.drop(columns=["_close_mean_z", "_pub_count_z"], inplace=True, errors="ignore")
    else:
        print("[WARN] Levels plot skipped: require 'close_mean' and 'pub_count'")

    # 2) GROWTH — ret_1d vs pub_growth (fallback: compute pub_growth if missing)
    need_growth = "pub_growth" not in df.columns
    can_compute_growth = {"pub_count"}.issubset(df.columns)
    if need_growth and can_compute_growth:
        df["pub_growth"] = df["pub_count"].pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    if "ret_1d" in df.columns and "pub_growth" in df.columns:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        df.plot(x="date", y="ret_1d", ax=ax1, label="Daily return (ret_1d)")
        ax1.axhline(0, linestyle="--", linewidth=1, color="black")
        ax1.set_ylabel("Daily return")
        ax1.set_xlabel("Date")

        ax2 = ax1.twinx()
        df.plot(x="date", y="pub_growth", ax=ax2, label="Publications (growth)", linestyle="--")
        ax2.set_ylabel("Publications (growth, pct change)")

        ax1.set_title(f"{args.sector}: Growth — Daily Return vs Publication Growth")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{args.sector}_growth_ret_vs_pub_growth.png", bbox_inches="tight")
        plt.close(fig)
    else:
        print("[WARN] Growth plot skipped: need 'ret_1d' and 'pub_growth' (or 'pub_count' to compute it)")

    # Heatmap 1: full feature subset (counts + shares)
    cols_hm_full = [
        "close_mean","ret_1d",
        "volume_sum","vol_4w","vol_growth","vol_z",
        "pub_count","pub_4w","pub_growth","pub_z",
        "top1_count","top2_count","top3_count","top4_count","top5_count",
        "top1_share","top2_share","top3_share","top4_share","top5_share",
    ]
    _corr_heatmap(
        df,
        cols_hm_full,
        title=f"{args.sector} correlations — all",
        outpath=PLOTS_DIR / f"{args.sector}_corr_all.png",
    )

    # Heatmap 2: compact core subset
    cols_hm_core = [
        "ret_1d","close_mean",
        "vol_4w","vol_growth",
        "pub_4w","pub_growth",
    ]
    _corr_heatmap(
        df,
        cols_hm_core,
        title=f"{args.sector} correlations — core",
        outpath=PLOTS_DIR / f"{args.sector}_corr_core.png",
    )

    # ---------- NEW: Heatmap 3 — categorical associations (Cramér's V) ----------
    cols_cats = ["top1","top2","top3","top4","top5"]
    _categorical_assoc_heatmap(
        df,
        cols_cats,
        title=f"{args.sector} categorical associations — top topics (Cramér's V)",
        outpath=PLOTS_DIR / f"{args.sector}_cat_assoc_top_topics.png",
    )

def plt_close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)

if __name__ == "__main__":
    main()
