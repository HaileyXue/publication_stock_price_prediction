#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "prices"
PROCESSED_DIR = DATA_DIR / "processed"
TOPICS_DIR = PROCESSED_DIR / "topics"
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def _map_calendar_to_trading(left_df, left_date_col, px_dates, tolerance_days=4):
    trade_ref = px_dates[["date"]].rename(columns={"date":"trade_date"}).sort_values("trade_date")
    mapped = pd.merge_asof(
        left=left_df.sort_values(left_date_col),
        right=trade_ref,
        left_on=left_date_col,
        right_on="trade_date",
        direction="forward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    return mapped.dropna(subset=["trade_date"])

def safe_pct_change(s: pd.Series):
    return s.pct_change().replace([np.inf, -np.inf], np.nan)

def build_features(sector: str) -> pd.DataFrame:
    px_path   = RAW_PRICES_DIR / f"sector_{sector}_daily_agg.csv"
    counts_p  = TOPICS_DIR / f"daily_topic_counts_{sector}.csv"
    wide_p    = TOPICS_DIR / f"daily_top5_wide_{sector}.csv"
    if not px_path.exists():   raise FileNotFoundError(px_path)
    if not counts_p.exists():  raise FileNotFoundError(counts_p)
    if not wide_p.exists():    raise FileNotFoundError(wide_p)

    px = pd.read_csv(px_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    counts = pd.read_csv(counts_p, parse_dates=["date"])
    wide   = pd.read_csv(wide_p,   parse_dates=["date"])

    # (A) total publications per CALENDAR day → map to TRADING day
    total_pub_calendar = wide[["date","total_publications"]].dropna()
    mapped_total = _map_calendar_to_trading(total_pub_calendar, "date", px, tolerance_days=4)
    pub_trading = (mapped_total.groupby("trade_date", as_index=False)["total_publications"]
                   .sum().rename(columns={"trade_date":"date","total_publications":"pub_count"}))

    # (B) zero-fill to full trading-day index
    all_trading = px[["date"]].copy()
    pub_trading = all_trading.merge(pub_trading, on="date", how="left")
    pub_trading["pub_count"] = pub_trading["pub_count"].fillna(0).astype(int)

    # (C) merge with prices
    df = pub_trading.merge(px, on="date", how="left").sort_values("date").reset_index(drop=True)

    # (D) pub features + label
    df["pub_4w"]     = df["pub_count"].rolling(20, min_periods=5).sum()
    df["pub_growth"] = safe_pct_change(df["pub_count"])
    roll_mean        = df["pub_count"].rolling(126, min_periods=30).mean()
    roll_std         = df["pub_count"].rolling(126, min_periods=30).std()
    df["pub_z"]      = (df["pub_count"] - roll_mean) / roll_std
    df["y_up_5d"]    = (df["ret_fwd_5d"] > 0).astype("Int64")

    # (E) per-topic counts (calendar) → trading day; build trading-day Top-5 wide
    mapped_counts = _map_calendar_to_trading(counts, "date", px, tolerance_days=4)
    trade_topic_counts = (mapped_counts.groupby(["trade_date","topic"], as_index=False)["count"]
                          .sum().rename(columns={"trade_date":"date"}))

    # Build Top-5 per trading day (names + counts)
    def topN_wide(counts_df, N=5):
        if counts_df.empty:
            cols = ["date"] + [f"top{i}" for i in range(1,N+1)] + [f"top{i}_count" for i in range(1,N+1)] + \
                   ["total_publications","top_list"]
            return pd.DataFrame(columns=cols)
        daily_total = counts_df.groupby("date", as_index=False)["count"].sum().rename(columns={"count":"total_publications"})
        topN = (counts_df.sort_values(["date","count","topic"], ascending=[True, False, True])
                        .groupby("date", as_index=False).head(N).copy())
        topN["rank"] = topN.groupby("date")["count"].rank(method="first", ascending=False).astype(int)
        topN["k_topic"] = "top" + topN["rank"].astype(str)
        topN["k_count"] = topN["k_topic"] + "_count"
        T = topN.pivot(index="date", columns="k_topic", values="topic").reset_index().rename_axis(None, axis=1)
        C = topN.pivot(index="date", columns="k_count", values="count").reset_index().rename_axis(None, axis=1)
        W = T.merge(C, on="date", how="outer").merge(daily_total, on="date", how="left")
        for i in range(1,6):
            if f"top{i}" not in W.columns: W[f"top{i}"]=pd.NA
            if f"top{i}_count" not in W.columns: W[f"top{i}_count"]=pd.NA
        W["top_list"] = (W[[f"top{i}" for i in range(1,6)]].fillna("")
                         .apply(lambda r: "|".join([t for t in r if t]), axis=1))
        keep = ["date"] + [f"top{i}" for i in range(1,6)] + [f"top{i}_count" for i in range(1,6)] + \
               ["total_publications","top_list"]
        return W[keep].sort_values("date")

    top5 = topN_wide(trade_topic_counts, N=5)

    # (F) merge top-5 + compute shares safely using pub_count
    out = df.merge(top5, on="date", how="left")
    denom = out["pub_count"].astype(float).where(out["pub_count"] > 0)
    for i in range(1,6):
        ccol, scol = f"top{i}_count", f"top{i}_share"
        if ccol in out.columns:
            out[scol] = (out[ccol].astype(float) / denom)

    # (G) final cleanup (replace inf with NaN) and column ordering
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf,-np.inf], np.nan)

    ordered = ["date","pub_count","close_mean","ret_1d","ret_fwd_5d",
               "volume_sum","vol_4w","vol_growth","vol_z",
               "pub_4w","pub_growth","pub_z","y_up_5d",
               "top1","top2","top3","top4","top5",
               "top1_count","top2_count","top3_count","top4_count","top5_count",
               "total_publications",
               "top1_share","top2_share","top3_share","top4_share","top5_share",
               "top_list"]
    # keep any extras at the end
    ordered = [c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]
    out = out.loc[:, ordered]

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    args = ap.parse_args()

    out = build_features(args.sector)
    out_path = FEATURES_DIR / f"features_{args.sector}.csv"
    out.to_csv(out_path, index=False)
    print(f"[Features] wrote {out_path} | rows={len(out)} | cols={len(out.columns)}")

if __name__ == "__main__":
    main()
