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

def _tag(start: str, end: str) -> str:
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    return f"{s}-{e}"

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

def build_features(sector: str, start: str, end: str) -> pd.DataFrame:
    tag = _tag(start, end)
    px_path   = RAW_PRICES_DIR / f"sector_{sector}_{tag}_daily_agg.csv"
    counts_p  = TOPICS_DIR / f"daily_topic_counts_{sector}_{tag}.csv"
    wide_p    = TOPICS_DIR / f"daily_top5_wide_{sector}_{tag}.csv"
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

    # (D) publication features + label
    df["pub_4w"]     = df["pub_count"].rolling(20, min_periods=5).sum()
    df["pub_growth"] = safe_pct_change(df["pub_count"])
    roll_mean        = df["pub_count"].rolling(126, min_periods=30).mean()
    roll_std         = df["pub_count"].rolling(126, min_periods=30).std()
    df["pub_z"]      = (df["pub_count"] - roll_mean) / roll_std
    df["y_up_5d"]    = (df["ret_fwd_5d"] > 0).astype("Int64")

    # (E) per-topic counts (calendar) → trading day
    mapped_counts = _map_calendar_to_trading(counts, "date", px, tolerance_days=4)
    trade_topic_counts = (mapped_counts.groupby(["trade_date","topic"], as_index=False)["count"]
                          .sum().rename(columns={"trade_date":"date"}))

    # --------- Daily Top-5 per trading day (names + counts) ----------
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

    top5_daily = topN_wide(trade_topic_counts, N=5)

    # --------- NEW: 4-week Top-5 topics, counts, shares (trading-day aligned) ----------
    # Build a full date x topic matrix of daily counts (zeros for days with no occurrences)
    all_dates = df["date"]
    pivot = (trade_topic_counts.pivot_table(index="date", columns="topic", values="count", fill_value=0)
                              .reindex(all_dates, fill_value=0))

    # Rolling 20 trading-day sum per topic (min 5 days)
    roll_4w = pivot.rolling(20, min_periods=5).sum()

    # For each date, take Top-5 topics by 4w sum
    records = []
    for dt, row in roll_4w.iterrows():
        # keep only positive totals
        row_pos = row[row > 0].sort_values(ascending=False).head(5)
        rec = {"date": dt}
        for i, (name, val) in enumerate(row_pos.items(), start=1):
            rec[f"top{i}_4w"] = name
            rec[f"top{i}_4w_count"] = float(val)
        records.append(rec)
    top4w = pd.DataFrame(records).sort_values("date")

    # Ensure all expected columns exist
    for i in range(1, 6):
        if f"top{i}_4w" not in top4w.columns:        top4w[f"top{i}_4w"] = pd.NA
        if f"top{i}_4w_count" not in top4w.columns:  top4w[f"top{i}_4w_count"] = np.nan

    # Build top_list_4w for convenience
    topic_cols_4w = [f"top{i}_4w" for i in range(1,6)]
    top4w["top_list_4w"] = (top4w[topic_cols_4w].fillna("")
                            .apply(lambda r: "|".join([t for t in r if t]), axis=1))

    # Merge 4w tops back to main df and compute shares vs pub_4w
    out = df.merge(top5_daily, on="date", how="left").merge(top4w, on="date", how="left")

    # Daily shares (already) and 4w shares (new)
    denom_daily = out["pub_count"].astype(float).where(out["pub_count"] > 0)
    for i in range(1,6):
        ccol = f"top{i}_count"
        scol = f"top{i}_share"
        if ccol in out.columns:
            out[scol] = (out[ccol].astype(float) / denom_daily)

    denom_4w = out["pub_4w"].astype(float).where(out["pub_4w"] > 0)
    for i in range(1,6):
        c4 = f"top{i}_4w_count"
        s4 = f"top{i}_4w_share"
        if c4 in out.columns:
            out[s4] = (out[c4].astype(float) / denom_4w)

    # Cleanup & ordering
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)

    ordered = [
        "date","pub_count","pub_4w","pub_growth","pub_z",
        "close_mean","ret_5d","ret_fwd_5d",
        "volume_sum","vol_4w","vol_growth","vol_z",
        "y_up_5d",
        # daily top-k
        "top1","top2","top3","top4","top5",
        "top1_count","top2_count","top3_count","top4_count","top5_count",
        "top1_share","top2_share","top3_share","top4_share","top5_share",
        "total_publications","top_list",
        # 4w top-k
        "top1_4w","top2_4w","top3_4w","top4_4w","top5_4w",
        "top1_4w_count","top2_4w_count","top3_4w_count","top4_4w_count","top5_4w_count",
        "top1_4w_share","top2_4w_share","top3_4w_share","top4_4w_share","top5_4w_share",
        "top_list_4w"
    ]
    ordered = [c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]
    out = out.loc[:, ordered]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    ap.add_argument("--start",  required=True)
    ap.add_argument("--end",    required=True)
    args = ap.parse_args()

    out = build_features(args.sector, args.start, args.end)
    tag = _tag(args.start, args.end)
    out_path = FEATURES_DIR / f"features_{args.sector}_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"[Features] wrote {out_path} | rows={len(out)} | cols={len(out.columns)}")

if __name__ == "__main__":
    main()
