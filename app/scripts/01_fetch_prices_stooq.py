#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yaml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "prices"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def _tag(start: str, end: str) -> str:
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    return f"{s}-{e}"

def fetch_stooq_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV; returns columns: date, close, volume, ticker."""
    tried = []
    for sym in (ticker, f"{ticker}.US", ticker.lower(), f"{ticker.lower()}.us"):
        tried.append(sym)
        try:
            df = pdr.DataReader(sym, "stooq", start=start, end=end)
            if df is not None and not df.empty:
                df = df.sort_index().reset_index()
                df = df.rename(columns={"Date":"date","Close":"close","Volume":"volume"})
                out = df[["date","close","volume"]].copy()
                out["ticker"] = ticker
                return out
        except Exception:
            continue
    raise RuntimeError(f"Stooq fetch failed for {ticker}. Tried: {tried}")

def build_sector_daily_agg(tickers, start, end, out_path: Path) -> pd.DataFrame:
    """Outer-join by date, equal-weight close, sum volume, build features."""
    frames = []
    req_start = pd.to_datetime(start)
    req_end   = pd.to_datetime(end)

    for t in tickers:
        p = RAW_PRICES_DIR / f"{t}.csv"

        def _read_cache(path: Path):
            if not path.exists():
                return pd.DataFrame(columns=["date","close","volume","ticker"])
            dfc = pd.read_csv(path, parse_dates=["date"])
            return dfc.sort_values("date").reset_index(drop=True)

        def _merge_and_write(path: Path, old_df: pd.DataFrame, add_df: pd.DataFrame):
            if add_df is None or add_df.empty:
                return old_df
            merged = (pd.concat([old_df, add_df], ignore_index=True)
                        .drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True))
            path.write_text(merged.to_csv(index=False))
            return merged

        # 1) Read cache (may be empty)
        cached = _read_cache(p)

        # 2) Ensure coverage
        need_fetch = False
        if cached.empty:
            need_fetch = True
        else:
            have_start = cached["date"].min()
            have_end   = cached["date"].max()
            if req_start < have_start or req_end > have_end:
                need_fetch = True

        if need_fetch:
            try:
                fresh = fetch_stooq_ticker(t, start, end)
            except Exception as e:
                print(f"[WARN] {t}: {e}", file=sys.stderr)
                # If we have *some* cache, proceed with it; otherwise skip ticker.
                if cached.empty:
                    continue
                fresh = None
            cached = _merge_and_write(p, cached, fresh)

        # 3) Subset to the requested window and rename for aggregation
        df = cached.loc[
            (cached["date"] >= req_start) & (cached["date"] <= req_end),
            ["date","close","volume"]
        ].rename(columns={"close": f"close_{t}", "volume": f"vol_{t}"})

        if df.empty:
            print(f"[WARN] {t}: no rows after subsetting {start}..{end}")
            continue

        frames.append(df)


    if not frames:
        raise FileNotFoundError("No per-ticker price files found for sector.")

    agg = frames[0]
    for f in frames[1:]:
        agg = agg.merge(f, on="date", how="outer")

    agg = agg.sort_values("date").reset_index(drop=True)
    close_cols = [c for c in agg.columns if c.startswith("close_")]
    vol_cols   = [c for c in agg.columns if c.startswith("vol_")]

    agg["close_mean"] = agg[close_cols].mean(axis=1, skipna=True)
    agg["volume_sum"] = agg[vol_cols].sum(axis=1, skipna=True) if vol_cols else 0.0
    agg = agg.dropna(subset=["close_mean"])
    agg["volume_sum"] = agg["volume_sum"].fillna(0)

    # Returns
    agg["ret_1d"]     = agg["close_mean"].pct_change()
    agg["ret_fwd_5d"] = agg["close_mean"].shift(-5)/agg["close_mean"] - 1

    # Volume features
    lv = np.log1p(agg["volume_sum"])
    agg["vol_4w"]     = agg["volume_sum"].rolling(20, min_periods=5).sum()
    agg["vol_growth"] = agg["volume_sum"].pct_change().replace([np.inf, -np.inf], np.nan)
    roll_mean         = lv.rolling(126, min_periods=30).mean()
    roll_std          = lv.rolling(126, min_periods=30).std()
    agg["vol_z"]      = (lv - roll_mean) / roll_std

    out = agg[["date","close_mean","ret_1d","ret_fwd_5d","volume_sum","vol_4w","vol_growth","vol_z"]]
    out.to_csv(out_path, index=False)
    print(f"[Prices] wrote {out_path} | rows={len(out)}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    ap.add_argument("--start",  required=True)  # YYYY-MM-DD
    ap.add_argument("--end",    required=True)  # YYYY-MM-DD
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "sector_map.yaml"))
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.sector not in cfg:
        raise SystemExit(f"sector must be one of: {list(cfg.keys())}")

    # cap window to <= 3 years
    start_dt = pd.to_datetime(args.start)
    end_dt   = pd.to_datetime(args.end)
    if end_dt < start_dt:
        raise SystemExit("end date must be >= start date")
    if end_dt - start_dt > pd.Timedelta(days=365*3+2):
        start_dt = end_dt - pd.Timedelta(days=365*3)
        print(f"[WARN] clipping start to {start_dt.date()} (max 3y)")

    tickers = cfg[args.sector]["tickers"]
    tag = _tag(start_dt, end_dt)
    out_path = RAW_PRICES_DIR / f"sector_{args.sector}_{tag}_daily_agg.csv"
    build_sector_daily_agg(tickers, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), out_path)

if __name__ == "__main__":
    main()
