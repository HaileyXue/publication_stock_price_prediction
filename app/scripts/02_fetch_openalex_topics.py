#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import pandas as pd
import yaml
from itertools import chain
from pyalex import Works, Topics

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TOPICS_DIR = PROCESSED_DIR / "topics"
TOPICS_DIR.mkdir(parents=True, exist_ok=True)

def resolve_topic_id(display_name: str):
    cand = Topics().search(display_name).get()
    if not cand:
        raise ValueError(f"No Topic found for {display_name!r}")
    for t in cand:
        if (t.get("display_name","") or "").strip().lower() == display_name.strip().lower():
            return t["id"], t
    return cand[0]["id"], cand[0]

def fetch_works_with_topics(topic_id: str, start: str, end: str, primary_only=True, per_page=200, n_max=None):
    base = (Works()
            .filter(from_publication_date=start, to_publication_date=end)
            .select(["id","publication_date","topics"]))
    q = base.filter(primary_topic={"id": topic_id}) if primary_only else base.filter(topics={"id": topic_id})
    pages = q.paginate(per_page=per_page, n_max=n_max)
    return list(chain(*pages))

def normalize_topics_list(rec):
    ts = rec.get("topics") or []
    names = set()
    for t in ts:
        name = (t.get("display_name") or "").strip()
        if name:
            names.add(name)
    return sorted(names)

def build_topic_tables(records: list, N=5):
    if not records:
        empty_counts = pd.DataFrame(columns=["date","topic","count"])
        empty_wide = pd.DataFrame(columns=["date"] +
                                  [f"top{i}" for i in range(1,N+1)] +
                                  [f"top{i}_count" for i in range(1,N+1)] +
                                  ["total_publications","top_list"])
        return empty_counts, empty_wide

    rows = []
    for r in records:
        d = r.get("publication_date")
        if not d: 
            continue
        try:
            dt = pd.to_datetime(d).normalize()
        except Exception:
            continue
        topics = normalize_topics_list(r)
        rows.append((r.get("id"), dt, topics))

    works_df = pd.DataFrame(rows, columns=["id","date","topics_list"]).drop_duplicates(subset=["id"])

    exploded = (works_df.explode("topics_list")
                          .dropna(subset=["topics_list"])
                          .rename(columns={"topics_list":"topic"}))
    daily_topic_counts = (exploded.groupby(["date","topic"], as_index=False)
                                  .size().rename(columns={"size":"count"}))

    total_pub = works_df.groupby("date", as_index=False).size().rename(columns={"size":"total_publications"})

    topN = (daily_topic_counts.sort_values(["date","count","topic"], ascending=[True, False, True])
                            .groupby("date", as_index=False)
                            .head(N).copy())
    topN["rank"] = topN.groupby("date")["count"].rank(method="first", ascending=False).astype(int)
    topN["k_topic"] = "top" + topN["rank"].astype(str)
    topN["k_count"] = topN["k_topic"] + "_count"

    T = topN.pivot(index="date", columns="k_topic", values="topic").reset_index().rename_axis(None, axis=1)
    C = topN.pivot(index="date", columns="k_count", values="count").reset_index().rename_axis(None, axis=1)
    W = T.merge(C, on="date", how="outer").merge(total_pub, on="date", how="left")

    for i in range(1, N+1):
        if f"top{i}" not in W.columns: W[f"top{i}"] = pd.NA
        if f"top{i}_count" not in W.columns: W[f"top{i}_count"] = pd.NA

    W = W[["date"] + [f"top{i}" for i in range(1,N+1)] +
                 [f"top{i}_count" for i in range(1,N+1)] + ["total_publications"]]
    W["top_list"] = (W[[f"top{i}" for i in range(1,N+1)]].fillna("")
                     .apply(lambda r: "|".join([t for t in r if t]), axis=1))

    return daily_topic_counts.sort_values(["date","topic"]), W.sort_values("date")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    ap.add_argument("--start",  required=True)
    ap.add_argument("--end",    required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "sector_map.yaml"))
    ap.add_argument("--primary_only", action="store_true", default=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.sector not in cfg:
        raise SystemExit(f"sector must be one of: {list(cfg.keys())}")

    topic_name = cfg[args.sector]["primary_topic_name"]
    topic_id, _ = resolve_topic_id(topic_name)
    recs = fetch_works_with_topics(topic_id, args.start, args.end, primary_only=args.primary_only)

    daily_topic_counts, daily_top5_wide = build_topic_tables(recs, N=5)
    # robust CSVs (quote all to survive commas inside topic names)
    out_counts = TOPICS_DIR / f"daily_topic_counts_{args.sector}.csv"
    out_wide   = TOPICS_DIR / f"daily_top5_wide_{args.sector}.csv"
    daily_topic_counts.to_csv(out_counts, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    daily_top5_wide.to_csv(out_wide,     index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"[OpenAlex] wrote {out_counts} and {out_wide}")

if __name__ == "__main__":
    main()
