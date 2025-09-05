# app.py
import sys, subprocess, glob, json
from pathlib import Path
import datetime as dt
import pandas as pd
import streamlit as st

# =========================
# Paths & basic setup
# =========================
ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
RAW_PRICES    = RAW_DIR / "prices"
PROCESSED_DIR = DATA_DIR / "processed"
TOPICS_DIR    = PROCESSED_DIR / "topics"
FEATURES_DIR  = PROCESSED_DIR / "features"
REPORTS_DIR   = DATA_DIR / "reports"
PLOTS_DIR     = REPORTS_DIR / "plots"
MODELS_DIR    = REPORTS_DIR / "models"
CONFIG_DIR    = ROOT / "app" / "config"
CONFIG_PATH   = CONFIG_DIR / "sector_map.yaml"

for p in [RAW_PRICES, TOPICS_DIR, FEATURES_DIR, PLOTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Literature & Market Prediction", layout="wide")
st.title("Use Literature Metadata for Stock Movement Prediction")

# =========================
# Sector list (from YAML if present)
# =========================
def load_sectors_from_yaml():
    try:
        import yaml
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)
            return list(cfg.keys())
    except Exception:
        pass
    return [
        "Biotech","Semiconductors","Energy","Financials","Healthcare",
        "Technology","Industrials","Consumer Discretionary","Consumer Staples",
        "Utilities","Materials","Real Estate","Communication Services",
    ]
SECTORS = load_sectors_from_yaml()

# =========================
# Script command templates (now all pass --start/--end)
# =========================
SCRIPT_CMDS = {
    "s1_fetch_prices": [
        sys.executable, str(ROOT / "app" / "scripts" / "01_fetch_prices_stooq.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}",
        "--config", str(CONFIG_PATH),
    ],
    "s2_fetch_openalex": [
        sys.executable, str(ROOT / "app" / "scripts" / "02_fetch_openalex_topics.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}",
        "--config", str(CONFIG_PATH),
    ],
    "s3_build_features": [
        sys.executable, str(ROOT / "app" / "scripts" / "03_build_features.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}",
    ],
    "s4_visualize": [
        sys.executable, str(ROOT / "app" / "scripts" / "04_visualize.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}",
    ],
    "s5_train_eval_nocat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}",
    ],
    "s5_train_eval_withcat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}", "--use-categories",
    ],
    "s5_train_eval_fast_nocat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}", "--fast", "--max-rows", "{maxrows}",
    ],
    "s5_train_eval_fast_withcat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}", "--start", "{start}", "--end", "{end}", "--use-categories", "--fast", "--max-rows", "{maxrows}",
    ],
}

# =========================
# Helpers & constants
# =========================
NUM_FEATURES = ["ret_1d","close_mean","vol_4w","vol_growth","pub_4w","pub_growth"]
CAT_FEATURES = ["top1", "top5"]

def run_py(cmd_list, label=None, kind="generic", stream=False):
    default_labels = {
        "generic": "Working…",
        "plots":   "Generating plots…",
        "model":   "Training & evaluating models…",
    }
    msg = label or default_labels.get(kind, default_labels["generic"])

    with st.status(msg, expanded=stream) as status:
        try:
            if stream:
                subprocess.run(cmd_list, check=True)
            else:
                subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            status.update(label="✅ Done", state="complete")
        except subprocess.CalledProcessError as e:
            status.update(label="❌ Step failed", state="error")
            st.error("Step failed. See logs below.")
            if hasattr(e, "stdout") and e.stdout:
                st.code(e.stdout[-4000:], language="bash")
            if hasattr(e, "stderr") and e.stderr:
                st.code(e.stderr[-4000:], language="bash")
            st.stop()

def fmt_date(d): 
    return pd.to_datetime(d).strftime("%Y-%m-%d")
def date_tag(start, end):
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    return f"{s}-{e}"

def prices_csv_tagged(sector, start, end):
    return RAW_PRICES / f"sector_{sector}_{date_tag(start,end)}_daily_agg.csv"
def topics_counts_csv_tagged(sector, start, end):
    return TOPICS_DIR / f"daily_topic_counts_{sector}_{date_tag(start,end)}.csv"
def topics_top5_csv_tagged(sector, start, end):
    return TOPICS_DIR / f"daily_top5_wide_{sector}_{date_tag(start,end)}.csv"
def features_csv_tagged(sector, start, end):
    return FEATURES_DIR / f"features_{sector}_{date_tag(start,end)}.csv"

def metrics_json_path(sector: str, with_cat: bool, start=None, end=None) -> Path:
    if start is None or end is None:
        return MODELS_DIR / f"{sector}_metrics{'_withcat' if with_cat else '_nocat'}.json"
    return MODELS_DIR / f"{sector}_metrics{'_withcat' if with_cat else '_nocat'}_{date_tag(start,end)}.json"

# =========================
# Orchestration (date-aware)
# =========================
def ensure_prices(sector, start, end, force=False):
    out = prices_csv_tagged(sector, start, end)
    if out.exists() and not force:
        st.success(f"Prices present — {out.name}")
        return
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end))
           for arg in SCRIPT_CMDS["s1_fetch_prices"]]
    run_py(cmd, label="Extracting and writing…")

def ensure_topics(sector, start, end, force=False):
    counts_p = topics_counts_csv_tagged(sector, start, end)
    wide_p   = topics_top5_csv_tagged(sector, start, end)
    if counts_p.exists() and wide_p.exists() and not force:
        st.success(f"Topics present — {counts_p.name}, {wide_p.name}")
        return
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end))
           for arg in SCRIPT_CMDS["s2_fetch_openalex"]]
    run_py(cmd, label="Extracting and writing…")

def ensure_features(sector, start, end, force=False):
    fcsv = features_csv_tagged(sector, start, end)
    if fcsv.exists() and not force:
        st.success(f"Features present — {fcsv.name}")
        return
    ensure_prices(sector, start, end, force=force)
    ensure_topics(sector, start, end, force=force)
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end))
           for arg in SCRIPT_CMDS["s3_build_features"]]
    run_py(cmd, label="Extracting and writing…")

def ensure_plots(sector, start, end, force=False):
    tag = date_tag(start, end)
    sentinel = PLOTS_DIR / f"{sector}_{tag}_price.png"
    if sentinel.exists() and not force:
        st.success(f"Plots present for {tag} — skipping.")
        return
    ensure_features(sector, start, end, force=False)
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end))
           for arg in SCRIPT_CMDS["s4_visualize"]]
    run_py(cmd, kind="plots")

def run_modeling(sector, start, end, with_categories=False, fast=False, max_rows=None):
    key = ("s5_train_eval_fast_withcat" if with_categories else "s5_train_eval_fast_nocat") if fast \
          else ("s5_train_eval_withcat" if with_categories else "s5_train_eval_nocat")
    cmd_tpl = SCRIPT_CMDS[key]
    if fast:
        max_rows = int(max_rows) if max_rows is not None else 3000
        cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end), maxrows=str(max_rows))
               for arg in cmd_tpl]
    else:
        cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end))
               for arg in cmd_tpl]
    run_py(cmd, kind="model", stream=True)

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Controls")
sector = st.sidebar.selectbox("Sector", SECTORS, index=SECTORS.index("Semiconductors") if "Semiconductors" in SECTORS else 0)
today = dt.date.today()
default_start = today.replace(year=today.year - 1)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date   = st.sidebar.date_input("End date",   value=today)
force_all  = st.sidebar.checkbox("Force refresh all steps", value=False)
if "with_categories" not in st.session_state:
    st.session_state.with_categories = False
st.session_state.with_categories = st.sidebar.checkbox("Include categorical features (top1..top5) for modeling & feature view", value=st.session_state.with_categories)

st.sidebar.markdown("---")
st.sidebar.caption("Files are skipped unless Force refresh is enabled.")

# =========================
# Tabs
# =========================
tab_build, tab_features, tab_plots, tab_model = st.tabs(
    ["🛠️ Build & Run", "📑 Features", "📊 Plots", "🤖 Modeling"]
)

# ---------- TAB: Build & Run ----------
with tab_build:
    st.subheader("Pipeline Orchestration")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Build / Update Data"):
            ensure_features(sector, start_date, end_date, force=force_all)
            st.success("Data build complete.")

    with c2:
        if st.button("Generate Plots"):
            ensure_plots(sector, start_date, end_date, force=force_all)
            st.success("Plot generation complete.")

    with c3:
        st.markdown("**Modeling Options**")
        fast_mode = st.checkbox("Fast mode (smaller models + early stopping)", value=True, key="fast_mode")
        max_rows  = st.number_input("Max rows (most recent)", min_value=500, max_value=20000, value=3000, step=500, key="max_rows")
        if st.button("Run Modeling"):
            ensure_features(sector, start_date, end_date, force=False)
            run_modeling(
                sector, start_date, end_date,
                with_categories=st.session_state.with_categories,
                fast=fast_mode,
                max_rows=int(max_rows)
            )
            st.success("Modeling complete.")

    # -------- Artifacts Status (date-scoped only) --------
    st.markdown("---")
    st.subheader("Artifacts Status")
    tag = date_tag(start_date, end_date)
    cached_items = [
        ("Prices",  RAW_PRICES / f"sector_{sector}_{tag}_daily_agg.csv"),
        ("Topics: daily_topic_counts", TOPICS_DIR / f"daily_topic_counts_{sector}_{tag}.csv"),
        ("Topics: daily_top5_wide",    TOPICS_DIR / f"daily_top5_wide_{sector}_{tag}.csv"),
        ("Features", FEATURES_DIR / f"features_{sector}_{tag}.csv"),
    ]
    for label, path in cached_items:
        if path.exists():
            st.success(f"✓ {label}: {path.name}")
        else:
            st.info(f"• {label}: not found ({path.name})")

    st.markdown("**Plots (date-scoped)**")
    plot_imgs = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_{tag}_*.png")))
    if plot_imgs:
        st.success(f"✓ {len(plot_imgs)} plot(s) generated")
        with st.expander("Show plot files"):
            for p in plot_imgs:
                st.write(Path(p).name)
    else:
        st.info("• No plots for this date range yet")

# ---------- TAB: Features ----------
with tab_features:
    st.subheader("Feature Table (model inputs only)")
    fcsv = FEATURES_DIR / f"features_{sector}_{date_tag(start_date, end_date)}.csv"
    if fcsv.exists():
        df = pd.read_csv(fcsv, parse_dates=["date"]).sort_values("date")
        used_cols = NUM_FEATURES.copy()
        if st.session_state.with_categories:
            used_cols += CAT_FEATURES
        used_cols = [c for c in used_cols if c in df.columns]
        show_cols = ["date"] + used_cols

        st.caption(
            f"Showing **{len(used_cols)}** model input columns "
            f"({'with' if st.session_state.with_categories else 'without'} categorical features)."
        )
        st.dataframe(df[show_cols].head(200), use_container_width=True)
        st.caption("Note: for speed, only the **first 200 rows** (head) are shown here.")
        # --- Feature explanations (only for columns shown) ---
        expl = {
            # numeric
            "close_mean": "Equal-weighted average closing price across sector tickers.",
            "ret_1d": "1-day return of the sector average price (close_mean).",
            "vol_4w": "Rolling 20-trading-day (~4 weeks) sum of total traded volume.",
            "vol_growth": "Day-over-day percent change in total traded volume.",
            "pub_4w": "Rolling 20-trading-day sum of mapped publication counts.",
            "pub_growth": "Day-over-day percent change in mapped publication counts.",
            # categorical (topic names)
            "top1": "Most frequent topic name on that day.",
            "top2": "2nd most frequent topic name.",
            "top3": "3rd most frequent topic name.",
            "top4": "4th most frequent topic name.",
            "top5": "5th most frequent topic name.",
        }

        # Only describe columns actually displayed in the table
        described_cols = [c for c in show_cols if c in expl]
        nums = [c for c in described_cols if c in {"ret_1d","close_mean","vol_4w","vol_growth","pub_4w","pub_growth"}]
        cats = [c for c in described_cols if c in {"top1","top2","top3","top4","top5"}]

        with st.expander("What do these features mean?"):
            if nums:
                st.markdown("**Numeric features**")
                for c in nums:
                    st.markdown(f"- **{c}** — {expl[c]}")
            if cats:
                st.markdown("**Categorical features (topics)**")
                for c in cats:
                    st.markdown(f"- **{c}** — {expl[c]}")
    else:
        st.info("Features not found yet. Run **Build / Update Data** in the Build & Run tab.")

# ---------- TAB: Plots ----------
with tab_plots:
    st.subheader("Visualization Outputs")
    tag = date_tag(start_date, end_date)

    # Base plots in the exact order (2 per row)
    ordered_plots = [
        PLOTS_DIR / f"{sector}_{tag}_price.png",                    # 1st (left)
        PLOTS_DIR / f"{sector}_{tag}_levels_price_vs_pubs.png",     # 2nd (right)
        PLOTS_DIR / f"{sector}_{tag}_volume_feats.png",
        PLOTS_DIR / f"{sector}_{tag}_pub_feats.png",
        PLOTS_DIR / f"{sector}_{tag}_corr_all.png",
        PLOTS_DIR / f"{sector}_{tag}_corr_core.png",
    ]

    # Only show Cramér's V heatmap if categorical features are selected
    if st.session_state.with_categories:
        ordered_plots.append(PLOTS_DIR / f"{sector}_{tag}_cat_assoc_top_topics.png")

    existing = [p for p in ordered_plots if p.exists()]
    missing  = [p for p in ordered_plots if not p.exists()]

    if not existing:
        st.info("No plots for this date range. Use **Generate Plots (04)**.")
    else:
        cols = st.columns(2)
        for i, p in enumerate(existing):
            with cols[i % 2]:
                st.image(str(p), use_container_width=True)  # no caption

    if missing:
        with st.expander("Missing expected plots"):
            for p in missing:
                st.write(p.name)

# ---------- TAB: Modeling ----------
with tab_model:
    st.subheader("Modeling Results")
    variant_label = "With categorical features" if st.session_state.with_categories else "No categorical features"
    mj_path = MODELS_DIR / f"{sector}_metrics{'_withcat' if st.session_state.with_categories else '_nocat'}_{date_tag(start_date, end_date)}.json"

    if mj_path.exists():
        try:
            metrics = json.loads(mj_path.read_text())
            model_name_map = {"logit": "Logistic Regression","rf": "Random Forest","xgb": "XGBoost"}
            rows = []
            for model_key, vals in metrics.items():
                pretty = model_name_map.get(model_key, model_key)
                rows.append({"Model": pretty,
                             "ROC-AUC": round(vals.get("roc_auc", float("nan")), 4),
                             "PR-AUC":  round(vals.get("pr_auc",  float("nan")), 4)})
            metrics_df = pd.DataFrame(rows).sort_values("Model")
            st.caption(f"{variant_label} ({date_tag(start_date, end_date)})")
            st.dataframe(metrics_df, use_container_width=True)
        except Exception:
            st.error("Could not parse metrics JSON.")
    else:
        st.info(f"No metrics found ({variant_label}). Run **Modeling** in the Build & Run tab.")

    st.markdown("---")
    st.subheader("Feature Importance")
    suffix = "_withcat" if st.session_state.with_categories else "_nocat"
    tag = date_tag(start_date, end_date)
    fi_csvs = sorted(glob.glob(str(MODELS_DIR / f"{sector}_*feature_importance{suffix}_{tag}.csv")))
    fi_pngs = sorted(glob.glob(str(PLOTS_DIR  / f"{sector}_*feature_importance{suffix}_{tag}.png")))
    if fi_csvs:
        with st.expander("Tables (CSV)"):
            for p in fi_csvs:
                st.write(Path(p).name)
                try:
                    st.dataframe(pd.read_csv(p), use_container_width=True)
                except Exception:
                    st.code(Path(p).read_text()[:4000])
    else:
        st.info("No feature-importance CSVs yet for this variant/date range.")

    if fi_pngs:
        with st.expander("Bar charts (Top-N)"):
            cols = st.columns(2)
            for i, p in enumerate(fi_pngs):
                with cols[i % 2]:
                    st.image(str(p), caption=Path(p).name, use_container_width=True)
    else:
        st.info("No feature-importance plots yet for this variant/date range.")

    st.markdown("---")
    st.subheader("ROC & PR Curves")
    roc_imgs = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_ROC_*{suffix}_{tag}.png")))
    pr_imgs  = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_PR_*{suffix}_{tag}.png")))
    if roc_imgs or pr_imgs:
        cols = st.columns(3)
        for i, p in enumerate(roc_imgs + pr_imgs):
            with cols[i % 3]:
                st.image(str(p), caption=Path(p).name, use_container_width=True)
    else:
        st.info("No ROC/PR curves yet for this variant/date range.")
