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
# Script command templates
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
        "--sector", "{sector}",
    ],
    "s4_visualize": [
        sys.executable, str(ROOT / "app" / "scripts" / "04_visualize.py"),
        "--sector", "{sector}",
    ],
    "s5_train_eval_nocat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}",
    ],
    "s5_train_eval_withcat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}", "--use-categories",
    ],
    "s5_train_eval_fast_nocat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}", "--fast", "--max-rows", "{maxrows}",
    ],
    "s5_train_eval_fast_withcat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}", "--use-categories", "--fast", "--max-rows", "{maxrows}",
    ],
}

# =========================
# Helpers & constants
# =========================
NUM_FEATURES = [
    "ret_1d","close_mean",
    "vol_4w","vol_growth",
    "pub_4w","pub_growth",
]
CAT_FEATURES = ["top1", "top5"]

def run_py(cmd_list, label="Extracting and writing…", stream=False):
    with st.status(label, expanded=stream) as status:
        try:
            if stream:
                # show live logs in the app (inherits stdout/stderr)
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

def sector_prices_csv(sector: str) -> Path:
    return RAW_PRICES / f"sector_{sector}_daily_agg.csv"

def topics_counts_csv(sector: str) -> Path:
    return TOPICS_DIR / f"daily_topic_counts_{sector}.csv"

def topics_top5_csv(sector: str) -> Path:
    return TOPICS_DIR / f"daily_top5_wide_{sector}.csv"

def features_csv(sector: str) -> Path:
    return FEATURES_DIR / f"features_{sector}.csv"

def expected_plot_paths(sector: str):
    return [
        PLOTS_DIR / f"{sector}_price.png",
        PLOTS_DIR / f"{sector}_volume.png",
        PLOTS_DIR / f"{sector}_pubs.png",
        PLOTS_DIR / f"{sector}_zscores.png",
        PLOTS_DIR / f"{sector}_levels_price_vs_pubs_z.png",
        PLOTS_DIR / f"{sector}_growth_ret_vs_pub_growth.png",
        PLOTS_DIR / f"{sector}_corr_all.png",
        PLOTS_DIR / f"{sector}_corr_core.png",
        PLOTS_DIR / f"{sector}_cat_assoc_top_topics.png",   # NEW
    ]

def metrics_json_path(sector: str, with_cat: bool) -> Path:
    return MODELS_DIR / f"{sector}_metrics{'_withcat' if with_cat else '_nocat'}.json"

def render_artifacts_status(sector: str):
    st.markdown("---")
    st.subheader("Artifacts Status")

    core = [
        ("Prices (sector daily aggregate)", sector_prices_csv(sector)),
        ("Topics: daily_topic_counts",      topics_counts_csv(sector)),
        ("Topics: daily_top5_wide",         topics_top5_csv(sector)),
        ("Features (model input table)",    features_csv(sector)),
    ]
    for label, path in core:
        if path.exists():
            st.success(f"✓ {label}: {path.name}")
        else:
            st.info(f"• {label}: not found")

    # Plots summary (only counts & filenames, not images)
    present_plots = [p for p in expected_plot_paths(sector) if p.exists()]
    missing_plots = [p for p in expected_plot_paths(sector) if not p.exists()]

    st.markdown("**Plots**")
    if present_plots:
        st.success(f"✓ {len(present_plots)} plot(s) generated")
        with st.expander("Show plot files"):
            for p in present_plots:
                st.write(p.name)
    else:
        st.info("• No plots found yet")

    if missing_plots:
        with st.expander("Missing (expected) plot files"):
            for p in missing_plots:
                st.write(p.name)

    # Metrics presence (both variants)
    for label, mpath in [
        ("Metrics (no categorical features)",  metrics_json_path(sector, False)),
        ("Metrics (with categorical features)", metrics_json_path(sector, True)),
    ]:
        if mpath.exists():
            st.success(f"✓ {label}: {mpath.name}")
        else:
            st.info(f"• {label}: not found")

# =========================
# Orchestration
# =========================
def ensure_prices(sector, start, end, force=False):
    out = sector_prices_csv(sector)
    if out.exists() and not force:
        st.success("Prices present — skipping.")
        return
    if not CONFIG_PATH.exists():
        st.error("Missing sector_map.yaml.")
        st.stop()
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end)) for arg in SCRIPT_CMDS["s1_fetch_prices"]]
    run_py(cmd, label="Extracting and writing…")

def ensure_topics(sector, start, end, force=False):
    counts_p = topics_counts_csv(sector)
    wide_p   = topics_top5_csv(sector)
    if counts_p.exists() and wide_p.exists() and not force:
        st.success("Topics present — skipping.")
        return
    if not CONFIG_PATH.exists():
        st.error("Missing sector_map.yaml.")
        st.stop()
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end)) for arg in SCRIPT_CMDS["s2_fetch_openalex"]]
    run_py(cmd, label="Extracting and writing…")

def ensure_features(sector, start, end, force=False):
    fcsv = features_csv(sector)
    if fcsv.exists() and not force:
        st.success("Features present — skipping.")
        return
    ensure_prices(sector, start, end, force=False)
    ensure_topics(sector, start, end, force=False)
    cmd = [arg.format(sector=sector) for arg in SCRIPT_CMDS["s3_build_features"]]
    run_py(cmd, label="Extracting and writing…")

def ensure_plots(sector, force=False):
    exp = expected_plot_paths(sector)
    missing = [p for p in exp if not p.exists()]
    if not missing and not force:
        st.success("All plots present — skipping.")
        return
    cmd = [arg.format(sector=sector) for arg in SCRIPT_CMDS["s4_visualize"]]
    run_py(cmd, label="Extracting and writing…")

def run_modeling(sector, with_categories=False, fast=False, max_rows=None):
    if fast:
        key = "s5_train_eval_fast_withcat" if with_categories else "s5_train_eval_fast_nocat"
        # default a sensible cap if none provided
        max_rows = int(max_rows) if max_rows is not None else 3000
        cmd_tpl = SCRIPT_CMDS[key]
        cmd = [arg.format(sector=sector, maxrows=str(max_rows)) for arg in cmd_tpl]
    else:
        key = "s5_train_eval_withcat" if with_categories else "s5_train_eval_nocat"
        cmd_tpl = SCRIPT_CMDS[key]
        cmd = [arg.format(sector=sector) for arg in cmd_tpl]

    run_py(cmd, label="Extracting and writing…", stream=True)

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
# persist modeling choice in session_state so other tabs can use it
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
            ensure_plots(sector, force=force_all)
            st.success("Plot generation complete.")

    with c3:
        # --- extra modeling options ---
        st.markdown("**Modeling Options**")
        fast_mode = st.checkbox("Fast mode (smaller models + early stopping)", value=True, key="fast_mode")
        max_rows  = st.number_input(
            "Max rows (most recent)", min_value=500, max_value=20000,
            value=3000, step=500, key="max_rows"
        )
        if st.button("Run Modeling"):
            ensure_features(sector, start_date, end_date, force=False)
            run_modeling(
                sector,
                with_categories=st.session_state.with_categories,
                fast=fast_mode,
                max_rows=int(max_rows)
            )
            st.success("Modeling complete.")

    # Only show this once in Build tab
    render_artifacts_status(sector)

# ---------- TAB: Features ----------
with tab_features:
    st.subheader("Feature Table (model inputs only)")
    fcsv = features_csv(sector)
    if fcsv.exists():
        df = pd.read_csv(fcsv, parse_dates=["date"]).sort_values("date")
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        dfw = df.loc[mask].copy()

        used_cols = NUM_FEATURES.copy()
        if st.session_state.with_categories:
            used_cols += CAT_FEATURES
        used_cols = [c for c in used_cols if c in dfw.columns]
        show_cols = ["date"] + used_cols

        st.caption(
            f"Showing **{len(used_cols)}** model input columns "
            f"({'with' if st.session_state.with_categories else 'without'} categorical features)."
        )
        st.dataframe(dfw[show_cols].head(200), use_container_width=True)
        st.caption("Note: for speed, only the **first 200 rows** (head) are shown here.")

        # --- Human-friendly feature descriptions ---
        expl = {
            "ret_1d": "Daily return of the sector basket (percentage change from previous trading day).",
            "close_mean": "Equal-weighted average closing price across sector ETFs on each trading day.",
            "vol_4w": "Rolling **4-week sum** of trading volume across the sector basket (smooths daily noise).",
            "vol_growth": "Day-over-day **percentage change** in total trading volume.",
            "pub_4w": "Rolling **4-week sum** of publication counts mapped to trading days.",
            "pub_growth": "Day-over-day **percentage change** in publications (0 if previous day is 0).",
            "top1": "Most frequent **topic name** on that day (from OpenAlex).",
            "top2": "2nd most frequent topic name.",
            "top3": "3rd most frequent topic name.",
            "top4": "4th most frequent topic name.",
            "top5": "5th most frequent topic name."
        }
        with st.expander("What do these features mean?"):
            nums = [c for c in NUM_FEATURES if c in used_cols]
            cats = [c for c in CAT_FEATURES if c in used_cols]
            if nums:
                st.markdown("**Numeric features**")
                for c in nums:
                    st.markdown(f"- **{c}** — {expl.get(c, 'n/a')}")
            if cats:
                st.markdown("**Categorical features (optional)**")
                for c in cats:
                    st.markdown(f"- **{c}** — {expl.get(c, 'n/a')}")
            st.caption("All publication features are aligned to trading days with a small forward tolerance.")
    else:
        st.info("Features not found yet. Run **Build / Update Data (01–03)** in the Build & Run tab.")

# ---------- TAB: Plots ----------
with tab_plots:
    st.subheader("Visualization Outputs")
    exp_paths = expected_plot_paths(sector)
    existing = [p for p in exp_paths if p.exists()]
    if existing:
        cols = st.columns(2)  # two per row for larger images
        for i, p in enumerate(existing):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_container_width=True)
    else:
        st.info("No plots found. Use **Generate Plots (04)** in the Build & Run tab.")

# ---------- TAB: Modeling ----------
with tab_model:
    st.subheader("Modeling Results")

    # Choose which metrics file to show (No categories vs With categories)
    variant_label = "With categorical features" if st.session_state.with_categories else "No categorical features"
    mj_path = metrics_json_path(sector, st.session_state.with_categories)

    if mj_path.exists():
        # Build a tidy table: rows = models, cols = ROC-AUC, PR-AUC
        try:
            metrics = json.loads(mj_path.read_text())
            # Map short keys -> full names
            model_name_map = {
                "logit": "Logistic Regression",
                "rf":    "Random Forest",
                "xgb":   "XGBoost",
            }
            rows = []
            for model_key, vals in metrics.items():
                pretty_name = model_name_map.get(model_key, model_key)  # fallback: original key
                rows.append({
                    "Model":  pretty_name,
                    "ROC-AUC": round(vals.get("roc_auc", float("nan")), 4),
                    "PR-AUC":  round(vals.get("pr_auc",  float("nan")), 4),
                })
            metrics_df = pd.DataFrame(rows).sort_values("Model")
            st.caption(f"{variant_label}")
            st.dataframe(metrics_df, use_container_width=True)
        except Exception:
            st.error("Could not parse metrics JSON.")
    else:
        st.info(f"No metrics found ({variant_label}). Run **Modeling (05)** in the Build & Run tab.")

        st.markdown("---")
    st.subheader("Feature Importance")

    # Tables (CSV)
    fi_glob_suffix = "_withcat" if st.session_state.with_categories else "_nocat"
    fi_csvs = sorted(glob.glob(str(MODELS_DIR / f"{sector}_*_feature_importance{fi_glob_suffix}.csv")))
    fi_pngs = sorted(glob.glob(str(PLOTS_DIR  / f"{sector}_*_feature_importance{fi_glob_suffix}.png")))

    if fi_csvs:
        with st.expander("Tables (CSV)"):
            for p in fi_csvs:
                st.write(Path(p).name)
                try:
                    st.dataframe(pd.read_csv(p), use_container_width=True)
                except Exception:
                    st.code(Path(p).read_text()[:4000])
    else:
        st.info("No feature-importance CSVs yet for this variant.")

    # Bar charts (Top-N) — under the CSVs
    if fi_pngs:
        with st.expander("Bar charts (Top-N)"):
            cols = st.columns(2)  # bigger plots
            for i, p in enumerate(fi_pngs):
                with cols[i % 2]:
                    st.image(str(p), caption=Path(p).name, use_container_width=True)
    else:
        st.info("No feature-importance plots yet for this variant.")

    # ROC/PR curves (keep these in Modeling tab)
    st.markdown("---")
    st.subheader("ROC & PR Curves")
    roc_imgs = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_ROC_*{'_withcat' if st.session_state.with_categories else '_nocat'}.png")))
    pr_imgs  = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_PR_*{'_withcat' if st.session_state.with_categories else '_nocat'}.png")))
    if roc_imgs or pr_imgs:
        cols = st.columns(3)
        for i, p in enumerate(roc_imgs + pr_imgs):
            with cols[i % 3]:
                st.image(str(p), caption=Path(p).name, use_container_width=True)
    else:
        st.info("No ROC/PR curves yet for this variant.")
