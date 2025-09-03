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

st.set_page_config(page_title="📚→📈 Literature & Market App", layout="wide")
st.title("📚→📈 Literature & Market App")

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
    # Fallback list if YAML missing
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
    # 01 — prices via Stooq (requires YAML sector map)
    "s1_fetch_prices": [
        sys.executable, str(ROOT / "app" / "scripts" / "01_fetch_prices_stooq.py"),
        "--sector", "{sector}",
        "--start",  "{start}",
        "--end",    "{end}",
        "--config", str(CONFIG_PATH),
    ],
    # 02 — OpenAlex topics (requires YAML sector map)
    "s2_fetch_openalex": [
        sys.executable, str(ROOT / "app" / "scripts" / "02_fetch_openalex_topics.py"),
        "--sector", "{sector}",
        "--start",  "{start}",
        "--end",    "{end}",
        "--config", str(CONFIG_PATH),
        # uses --primary_only default True; no flag needed
    ],
    # 03 — build features
    "s3_build_features": [
        sys.executable, str(ROOT / "app" / "scripts" / "03_build_features.py"),
        "--sector", "{sector}",
    ],
    # 04 — visualize
    "s4_visualize": [
        sys.executable, str(ROOT / "app" / "scripts" / "04_visualize.py"),
        "--sector", "{sector}",
    ],
    # 05 — train/eval (toggle categories)
    "s5_train_eval_nocat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}",
    ],
    "s5_train_eval_withcat": [
        sys.executable, str(ROOT / "app" / "scripts" / "05_train_eval.py"),
        "--sector", "{sector}",
        "--use-categories",
    ],
}

# =========================
# Helpers
# =========================
def run_py(cmd_list):
    """Run a Python command and surface logs in Streamlit."""
    with st.status(f"Running: {' '.join(map(str, cmd_list))}", expanded=False) as status:
        try:
            proc = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            if proc.stdout:
                st.code(proc.stdout, language="bash")
            if proc.stderr:
                st.code(proc.stderr, language="bash")
            status.update(label="✅ Done", state="complete")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Command failed (exit {e.returncode})")
            if e.stdout: st.code(e.stdout, language="bash")
            if e.stderr: st.code(e.stderr, language="bash")
            st.stop()

def fmt_date(d):
    return pd.to_datetime(d).strftime("%Y-%m-%d")

# Expected artifact paths
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
    ]

def metrics_json_path(sector: str, with_cat: bool) -> Path:
    suffix = "_withcat" if with_cat else "_nocat"
    return MODELS_DIR / f"{sector}_metrics{suffix}.json"

# =========================
# Orchestration steps
# =========================
def ensure_prices(sector, start, end, force=False):
    out = sector_prices_csv(sector)
    if out.exists() and not force:
        st.success(f"Prices present: {out.name} — skipping 01.")
        return
    if not CONFIG_PATH.exists():
        st.error(f"Missing config: {CONFIG_PATH}. Add your sector_map.yaml.")
        st.stop()
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end)) for arg in SCRIPT_CMDS["s1_fetch_prices"]]
    run_py(cmd)

def ensure_topics(sector, start, end, force=False):
    counts_p = topics_counts_csv(sector)
    wide_p   = topics_top5_csv(sector)
    if counts_p.exists() and wide_p.exists() and not force:
        st.success(f"Topics present: {counts_p.name}, {wide_p.name} — skipping 02.")
        return
    if not CONFIG_PATH.exists():
        st.error(f"Missing config: {CONFIG_PATH}. Add your sector_map.yaml.")
        st.stop()
    cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end)) for arg in SCRIPT_CMDS["s2_fetch_openalex"]]
    run_py(cmd)

def ensure_features(sector, start, end, force=False):
    """
    If features exist and not forced → skip.
    Else: ensure prices + topics, then run 03 to build features.
    """
    fcsv = features_csv(sector)
    if fcsv.exists() and not force:
        st.success(f"Features present: {fcsv.name} — skipping 03.")
        return
    # Need inputs first
    ensure_prices(sector, start, end, force=False)  # don't force sub-steps unless top-level force is on
    ensure_topics(sector, start, end, force=False)
    # Build features
    cmd = [arg.format(sector=sector) for arg in SCRIPT_CMDS["s3_build_features"]]
    run_py(cmd)

def ensure_plots(sector, force=False):
    exp = expected_plot_paths(sector)
    missing = [p for p in exp if not p.exists()]
    if not missing and not force:
        st.success("All plots present — skipping 04.")
        return
    cmd = [arg.format(sector=sector) for arg in SCRIPT_CMDS["s4_visualize"]]
    run_py(cmd)

def run_modeling(sector, with_categories=False):
    key = "s5_train_eval_withcat" if with_categories else "s5_train_eval_nocat"
    cmd = [arg.format(sector=sector) for arg in SCRIPT_CMDS[key]]
    run_py(cmd)

# =========================
# UI Controls
# =========================
# =========================
# Sidebar Controls (moved from top)
# =========================
st.sidebar.header("Controls")
sector = st.sidebar.selectbox("Sector", SECTORS, index=SECTORS.index("Semiconductors") if "Semiconductors" in SECTORS else 0)

today = dt.date.today()
default_start = today.replace(year=today.year - 1)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date   = st.sidebar.date_input("End date",   value=today)

force_all  = st.sidebar.checkbox("Force refresh all steps", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("Files are skipped if already present unless force refresh is enabled.")

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
        if st.button("Build / Update Data (01–03)"):
            ensure_features(sector, start_date, end_date, force=force_all)
            st.success("Data build complete.")
    with c2:
        if st.button("Generate Plots (04)"):
            ensure_plots(sector, force=force_all)
            st.success("Plot generation complete.")
    with c3:
        with_categories = st.checkbox("Include categorical features (top1..top5) in 05", value=False)
        if st.button("Run Modeling (05)"):
            # Ensure features exist first
            ensure_features(sector, start_date, end_date, force=False)
            run_modeling(sector, with_categories=with_categories)
            st.success("Modeling complete.")

    st.markdown("---")
    st.subheader("Artifacts Status")
    # Prices / Topics / Features presence
    statuses = [
        ("Prices", sector_prices_csv(sector)),
        ("Topics: daily_topic_counts", topics_counts_csv(sector)),
        ("Topics: daily_top5_wide", topics_top5_csv(sector)),
        ("Features", features_csv(sector)),
    ]
    for label, path in statuses:
        if path.exists():
            st.success(f"✓ {label}: {path.name}")
        else:
            st.info(f"• {label}: not found")

# ---------- TAB: Features ----------
with tab_features:
    st.subheader("Feature Table")
    fcsv = features_csv(sector)
    if fcsv.exists():
        df = pd.read_csv(fcsv, parse_dates=["date"]).sort_values("date")
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        dfw = df.loc[mask].copy()
        st.caption(f"Loaded {len(dfw):,} rows from {fcsv.name} for {start_date} → {end_date}")
        st.dataframe(dfw.head(200))
    else:
        st.info("Features not found yet. Run **Build / Update Data (01–03)** in the Build & Run tab.")

# ---------- TAB: Plots ----------
with tab_plots:
    st.subheader("Visualization Outputs (from 04_visualize.py)")
    pngs = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_*.png")))
    if pngs:
        cols = st.columns(3)
        for i, p in enumerate(pngs):
            with cols[i % 3]:
                st.image(p, caption=Path(p).name, use_column_width=True)
    else:
        st.info("No plots found. Use **Generate Plots (04)** in the Build & Run tab.")

# ---------- TAB: Modeling ----------
with tab_model:
    st.subheader("Modeling Outputs (from 05_train_eval.py)")

    # Metrics JSON for both variants if present
    for variant, with_cat in [("No categories", False), ("With categories", True)]:
        mj = metrics_json_path(sector, with_cat)
        if mj.exists():
            st.success(f"Metrics found ({variant}): {mj.name}")
            try:
                data = json.loads(mj.read_text())
                st.json(data)
            except Exception:
                st.code(mj.read_text()[:8000])
        else:
            st.info(f"Metrics not found ({variant}). Run **Modeling (05)** in the Build & Run tab.")

    # Feature-importance CSVs & plots
    st.markdown("---")
    st.subheader("Feature Importance")
    fi_csvs = sorted(glob.glob(str(MODELS_DIR / f"{sector}_*_feature_importance*.csv")))
    fi_pngs = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_*_feature_importance*.png")))
    if fi_csvs or fi_pngs:
        if fi_csvs:
            with st.expander("Tables (CSV)"):
                for p in fi_csvs:
                    st.write(Path(p).name)
                    try:
                        st.dataframe(pd.read_csv(p))
                    except Exception:
                        st.code(Path(p).read_text()[:4000])
        if fi_pngs:
            with st.expander("Bar charts (Top-N)"):
                cols = st.columns(3)
                for i, p in enumerate(fi_pngs):
                    with cols[i % 3]:
                        st.image(p, caption=Path(p).name, use_column_width=True)
    else:
        st.info("No feature-importance outputs yet.")

    # ROC/PR curves
    st.markdown("---")
    st.subheader("ROC & PR Curves")
    roc_imgs = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_ROC_*.png")))
    pr_imgs  = sorted(glob.glob(str(PLOTS_DIR / f"{sector}_PR_*.png")))
    if roc_imgs or pr_imgs:
        cols = st.columns(3)
        for i, p in enumerate(roc_imgs + pr_imgs):
            with cols[i % 3]:
                st.image(p, caption=Path(p).name, use_column_width=True)
    else:
        st.info("No ROC/PR curves yet.")
