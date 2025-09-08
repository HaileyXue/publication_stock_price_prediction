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
        "AI","Biotech","Semiconductors","Energy","Financials","Healthcare",
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
NUM_FEATURES = [
    "ret_5d", "vol_4w","vol_growth","pub_4w", "pub_growth",
    "top2_4w_share", "top5_4w_share",
]
CAT_FEATURES = [
    "top2","top5",
    "top2_4w","top5_4w",
]

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

def run_modeling(sector, start, end, with_categories=False, include_pub=False, fast=False, max_rows=None):
    # decide which command template to use
    key = ("s5_train_eval_fast_withcat" if with_categories else "s5_train_eval_fast_nocat") if fast \
          else ("s5_train_eval_withcat" if with_categories else "s5_train_eval_nocat")
    cmd_tpl = SCRIPT_CMDS[key]

    # base args
    if fast:
        max_rows = int(max_rows) if max_rows is not None else 3000
        cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end), maxrows=str(max_rows))
               for arg in cmd_tpl]
    else:
        cmd = [arg.format(sector=sector, start=fmt_date(start), end=fmt_date(end))
               for arg in cmd_tpl]

    # add --include-pub ONLY if the user selected it
    if include_pub:
        cmd.append("--include-pub")

    run_py(cmd, label="Training & evaluating models…", stream=True)

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

# --- Sidebar toggles (independent) ---
if "with_categories" not in st.session_state:
    st.session_state.with_categories = False
if "include_pub_numeric" not in st.session_state:
    st.session_state.include_pub_numeric = False

st.session_state.include_pub_numeric = st.sidebar.checkbox(
    "Include publication numeric features", value=st.session_state.include_pub_numeric
)
st.session_state.with_categories = st.sidebar.checkbox(
    "Include categorical topic features (top* / top*_4w)", value=st.session_state.with_categories
)

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

        # --- helpers to run/wait per variant ---
        import time, json

        def _vlabel(use_pubs: bool, use_cats: bool) -> str:
            return f"pubs={'on' if use_pubs else 'off'}, cats={'on' if use_cats else 'off'}"

        def _safe_load_json(path: Path, retries=12, sleep_s=0.25):
            for _ in range(retries):
                if path.exists():
                    try:
                        return json.loads(path.read_text())
                    except json.JSONDecodeError:
                        time.sleep(sleep_s)
                else:
                    time.sleep(sleep_s)
            return {}

        def _wait_for_metrics(use_cats: bool, use_pubs: bool, timeout_s: float = 120.0) -> bool:
            """
            Wait until the date-scoped metrics file exists AND contains the right variant key.
            Variant keys are 'nopub' or 'withpub' inside each *_metrics*.json.
            """
            p = metrics_json_path(sector, with_cat=use_cats, start=start_date, end=end_date)
            vkey = "withpub" if use_pubs else "nopub"
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                blob = _safe_load_json(p, retries=1, sleep_s=0.25)
                if blob and vkey in blob and blob[vkey]:
                    return True
                time.sleep(0.3)
            return False

        if st.button("Run Modeling"):
            ensure_features(sector, start_date, end_date, force=False)

            pubs = st.session_state.include_pub_numeric
            cats = st.session_state.with_categories

            # If BOTH toggles are on -> run all four variants sequentially
            if pubs and cats:
                variants = [(False, False), (True, False), (False, True), (True, True)]
                with st.status("Running all 4 model variants…", expanded=True) as status:
                    for use_pubs, use_cats in variants:
                        lbl = _vlabel(use_pubs, use_cats)
                        st.write(f"• Training: {lbl}")
                        run_modeling(
                            sector, start_date, end_date,
                            with_categories=use_cats,
                            include_pub=use_pubs,
                            fast=fast_mode,
                            max_rows=int(max_rows)
                        )
                        ok = _wait_for_metrics(use_cats, use_pubs, timeout_s=120.0)
                        if ok:
                            st.write(f"  ↳ ✅ Metrics saved for {lbl}")
                        else:
                            st.warning(f"  ↳ ⚠️ Metrics not detected for {lbl} (check trainer write).")
                    status.update(label="All variants finished (or attempted).", state="complete")
                st.success("Modeling complete.")
                st.rerun()

            # Otherwise, keep current behavior (run exactly what the toggles imply)
            else:
                run_modeling(
                    sector, start_date, end_date,
                    with_categories=cats,
                    include_pub=pubs,
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
    tag = date_tag(start_date, end_date)
    fcsv = FEATURES_DIR / f"features_{sector}_{tag}.csv"

    if fcsv.exists():
        df = pd.read_csv(fcsv, parse_dates=["date"]).sort_values("date")

        # --- Select columns based on toggles ---
        base_num = ["ret_5d", "vol_4w", "vol_growth"]
        pub_num  = ["pub_4w", "pub_growth", "top2_4w_share", "top5_4w_share"] if st.session_state.include_pub_numeric else []
        cats     = ["top2","top5","top2_4w","top5_4w"] if st.session_state.with_categories else []

        selected_cols = [c for c in (base_num + pub_num + cats) if c in df.columns]
        show_cols = ["date"] + selected_cols

        st.caption(
            f"Showing **{len(selected_cols)}** feature column(s) "
            f"({'with' if st.session_state.include_pub_numeric else 'without'} publication numerics; "
            f"{'with' if st.session_state.with_categories else 'without'} categorical topics)."
        )
        st.dataframe(df[show_cols].head(200), width="stretch")
        st.caption("Note: for speed, only the **first 200 rows** are shown.")

        # --- Feature explanations (only for columns shown) ---
        EXPL = {
            # numeric
            "ret_5d":        "5-day percentage return (past window).",
            "vol_4w":        "Rolling 4-week sum of trading volume.",
            "vol_growth":    "Day-over-day % change in trading volume.",
            "pub_4w":        "Rolling 4-week sum of publications mapped to trading days.",
            "pub_growth":    "Day-over-day % change in publication counts.",
            "top2_4w_share": "Share of the 4-week publications taken by the 2nd most frequent topic over that window.",
            "top5_4w_share": "Share of the 4-week publications taken by the 5th most frequent topic over that window.",
            # categorical
            "top2":   "Daily 2nd most frequent topic name.",
            "top5":   "Daily 5th most frequent topic name.",
            "top2_4w":"4-week #2 topic over the rolling window (topic name).",
            "top5_4w":"4-week #5 topic over the rolling window (topic name).",
        }

        NUM_SET = {
            "ret_5d","vol_4w","vol_growth","pub_4w","pub_growth","top2_4w_share","top5_4w_share"
        }
        CAT_SET = {"top2","top5","top2_4w","top5_4w"}

        described_cols = [c for c in selected_cols if c in EXPL]
        nums = [c for c in described_cols if c in NUM_SET]
        catz = [c for c in described_cols if c in CAT_SET]

        if nums or catz:
            with st.expander("What do these features mean?"):
                if nums:
                    st.markdown("**Numeric features**")
                    for c in nums:
                        st.markdown(f"- **{c}** — {EXPL[c]}")
                if catz:
                    st.markdown("**Categorical features (topics)**")
                    for c in catz:
                        st.markdown(f"- **{c}** — {EXPL[c]}")

    else:
        st.info("Features not found yet for this date range. Run **Build / Update Data** in the Build & Run tab.")

# ---------- TAB: Plots ----------
with tab_plots:
    st.subheader("Visualization Outputs")
    tag = date_tag(start_date, end_date)

    # Base plots in the exact order (2 per row)
    ordered_plots = [
        PLOTS_DIR / f"{sector}_{tag}_price.png",
        PLOTS_DIR / f"{sector}_{tag}_levels_price_vs_pubs.png",
        PLOTS_DIR / f"{sector}_{tag}_ret5d_vs_pub_growth.png",   # new
        PLOTS_DIR / f"{sector}_{tag}_ret5d_vs_pub4w.png",        # new
        PLOTS_DIR / f"{sector}_{tag}_class_balance.png",
        PLOTS_DIR / f"{sector}_{tag}_violin_pub4w_by_y.png",     # new
        PLOTS_DIR / f"{sector}_{tag}_granger_pubGrowth_vs_ret5d.png",  # new
        PLOTS_DIR / f"{sector}_{tag}_leadlag_heatmap.png",       # new
        PLOTS_DIR / f"{sector}_{tag}_corr_all.png",
        PLOTS_DIR / f"{sector}_{tag}_corr_core.png",
    ]

    # Only show Cramér's V heatmap if categorical features are selected
    if st.session_state.with_categories:
        ordered_plots.append(PLOTS_DIR / f"{sector}_{tag}_cat_assoc_top_topics.png")

    existing = [p for p in ordered_plots if p.exists()]
    missing  = [p for p in ordered_plots if not p.exists()]

    if not existing:
        st.info("No plots for this date range. Use **Generate Plots**.")
    else:
        cols = st.columns(2)
        for i, p in enumerate(existing):
            with cols[i % 2]:
                st.image(str(p), width="stretch")  # no caption

    if missing:
        with st.expander("Missing expected plots"):
            for p in missing:
                st.write(p.name)

    # --- Top 5 topics (only when categorical features are selected) ---
    if st.session_state.with_categories:
        fcsv_tagged = FEATURES_DIR / f"features_{sector}_{tag}.csv"
        if fcsv_tagged.exists():
            df_feat = pd.read_csv(fcsv_tagged)

            # Prefer count-weighted aggregation if *_count columns exist; otherwise fallback to frequency
            count_frames = []
            for i in range(1, 5 + 1):
                name_col  = f"top{i}"
                count_col = f"top{i}_count"
                if name_col in df_feat.columns and count_col in df_feat.columns:
                    tmp = df_feat[[name_col, count_col]].dropna()
                    if not tmp.empty:
                        tmp.columns = ["topic", "count"]
                        count_frames.append(tmp)

            if count_frames:
                tt = pd.concat(count_frames, ignore_index=True)
                top5 = (tt.groupby("topic", dropna=True)["count"]
                          .sum()
                          .sort_values(ascending=False)
                          .head(5)
                          .reset_index())
            else:
                # Fallback: topic frequency (when *_count not present)
                vals = []
                for i in range(1, 5 + 1):
                    c = f"top{i}"
                    if c in df_feat.columns:
                        vals.append(df_feat[c])
                if vals:
                    s = pd.concat(vals, ignore_index=True).dropna()
                    top5 = (s.value_counts()
                              .head(5)
                              .rename_axis("topic")
                              .reset_index(name="count"))
                else:
                    top5 = pd.DataFrame(columns=["topic", "count"])

            if not top5.empty:
                st.markdown("---")
                st.subheader("Top 5 topics in this date range")
                st.dataframe(top5, hide_index=True, width="stretch")
        else:
            st.info("Features file for this date range not found; generate features to see top topics.")


# ---------- TAB: Modeling ----------
with tab_model:
    st.subheader("Modeling Results")
    tag = date_tag(start_date, end_date)  # same tag used by trainer

    pubs = st.session_state.include_pub_numeric
    cats = st.session_state.with_categories

    # File paths (date-scoped)
    suffix_nocat   = "_nocat"
    suffix_withcat = "_withcat"
    path_nocat     = MODELS_DIR / f"{sector}_metrics{suffix_nocat}_{tag}.json"
    path_withcat   = MODELS_DIR / f"{sector}_metrics{suffix_withcat}_{tag}.json"

    # ---- Helpers: read-only (no training kicks off here) ----
    import time, json, glob, re
    from pathlib import Path

    def safe_load_metrics(path: Path, retries: int = 6, sleep_s: float = 0.2) -> dict:
        """Robust read that tolerates just-written JSON; NEVER triggers training."""
        if not path.exists():
            return {}
        for _ in range(retries):
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                time.sleep(sleep_s)
        return {}

    # Scenarios to display (purely for rendering)
    scenarios = [("_nocat", "nopub",   "Baseline — no publication numerics, no categoricals")]
    if pubs:
        scenarios.append(("_nocat",   "withpub", "With publication numerics (no categoricals)"))
    if cats:
        scenarios.append(("_withcat", "nopub",   "Categoricals only (no publication numerics)"))
    if pubs and cats:
        scenarios.append(("_withcat", "withpub", "With publication numerics + categoricals"))

    # Load available metrics (read-only)
    metrics_nocat   = safe_load_metrics(path_nocat)
    metrics_withcat = safe_load_metrics(path_withcat)

    def _get_metrics(suffix, variant):
        if suffix == "_nocat":
            return metrics_nocat.get(variant) if metrics_nocat else None
        else:
            return metrics_withcat.get(variant) if metrics_withcat else None

    def _as_table(blob, label_suffix):
        if not blob:
            st.info(f"No metrics found for **{label_suffix}** (tag {tag}). Use **Run Modeling** in the Build & Run tab.")
            return
        name_map = {"logit": "Logistic Regression","rf": "Random Forest","xgb": "XGBoost"}
        rows = []
        for mk, vals in blob.items():
            rows.append({
                "Model":    name_map.get(mk, mk),
                "ROC-AUC":  round(vals.get("roc_auc",  float("nan")), 4),
                "PR-AUC":   round(vals.get("pr_auc",   float("nan")), 4),
                "Accuracy": round(vals.get("accuracy", float("nan")), 4),
            })
        dfm = pd.DataFrame(rows).sort_values("Model")
        st.caption(label_suffix)
        st.dataframe(dfm, width="stretch")

    # ---------- Evaluation tables ----------
    st.markdown("**Evaluation tables**")
    for suf, var, label in scenarios:
        _as_table(_get_metrics(suf, var), label)

    st.markdown("---")
    st.subheader("Feature Importance")

    # ---------- Feature importance (CSV + bar plots), read-only ----------
    name_map = {"logit": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}

    def _model_name_from_path(path_str: str):
        stem = Path(path_str).stem
        m = re.search(rf"{re.escape(sector)}_(.+?)_feature_importance", stem)
        key = m.group(1) if m else "unknown"
        return (name_map.get(key, key), key)

    for suf, var, label in scenarios:
        st.markdown(f"**{label}** • {tag}")

        fi_csvs = sorted(glob.glob(str(MODELS_DIR / f"{sector}_*feature_importance{suf}_{var}_{tag}.csv")))
        fi_pngs = sorted(glob.glob(str(PLOTS_DIR  / f"{sector}_*feature_importance{suf}_{var}_{tag}.png")))

        if fi_csvs:
            with st.expander("Tables (CSV)"):
                combined = []
                for p in fi_csvs:
                    try:
                        df = pd.read_csv(p)
                        human_name, _ = _model_name_from_path(p)
                        df.insert(0, "Model", human_name)
                        combined.append(df)
                    except Exception:
                        st.code(Path(p).read_text()[:4000])
                if combined:
                    df_all = pd.concat(combined, ignore_index=True)
                    st.dataframe(df_all, width="stretch")
        else:
            st.info("No feature-importance CSVs for this scenario/date range yet.")

        if fi_pngs:
            with st.expander("Bar charts (Top-N)"):
                cols = st.columns(min(2, len(fi_pngs)))
                for i, p in enumerate(fi_pngs):
                    human_name, _ = _model_name_from_path(p)
                    with cols[i % len(cols)]:
                        st.image(str(p), width="stretch")
                        st.caption(human_name)
        else:
            st.info("No feature-importance plots for this scenario/date range yet.")

        st.markdown("")  # spacing

    # ---------- ROC curves ----------
    st.markdown("---")
    st.subheader("ROC curves")

    ordered_for_roc = [
        ("_nocat",   "nopub"),
        ("_nocat",   "withpub") if pubs else None,
        ("_withcat", "nopub")   if cats else None,
        ("_withcat", "withpub") if pubs and cats else None,
    ]
    ordered_for_roc = [x for x in ordered_for_roc if x is not None]

    model_order = [("logit", "Logistic Regression"),
                   ("rf",    "Random Forest"),
                   ("xgb",   "XGBoost")]

    def _roc_path(model_key, suffix, variant):
        return PLOTS_DIR / f"{sector}_ROC_{model_key}{suffix}_{variant}_{tag}.png"

    for mk, title in model_order:
        imgs = [p for (suf, var) in ordered_for_roc
                for p in [_roc_path(mk, suf, var)]
                if p.exists()]
        if not imgs:
            continue

        st.markdown(f"**{title}**")
        if len(imgs) == 1:
            left, mid, right = st.columns([1, 1, 1])
            with mid:
                st.image(str(imgs[0]), width="stretch")
        elif len(imgs) == 2:
            c1, c2 = st.columns(2)
            with c1: st.image(str(imgs[0]), width="stretch")
            with c2: st.image(str(imgs[1]), width="stretch")
        else:
            cols = st.columns(min(3, len(imgs)))
            for i, pth in enumerate(imgs):
                with cols[i % len(cols)]:
                    st.image(str(pth), width="stretch")
