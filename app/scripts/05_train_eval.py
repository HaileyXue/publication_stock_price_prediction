#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, classification_report, accuracy_score
)
from xgboost import XGBClassifier

# ----------------- Paths -----------------
DATA_DIR     = Path(__file__).resolve().parents[2] / "data"
FEATURES_DIR = DATA_DIR / "processed" / "features"
REPORTS_DIR  = DATA_DIR / "reports"
PLOTS_DIR    = REPORTS_DIR / "plots"
MODELS_DIR   = REPORTS_DIR / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Helpers -----------------
def _tag(start: str, end: str) -> str:
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    return f"{s}-{e}"

def build_preprocessor(train_df, num_cols_all, cat_cols_all, use_categories: bool):
    num_cols = [c for c in num_cols_all if c in train_df.columns]
    transformers = [
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
    ]
    if use_categories:
        cat_cols = [c for c in cat_cols_all if c in train_df.columns]
        if cat_cols:
            transformers.append((
                "cat",
                Pipeline([
                    ("ohe", OneHotEncoder(handle_unknown="ignore",
                                          min_frequency=5, sparse_output=False)),
                ]),
                cat_cols
            ))
    return ColumnTransformer(transformers)

def _clean_feat_names(names: np.ndarray) -> list[str]:
    out = []
    for n in names:
        s = str(n)
        if "__" in s:
            s = s.split("__", 1)[1]
        out.append(s)
    return out

def export_feature_importance(model: Pipeline, model_name: str, sector: str,
                              suffix: str, variant: str, tag: str, topn: int = 25):
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    feat_names = _clean_feat_names(pre.get_feature_names_out())

    if isinstance(clf, LogisticRegression):
        coef = clf.coef_.ravel()
        imp_df = pd.DataFrame({
            "feature": feat_names,
            "coef": coef,
            "importance": np.abs(coef),
            "sign": np.sign(coef),
        }).sort_values("importance", ascending=False)
    elif hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feat_names,
            "importance": fi
        }).sort_values("importance", ascending=False)
    else:
        print(f"[WARN] {model_name}{suffix}{variant}_{tag}: no importances/coef.")
        return

    csv_path = MODELS_DIR / f"{sector}_{model_name}_feature_importance{suffix}{variant}_{tag}.csv"
    imp_df.to_csv(csv_path, index=False)

    top = imp_df.head(topn).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.4 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"{sector} — {model_name} Top-{topn} {suffix}{variant} [{tag}]")
    plt.tight_layout()
    png_path = PLOTS_DIR / f"{sector}_{model_name}_feature_importance{suffix}{variant}_{tag}.png"
    plt.savefig(png_path, bbox_inches="tight"); plt.close()

    print(f"[{model_name}{suffix}{variant}_{tag}] feature importance → {csv_path.name}, {png_path.name}")

def eval_and_plot(model_name, model, X_train, y_train, X_test, y_test,
                  sector, suffix, variant, tag, fast=False):
    """
    Fits the model, evaluates ROC-AUC / PR-AUC / Accuracy (threshold=0.5),
    saves ROC/PR plots, and returns metrics.
    """
    # Fit
    print(f"[{model_name}{suffix}{variant}_{tag}] Fitting…")
    model.fit(X_train, y_train)
    print(f"[{model_name}{suffix}{variant}_{tag}] Predicting…")
    p = model.predict_proba(X_test)[:, 1]

    preds   = (p > 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, p)
    pr_auc  = average_precision_score(y_test, p)
    acc     = accuracy_score(y_test, preds)
    rep     = classification_report(y_test, preds, digits=3, output_dict=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{model_name} (ROC-AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{sector} ROC — {model_name} {suffix}{variant} [{tag}]")
    plt.legend(loc="lower right")
    roc_path = PLOTS_DIR / f"{sector}_ROC_{model_name}{suffix}{variant}_{tag}.png"
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, p)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"{model_name} (AP={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{sector} PR — {model_name} {suffix}{variant} [{tag}]")
    plt.legend(loc="lower left")
    pr_path = PLOTS_DIR / f"{sector}_PR_{model_name}{suffix}{variant}_{tag}.png"
    plt.savefig(pr_path, bbox_inches="tight"); plt.close()

    print(f"[{model_name}{suffix}{variant}_{tag}] ROC-AUC={roc_auc:.3f} | AP={pr_auc:.3f} | ACC={acc:.3f}")
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "classification_report": rep,
        "roc_plot": str(roc_path),
        "pr_plot":  str(pr_path),
    }

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    ap.add_argument("--start",  required=True)
    ap.add_argument("--end",    required=True)
    ap.add_argument("--use-categories", action="store_true",
                    help="Include top1..top5 as one-hot categorical features")
    ap.add_argument("--include-pub", action="store_true",
                    help="If set, also train the with-pub variant. If not set, only nopub is trained.")
    ap.add_argument("--fast", action="store_true",
                    help="Faster training: fewer trees/shallow depth, early stopping not used here (pipeline).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Optional cap: use only the most recent N rows for train/test")
    args = ap.parse_args()

    sector = args.sector
    tag    = _tag(args.start, args.end)
    suffix_base = "_withcat" if args.use_categories else "_nocat"

    # ----------------- Load & prepare data -----------------
    fpath = FEATURES_DIR / f"features_{sector}_{tag}.csv"
    df = pd.read_csv(fpath, encoding="utf-8-sig")
    label_col = "y_up_5d"

    # Candidate columns
    # Candidate columns for modeling
    numeric_base = ["ret_5d", "vol_4w", "vol_growth"]
    numeric_pub  = ["pub_4w", "pub_growth", "top2_4w_share", "top5_4w_share"]  # gated by --include-pub

    # Categorical columns (if --use-categories is on)
    cat_cols_all = ["top2","top5","top2_4w","top5_4w"]

    d = df.dropna(subset=[label_col]).copy()
    if args.max_rows is not None and len(d) > args.max_rows:
        d = d.tail(args.max_rows).copy()
        print(f"[Info] Using last {len(d)} rows due to --max-rows={args.max_rows}")

    # Clean infs in numeric features
    for c in numeric_base + numeric_pub:
        if c in d.columns:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    print(f"[Info] Rows after filtering: {len(d)}")

    # Time-based 70/30 split
    split = int(0.7 * len(d))
    train, test = d.iloc[:split].copy(), d.iloc[split:].copy()
    y_train = train[label_col].astype(int).values
    y_test  = test[label_col].astype(int).values

    # Class imbalance handling for XGB
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(pos, 1))

    # Hyperparameters
    if args.fast:
        rf_params  = dict(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, class_weight="balanced_subsample",
            n_jobs=-1, oob_score=False
        )
        xgb_params = dict(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
            reg_lambda=1.0, reg_alpha=0.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist",
            random_state=42, n_jobs=-1,
            scale_pos_weight=spw
        )
    else:
        rf_params  = dict(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            random_state=42, class_weight="balanced_subsample",
            n_jobs=-1, oob_score=True
        )
        xgb_params = dict(
            n_estimators=800, learning_rate=0.03, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
            reg_lambda=1.0, reg_alpha=0.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist",
            random_state=42, n_jobs=-1,
            scale_pos_weight=spw
        )

    print(f"[Info] Fast mode: {args.fast}. RF trees={rf_params['n_estimators']}, XGB trees={xgb_params['n_estimators']}")

    # Build model constructors (pipelines will be created per-variant)
    def make_models(train_df, use_categories, num_cols_all, cat_cols_all):
        pre = build_preprocessor(train_df, num_cols_all, cat_cols_all, use_categories)
        logit = Pipeline([("pre", pre),
            ("clf", LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=2000))])
        rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(**rf_params))])
        xgb = Pipeline([("pre", pre), ("clf", XGBClassifier(**xgb_params))])
        return {"logit": logit, "rf": rf, "xgb": xgb}

    results = {}

    # -------- Variant A: baseline (no publication numerics) --------
    num_cols_all = numeric_base[:]  # no pubs
    models = make_models(train, args.use_categories, num_cols_all, cat_cols_all)
    suffix, variant = suffix_base, "_nopub"
    res_nopub = {}
    for name, mdl in models.items():
        res_nopub[name] = eval_and_plot(name, mdl, train, y_train, test, y_test,
                                        sector, suffix, variant, tag, fast=args.fast)
        export_feature_importance(mdl, name, sector, suffix, variant, tag, topn=25)
    results["nopub"] = res_nopub

    # -------- Variant B: add publication numerics (if requested) --------
    if args.include_pub:
        num_cols_all = numeric_base + numeric_pub
        models = make_models(train, args.use_categories, num_cols_all, cat_cols_all)
        variant = "_withpub"
        res_withpub = {}
        for name, mdl in models.items():
            res_withpub[name] = eval_and_plot(name, mdl, train, y_train, test, y_test,
                                              sector, suffix, variant, tag, fast=args.fast)
            export_feature_importance(mdl, name, sector, suffix, variant, tag, topn=25)
        results["withpub"] = res_withpub

    # Write combined metrics
    out_json = MODELS_DIR / f"{sector}_metrics{suffix}_{tag}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[Eval] wrote {out_json}")

if __name__ == "__main__":
    main()
