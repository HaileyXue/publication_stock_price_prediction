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
    roc_curve, classification_report
)
from xgboost import XGBClassifier

DATA_DIR     = Path(__file__).resolve().parents[2] / "data"
FEATURES_DIR = DATA_DIR / "processed" / "features"
REPORTS_DIR  = DATA_DIR / "reports"
PLOTS_DIR    = REPORTS_DIR / "plots"
MODELS_DIR   = REPORTS_DIR / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Preprocessor -----------------
def build_preprocessor(train_df, use_categories: bool, num_cols_all, cat_cols_all):
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
                    # Dense output → works for XGB, RF, Logit
                    ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
                ]),
                cat_cols
            ))
    return ColumnTransformer(transformers)

# ----------------- Eval & Plot -----------------
def eval_and_plot(model_name, model, X_train, y_train, X_test, y_test, sector, suffix):
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    preds = (p > 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, p)
    pr_auc  = average_precision_score(y_test, p)
    rep     = classification_report(y_test, preds, digits=3, output_dict=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{model_name} (ROC-AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{sector} ROC — {model_name} {suffix}")
    plt.legend(loc="lower right")
    roc_path = PLOTS_DIR / f"{sector}_ROC_{model_name}{suffix}.png"
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, p)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"{model_name} (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{sector} PR — {model_name} {suffix}")
    plt.legend(loc="lower left")
    pr_path = PLOTS_DIR / f"{sector}_PR_{model_name}{suffix}.png"
    plt.savefig(pr_path, bbox_inches="tight"); plt.close()

    print(f"[{model_name}{suffix}] ROC-AUC={roc_auc:.3f} | PR-AUC={pr_auc:.3f}")
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": rep,
        "roc_plot": str(roc_path),
        "pr_plot":  str(pr_path),
    }

# ----------------- Feature Importance -----------------
def _clean_feat_names(names: np.ndarray) -> list[str]:
    # ColumnTransformer prefixes like "num__" or "cat__"; flatten to cleaner names.
    out = []
    for n in names:
        s = str(n)
        if "__" in s:
            s = s.split("__", 1)[1]
        out.append(s)
    return out

def export_feature_importance(model: Pipeline, model_name: str, sector: str, suffix: str, topn: int = 25):
    """
    Saves:
      - CSV of all features + importance (and sign for Logit)
      - PNG bar chart of Top-N features
    """
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    feat_names = pre.get_feature_names_out()
    feat_names = _clean_feat_names(feat_names)

    imp_df = None
    if isinstance(clf, LogisticRegression):
        # coef_: shape (1, n_features) for binary
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
        print(f"[WARN] {model_name}{suffix}: no feature_importances_ / coef_ available.")
        return

    # Save CSV
    csv_path = MODELS_DIR / f"{sector}_{model_name}_feature_importance{suffix}.csv"
    imp_df.to_csv(csv_path, index=False)

    # Plot Top-N
    top = imp_df.head(topn).iloc[::-1]  # plot from bottom up
    plt.figure(figsize=(9, max(4, 0.4 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"{sector} — {model_name} Top-{topn} Features {suffix}")
    plt.tight_layout()
    png_path = PLOTS_DIR / f"{sector}_{model_name}_feature_importance{suffix}.png"
    plt.savefig(png_path, bbox_inches="tight"); plt.close()

    print(f"[{model_name}{suffix}] feature importance → CSV: {csv_path.name}, Plot: {png_path.name}")

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    ap.add_argument("--use-categories", action="store_true",
                    help="Include top1..top5 as one-hot categorical features")
    args = ap.parse_args()

    sector = args.sector
    suffix = "_withcat" if args.use_categories else "_nocat"

    df = pd.read_csv(FEATURES_DIR / f"features_{sector}.csv", encoding="utf-8-sig")

    # Label + features (avoid leakage; exclude ret_fwd_5d)
    label_col = "y_up_5d"
    num_cols_all = [
        "ret_1d","close_mean",
        "vol_4w","vol_growth",
        "pub_4w","pub_growth",
    ]
    cat_cols_all = ["top1", "top5"]

    d = df.dropna(subset=[label_col]).copy()
    for c in num_cols_all:
        if c in d.columns:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    # time-based 70/30 split
    split = int(0.7 * len(d))
    train, test = d.iloc[:split].copy(), d.iloc[split:].copy()
    y_train = train[label_col].astype(int).values
    y_test  = test[label_col].astype(int).values

    # class imbalance handling for XGB
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = float(neg / max(pos, 1))

    pre = build_preprocessor(train, use_categories=args.use_categories,
                             num_cols_all=num_cols_all, cat_cols_all=cat_cols_all)

    # Models
    logit = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=2000
        )),
    ])

    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            random_state=42, class_weight="balanced_subsample",
            n_jobs=-1, oob_score=True
        )),
    ])

    xgb = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=spw
        )),
    ])

    results = {}
    for name, mdl in [("logit", logit), ("rf", rf), ("xgb", xgb)]:
        results[name] = eval_and_plot(name, mdl, train, y_train, test, y_test, sector, suffix)
        # export feature importances per model
        export_feature_importance(mdl, name, sector, suffix, topn=25)

    out_json = MODELS_DIR / f"{sector}_metrics{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[Eval] wrote {out_json}")

if __name__ == "__main__":
    main()
