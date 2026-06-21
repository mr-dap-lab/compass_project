"""
COMPASS — Model 3: Financial Anomaly Detection (Isolation Forest)
Phase 4 Deliverable

Trains an unsupervised Isolation Forest on the 8 engineered anomaly features
to surface associations with statistically unusual financial behavior (fraud,
mismanagement, or data-quality issues) WITHOUT using labeled fraud examples.

The 6 embedded fraud cases in data/full/_fraud_key.csv are used ONLY to
evaluate the model (Precision@K), never as training input.

Method:   StandardScaler -> IsolationForest(contamination=0.02, random_state=42)
Evaluate: rank all 300 associations by anomaly score; take top-K (K=6,
          matching the known fraud count); Precision@K target >= 0.70.
Explain:  SHAP values on the flagged anomalies (Lundberg & Lee, 2017); if SHAP
          is unavailable, fall back to per-feature z-score contributions.

Outputs:  data/models/model3_isolation_forest.pkl
          data/models/model3_anomaly_scores.csv
          docs/model3_results.png

Author: Diego Avella
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).resolve().parent.parent
FEAT = BASE / "data" / "features"
FULL = BASE / "data" / "full"
MODELS = BASE / "data" / "models"
DOCS = BASE / "docs"
MODELS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def load(folder, name):
    """Prefer Parquet (used on the Mac); fall back to CSV (sandbox)."""
    pq = folder / f"{name}.parquet"
    csv = folder / f"{name}.csv"
    try:
        return pd.read_parquet(pq)
    except Exception:
        return pd.read_csv(csv)


def main():
    print("=" * 64)
    print("COMPASS Model 3 — Financial Anomaly Detection (Isolation Forest)")
    print("=" * 64)

    feats = load(FEAT, "model3_anomaly_features")
    fraud_key = pd.read_csv(FULL / "_fraud_key.csv")
    fraud_caids = set(fraud_key["caid"])
    n_fraud = len(fraud_caids)
    print(f"Loaded Model 3 features: {len(feats)} associations, "
          f"{len(feats.columns) - 1} features")
    print(f"Embedded fraud cases (evaluation only): {n_fraud}")

    feature_cols = [c for c in feats.columns if c != "caid"]
    # Sanitize: replace +/-inf with NaN, then fill NaN with the column median
    # (a single inf can appear in reserve_volatility when a reserve series is
    # degenerate; median imputation is robust and does not distort the ranking).
    feats[feature_cols] = feats[feature_cols].replace([np.inf, -np.inf], np.nan)
    feats[feature_cols] = feats[feature_cols].fillna(feats[feature_cols].median())
    X = feats[feature_cols].values
    caids = feats["caid"].values

    # ── Standardize, then Isolation Forest ──────────────────────────────
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.02,          # ~6 of 300 expected anomalies
        max_samples="auto",
        random_state=SEED,
    )
    iso.fit(Xs)

    # decision_function: higher = more normal; we invert so higher = more anomalous
    raw = iso.decision_function(Xs)
    anomaly_score = -raw
    pred_flag = (iso.predict(Xs) == -1).astype(int)

    results = pd.DataFrame({
        "caid": caids,
        "anomaly_score": anomaly_score,
        "iforest_flag": pred_flag,
        "is_true_fraud": [int(c in fraud_caids) for c in caids],
    }).sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # ── Precision@K (K = number of known fraud cases) ───────────────────
    K = n_fraud
    topK = results.head(K)
    hits = int(topK["is_true_fraud"].sum())
    precision_at_k = hits / K
    print("-" * 64)
    print(f"Top-{K} flagged associations (by anomaly score):")
    for _, r in topK.iterrows():
        tag = "  <-- TRUE FRAUD" if r["is_true_fraud"] else ""
        print(f"    {r['caid']:18s} score={r['anomaly_score']:+.4f}{tag}")
    print("-" * 64)
    print(f"Precision@{K} = {hits}/{K} = {precision_at_k:.3f}")
    target = 0.70
    verdict = "MET" if precision_at_k >= target else "NOT MET"
    print(f"Target Precision@K >= {target:.2f} -> {verdict}")

    # Average precision over the full ranking (how well fraud sorts to the top)
    from sklearn.metrics import average_precision_score, roc_auc_score
    ap = average_precision_score(results["is_true_fraud"], results["anomaly_score"])
    auc = roc_auc_score(results["is_true_fraud"], results["anomaly_score"])
    print(f"Average precision (full ranking) = {ap:.3f}")
    print(f"ROC-AUC (anomaly score vs. fraud label) = {auc:.3f}")

    # ── Explanations: SHAP on flagged anomalies, fallback to z-score ────
    explain_method = "shap"
    feature_contrib = None
    try:
        import shap
        # KernelExplainer is model-agnostic; background = the standardized data
        # Use a small background sample for speed.
        bg = shap.sample(Xs, 50, random_state=SEED)
        explainer = shap.KernelExplainer(lambda d: -iso.decision_function(d), bg)
        topK_idx = [list(caids).index(c) for c in topK["caid"]]
        shap_vals = explainer.shap_values(Xs[topK_idx], nsamples=100, silent=True)
        feature_contrib = np.abs(np.array(shap_vals)).mean(axis=0)
        print("-" * 64)
        print("Explanation method: SHAP (mean |value| over top-K flags)")
    except Exception as e:
        explain_method = "zscore"
        # Fallback: mean absolute z-score contribution among flagged anomalies
        topK_idx = [list(caids).index(c) for c in topK["caid"]]
        feature_contrib = np.abs(Xs[topK_idx]).mean(axis=0)
        print("-" * 64)
        print(f"Explanation method: z-score fallback (SHAP unavailable: {e})")

    contrib = (pd.DataFrame({"feature": feature_cols, "contribution": feature_contrib})
               .sort_values("contribution", ascending=False).reset_index(drop=True))
    print("Top anomaly drivers:")
    for _, r in contrib.head(5).iterrows():
        print(f"    {r['feature']:22s} {r['contribution']:.4f}")

    # ── Persist ─────────────────────────────────────────────────────────
    results.to_csv(MODELS / "model3_anomaly_scores.csv", index=False)
    bundle = {
        "scaler": scaler,
        "model": iso,
        "feature_cols": feature_cols,
        "contamination": 0.02,
        "precision_at_k": precision_at_k,
        "k": K,
        "average_precision": ap,
        "roc_auc": auc,
        "explain_method": explain_method,
        "feature_contrib": dict(zip(feature_cols, [float(x) for x in feature_contrib])),
    }
    with open(MODELS / "model3_isolation_forest.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # ── Figure (color allowed for graphs) ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("COMPASS Model 3 — Financial Anomaly Detection (Phase 4)",
                 fontsize=15, fontweight="bold", color="#1B3A6B")

    # Panel 1: score distribution with fraud cases marked
    ax = axes[0]
    normal = results[results["is_true_fraud"] == 0]["anomaly_score"]
    fraud = results[results["is_true_fraud"] == 1]["anomaly_score"]
    ax.hist(normal, bins=40, color="#2E75B6", alpha=0.8, label="Normal")
    for i, v in enumerate(fraud):
        ax.axvline(v, color="#8B1A1A", linewidth=1.6,
                   label="Embedded fraud" if i == 0 else None)
    ax.set_title("Anomaly score distribution", color="#1B3A6B")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Count")
    ax.legend()

    # Panel 2: top-K ranking, colored by true/false
    ax = axes[1]
    tk = results.head(K)
    colors = ["#1F6B3A" if v else "#8B4500" for v in tk["is_true_fraud"]]
    ax.barh(range(len(tk))[::-1], tk["anomaly_score"], color=colors)
    ax.set_yticks(range(len(tk))[::-1])
    ax.set_yticklabels(tk["caid"], fontsize=8)
    ax.set_title(f"Top-{K} flagged  (Precision@{K} = {precision_at_k:.2f})",
                 color="#1B3A6B")
    ax.set_xlabel("Anomaly score")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1F6B3A", label="True fraud"),
                       Patch(color="#8B4500", label="False alarm")], fontsize=8)

    # Panel 3: feature contributions
    ax = axes[2]
    cc = contrib.iloc[::-1]
    ax.barh(cc["feature"], cc["contribution"], color="#2E75B6")
    ax.set_title(f"Anomaly drivers ({explain_method})", color="#1B3A6B")
    ax.set_xlabel("Mean contribution")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = DOCS / "model3_results.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nFigure saved: {out_png}")
    print("Model 3 training complete.")


if __name__ == "__main__":
    main()
