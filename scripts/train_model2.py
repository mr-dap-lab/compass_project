"""
COMPASS Phase 3 — Model 2: Reserve Fund Failure Probability
============================================================
Predicts the probability that an association's reserve fund falls below
25% funded in the outcome window. Unlike Model 1 (a ranking/alerting
classifier), Model 2 is consumed as a PROBABILITY, so calibration quality
matters as much as discrimination.

Protocol:
  1. Three candidates (LogReg / RF / GB), each wrapped in
     CalibratedClassifierCV(method="sigmoid")  — Platt scaling.
  2. Stratified 5-fold CV; out-of-fold probabilities pooled.
  3. Selection by Brier score (lower = better calibrated + accurate),
     with ROC-AUC reported for discrimination.

Proposal targets: Brier <= 0.15, ROC-AUC >= 0.78.

Outputs
-------
  data/models/model2_cv_results.csv      pooled-OOF metrics, all models
  data/models/model2_oof_predictions.csv per-association OOF probabilities
  data/models/model2_champion.pkl        calibrated champion fit on all data
  docs/model2_results.png                4-panel evaluation figure

Reproducibility: RANDOM_SEED = 42. Run:  python3 scripts/train_model2.py
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
BASE = Path(__file__).resolve().parent.parent
FEATURES = BASE / "data" / "features"
MODELS_DIR = BASE / "data" / "models"
DOCS = BASE / "docs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NAVY, BLUE, GREEN, ORANGE, RED = "#1B3A6B", "#2E75B6", "#1F6B3A", "#8B4500", "#8B1A1A"

# ── 1. Load ──────────────────────────────────────────────────────────────
def load_features(stem: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(FEATURES / f"{stem}.parquet")
    except Exception:
        return pd.read_csv(FEATURES / f"{stem}.csv")

df = load_features("model2_reserve_features")
TARGET = "target_reserve_failure"
print(f"Loaded Model 2 features: {df.shape[0]} rows, "
      f"{int(df[TARGET].sum())} positive ({df[TARGET].mean():.1%})")

NUMERIC = ["building_age", "avg_funded_pct", "last_funded_pct",
           "funding_trend", "contribution_ratio", "total_open_workorders",
           "critical_deferred_count", "avg_deferred_days",
           "open_workorder_cost"]
CATEGORICAL = ["association_type"]
BINARY = ["coastal_flag"]

X = df[NUMERIC + CATEGORICAL + BINARY].copy()
X["coastal_flag"] = X["coastal_flag"].astype(int)
y = df[TARGET].astype(int)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
    ("bin", "passthrough", BINARY),
])

# ── 2. Candidates: each Platt-calibrated ─────────────────────────────────
BASE_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, C=0.5, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=600, max_depth=6, min_samples_leaf=4,
        random_state=RANDOM_SEED),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED),
}

def calibrated(clf):
    """Platt scaling (sigmoid) with internal 3-fold calibration split."""
    return CalibratedClassifierCV(clf, method="sigmoid", cv=3)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ── 3. Pooled OOF evaluation ─────────────────────────────────────────────
print("\n=== Pooled out-of-fold evaluation (Platt-calibrated, 5-fold) ===")
results = {}
for name, clf in BASE_MODELS.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", calibrated(clf))])
    oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    results[name] = {
        "oof": oof,
        "brier": brier_score_loss(y, oof),
        "roc_auc": roc_auc_score(y, oof),
        "pr_auc": average_precision_score(y, oof),
        "f1_at_0.5": f1_score(y, (oof >= 0.5).astype(int)),
    }
    r = results[name]
    print(f"  {name:20s} Brier={r['brier']:.4f}  ROC-AUC={r['roc_auc']:.3f}  "
          f"PR-AUC={r['pr_auc']:.3f}  F1@0.5={r['f1_at_0.5']:.3f}")

pd.DataFrame([{ "model": n, **{k: v for k, v in r.items() if k != "oof"}}
              for n, r in results.items()]) \
  .to_csv(MODELS_DIR / "model2_cv_results.csv", index=False)

champion_name = min(results, key=lambda n: results[n]["brier"])
champ = results[champion_name]
oof = champ["oof"]
print(f"\nCHAMPION (lowest Brier): {champion_name}")
print(f"  Brier = {champ['brier']:.4f}  |  ROC-AUC = {champ['roc_auc']:.3f}  "
      f"|  PR-AUC = {champ['pr_auc']:.3f}")
print(f"Targets:  Brier <= 0.15   -> "
      f"{'MET' if champ['brier'] <= 0.15 else 'NOT MET'} ({champ['brier']:.4f})")
print(f"          ROC-AUC >= 0.78 -> "
      f"{'MET' if champ['roc_auc'] >= 0.78 else 'NOT MET'} ({champ['roc_auc']:.3f})")

# ── 4. Persist OOF predictions + champion fitted on all data ────────────
pd.DataFrame({"caid": df["caid"], "y_true": y, "y_prob_oof": oof}) \
  .to_csv(MODELS_DIR / "model2_oof_predictions.csv", index=False)

champion_full = Pipeline([("prep", preprocessor),
                          ("clf", calibrated(BASE_MODELS[champion_name]))])
champion_full.fit(X, y)
with open(MODELS_DIR / "model2_champion.pkl", "wb") as fh:
    pickle.dump({"pipeline": champion_full, "champion": champion_name,
                 "numeric": NUMERIC, "categorical": CATEGORICAL,
                 "binary": BINARY,
                 "oof_brier": champ["brier"],
                 "oof_roc_auc": champ["roc_auc"],
                 "oof_pr_auc": champ["pr_auc"]}, fh)

# Feature importance: take from the uncalibrated base model fit on all data
base_pipe = Pipeline([("prep", preprocessor),
                      ("clf", BASE_MODELS[champion_name])])
base_pipe.fit(X, y)
feat_names = base_pipe.named_steps["prep"].get_feature_names_out()
clf_fit = base_pipe.named_steps["clf"]
importance = (clf_fit.feature_importances_
              if hasattr(clf_fit, "feature_importances_")
              else np.abs(clf_fit.coef_[0]))
imp = pd.Series(importance, index=[n.split("__")[-1] for n in feat_names]) \
        .sort_values(ascending=True).tail(10)

# ── 5. Figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("COMPASS Model 2 — Reserve Fund Failure Probability (Phase 3)",
             fontsize=15, fontweight="bold", color=NAVY)

# (a) Brier comparison
ax = axes[0, 0]
order = list(BASE_MODELS)
vals = [results[m]["brier"] for m in order]
colors = [GREEN if m == champion_name else BLUE for m in order]
ax.bar(order, vals, color=colors, edgecolor=NAVY)
ax.axhline(0.15, color=RED, ls="--", lw=1.5, label="Target Brier = 0.15 (max)")
ax.set_ylabel("Brier score (pooled OOF — lower is better)")
ax.set_title("Model comparison — Brier score", color=NAVY)
ax.legend(); ax.tick_params(axis="x", rotation=10)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontweight="bold", color=NAVY)

# (b) Calibration curve
ax = axes[0, 1]
frac_pos, mean_pred = calibration_curve(y, oof, n_bins=8, strategy="quantile")
ax.plot(mean_pred, frac_pos, marker="o", color=GREEN, lw=2,
        label=f"{champion_name} (Platt)")
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed failure fraction")
ax.set_title("Calibration curve — pooled OOF (quantile bins)", color=NAVY)
ax.legend(loc="upper left")

# (c) ROC curve
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y, oof)
ax.plot(fpr, tpr, color=GREEN, lw=2.5,
        label=f"OOF ROC-AUC = {champ['roc_auc']:.3f}")
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve — pooled out-of-fold", color=NAVY)
ax.legend(loc="lower right")

# (d) Feature importance
ax = axes[1, 1]
ax.barh(imp.index, imp.values, color=BLUE, edgecolor=NAVY)
ax.set_title(f"Top drivers — {champion_name} (base estimator)", color=NAVY)
ax.set_xlabel("Feature importance" if hasattr(clf_fit, "feature_importances_")
              else "|coefficient| (standardized)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_png = DOCS / "model2_results.png"
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")
print("Model 2 training complete.")
