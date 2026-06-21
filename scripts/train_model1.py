"""
COMPASS Phase 3 — Model 1: Association-Level Delinquency Risk Classifier
=========================================================================
Compares Logistic Regression, Random Forest, and Gradient Boosting.

Evaluation protocol (small-sample aware: n=300, 57 positives):
  1. Stratified 5-fold CV. Out-of-fold (OOF) probabilities are POOLED
     across folds, so every metric is computed once on all 300 held-out
     predictions instead of averaging five noisy ~60-row folds.
  2. The F1 operating threshold is chosen on the pooled OOF curve and
     reported alongside the threshold-free metrics (ROC-AUC, PR-AUC).
  3. NOISE-CEILING ANALYSIS: the same protocol is repeated with the
     generator's true latent risk profile added as an "oracle" feature
     (read from data/full/_risk_profile_key.csv — EVALUATION ONLY, never
     a production feature). The oracle's F1 bounds what any model could
     achieve, because outcomes in the 6-month outcome window contain
     irreducible randomness (monthly payment-miss and recovery draws).

Proposal targets: F1 >= 0.72, ROC-AUC >= 0.80.

Outputs
-------
  data/models/model1_cv_results.csv      pooled-OOF metrics, all models
  data/models/model1_oof_predictions.csv per-association OOF probabilities
  data/models/model1_champion.pkl        champion pipeline fit on all data
  docs/model1_results.png                4-panel evaluation figure

Reproducibility: RANDOM_SEED = 42. Run:  python3 scripts/train_model1.py
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
BASE = Path(__file__).resolve().parent.parent
FEATURES = BASE / "data" / "features"
FULL = BASE / "data" / "full"
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

df = load_features("model1_delinquency_features")
TARGET = "target_delinquency"
print(f"Loaded Model 1 features: {df.shape[0]} rows, "
      f"{int(df[TARGET].sum())} positive ({df[TARGET].mean():.1%})")

# ── 2. Preprocessing ─────────────────────────────────────────────────────
# 15 state levels over 300 rows would be sparse -> 4 macro-regions.
REGION_MAP = {
    "FL": "SOUTH", "GA": "SOUTH", "NC": "SOUTH", "VA": "SOUTH", "TX": "SOUTH",
    "CA": "WEST", "AZ": "WEST", "NV": "WEST", "WA": "WEST", "CO": "WEST",
    "IL": "MIDWEST", "OH": "MIDWEST",
    "NY": "NORTHEAST", "MA": "NORTHEAST", "PA": "NORTHEAST",
}
df["region"] = df["state"].map(REGION_MAP).fillna("OTHER")

NUMERIC = ["total_units", "avg_delinq_rate_12mo", "max_delinq_rate_12mo",
           "last_delinq_rate", "delinq_trend_3mo", "collection_ratio_12mo",
           "lien_rate", "attorney_referral_rate", "rented_pct", "vacant_pct",
           # Phase 3 enrichment — persistence & severity depth
           "pct_months_serious_12mo", "delinq_trend_12mo",
           "delinq_volatility_12mo", "avg_late_stage_rate_12mo"]
CATEGORICAL = ["association_type", "region"]
BINARY = ["coastal_flag"]

X = df[NUMERIC + CATEGORICAL + BINARY].copy()
X["coastal_flag"] = X["coastal_flag"].astype(int)
y = df[TARGET].astype(int)

def make_prep(cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ("bin", "passthrough", BINARY),
    ])

# ── 3. Candidates (regularized for n=300) ────────────────────────────────
CANDIDATES = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", C=0.5, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=600, max_depth=7, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_SEED),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
THRESH_GRID = np.arange(0.05, 0.95, 0.01)

def pooled_oof_eval(X_, y_, clf, cat_cols):
    """Pooled out-of-fold probabilities + metrics for one candidate."""
    pipe = Pipeline([("prep", make_prep(cat_cols)), ("clf", clf)])
    oof = cross_val_predict(pipe, X_, y_, cv=skf, method="predict_proba")[:, 1]
    f1s = [f1_score(y_, (oof >= t).astype(int)) for t in THRESH_GRID]
    k = int(np.argmax(f1s))
    thr = float(THRESH_GRID[k])
    pred = (oof >= thr).astype(int)
    return {
        "oof": oof, "threshold": thr,
        "f1": f1s[k],
        "roc_auc": roc_auc_score(y_, oof),
        "pr_auc": average_precision_score(y_, oof),
        "precision": precision_score(y_, pred),
        "recall": recall_score(y_, pred),
    }

# ── 4. Evaluate the three candidates ─────────────────────────────────────
print("\n=== Pooled out-of-fold evaluation (stratified 5-fold) ===")
results = {}
for name, clf in CANDIDATES.items():
    r = pooled_oof_eval(X, y, clf, CATEGORICAL)
    results[name] = r
    print(f"  {name:20s} F1={r['f1']:.3f} @thr={r['threshold']:.2f}  "
          f"ROC-AUC={r['roc_auc']:.3f}  PR-AUC={r['pr_auc']:.3f}  "
          f"P={r['precision']:.3f}  R={r['recall']:.3f}")

pd.DataFrame([{ "model": n, **{k: v for k, v in r.items() if k != "oof"}}
              for n, r in results.items()]) \
  .to_csv(MODELS_DIR / "model1_cv_results.csv", index=False)

champion_name = max(results, key=lambda n: results[n]["f1"])
champ = results[champion_name]
print(f"\nCHAMPION: {champion_name}  |  OOF F1 = {champ['f1']:.3f} "
      f"@ thr {champ['threshold']:.2f}  |  ROC-AUC = {champ['roc_auc']:.3f}  "
      f"|  PR-AUC = {champ['pr_auc']:.3f}")

# ── 5. Oracle noise-ceiling analysis ─────────────────────────────────────
# The latent risk profile is the generator's ground truth. Adding it as a
# feature bounds the achievable F1: residual error reflects pure outcome-
# window randomness, not model weakness. EVALUATION ONLY.
key = pd.read_csv(FULL / "_risk_profile_key.csv")
df_o = df.merge(key, on="caid")
X_o = df_o[NUMERIC + CATEGORICAL + ["risk_profile"] + BINARY].copy()
X_o["coastal_flag"] = X_o["coastal_flag"].astype(int)
oracle = pooled_oof_eval(X_o, df_o[TARGET].astype(int),
                         CANDIDATES["Gradient Boosting"],
                         CATEGORICAL + ["risk_profile"])
ceiling_pct = 100 * champ["f1"] / oracle["f1"]
print(f"\nORACLE (true risk profile added): F1 ceiling = {oracle['f1']:.3f}")
print(f"Champion reaches {ceiling_pct:.1f}% of the oracle noise ceiling.")
print(f"Targets:  ROC-AUC >= 0.80 -> "
      f"{'MET' if champ['roc_auc'] >= 0.80 else 'NOT MET'} "
      f"({champ['roc_auc']:.3f})")
print(f"          F1 >= 0.72      -> "
      f"{'MET' if champ['f1'] >= 0.72 else 'NOT MET'} "
      f"({champ['f1']:.3f}; oracle ceiling {oracle['f1']:.3f})")

# ── 6. Persist OOF predictions + champion fitted on all data ────────────
oof = champ["oof"]
thr = champ["threshold"]
pred = (oof >= thr).astype(int)
pd.DataFrame({"caid": df["caid"], "y_true": y, "y_prob_oof": oof,
              "y_pred_oof": pred}) \
  .to_csv(MODELS_DIR / "model1_oof_predictions.csv", index=False)

champion_full = Pipeline([("prep", make_prep(CATEGORICAL)),
                          ("clf", CANDIDATES[champion_name])])
champion_full.fit(X, y)
with open(MODELS_DIR / "model1_champion.pkl", "wb") as fh:
    pickle.dump({"pipeline": champion_full, "champion": champion_name,
                 "numeric": NUMERIC, "categorical": CATEGORICAL,
                 "binary": BINARY, "region_map": REGION_MAP,
                 "threshold": thr,
                 "oof_f1": champ["f1"], "oof_roc_auc": champ["roc_auc"],
                 "oof_pr_auc": champ["pr_auc"],
                 "oracle_f1_ceiling": oracle["f1"]}, fh)

# Feature importance from full-data champion
feat_names = champion_full.named_steps["prep"].get_feature_names_out()
clf_fit = champion_full.named_steps["clf"]
importance = (clf_fit.feature_importances_
              if hasattr(clf_fit, "feature_importances_")
              else np.abs(clf_fit.coef_[0]))
imp = pd.Series(importance, index=[n.split("__")[-1] for n in feat_names]) \
        .sort_values(ascending=True).tail(10)

# ── 7. Figure ────────────────────────────────────────────────────────────
cm = confusion_matrix(y, pred)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("COMPASS Model 1 — Delinquency Risk Classifier (Phase 3)",
             fontsize=15, fontweight="bold", color=NAVY)

# (a) F1 comparison + targets + oracle ceiling
ax = axes[0, 0]
order = list(CANDIDATES)
vals = [results[m]["f1"] for m in order]
colors = [GREEN if m == champion_name else BLUE for m in order]
ax.bar(order, vals, color=colors, edgecolor=NAVY)
ax.axhline(0.72, color=RED, ls="--", lw=1.5, label="Proposal target F1 = 0.72")
ax.axhline(oracle["f1"], color=ORANGE, ls=":", lw=2,
           label=f"Oracle noise ceiling = {oracle['f1']:.3f}")
ax.set_ylim(0, 1.0); ax.set_ylabel("Pooled OOF F1")
ax.set_title("Model comparison — pooled out-of-fold F1", color=NAVY)
ax.legend(loc="upper left", fontsize=9); ax.tick_params(axis="x", rotation=10)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", color=NAVY)

# (b) ROC curve (pooled OOF)
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y, oof)
ax.plot(fpr, tpr, color=GREEN, lw=2.5,
        label=f"{champion_name} (OOF AUC = {champ['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve — pooled out-of-fold", color=NAVY)
ax.legend(loc="lower right")

# (c) Confusion matrix (pooled OOF at tuned threshold)
ax = axes[1, 0]
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=18,
            fontweight="bold", color="white" if v > cm.max()/2 else NAVY)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: healthy", "Pred: at-risk"])
ax.set_yticklabels(["True: healthy", "True: at-risk"])
ax.set_title(f"Confusion matrix — pooled OOF (F1 = {champ['f1']:.3f} "
             f"@ thr {thr:.2f})", color=NAVY)

# (d) Feature importance
ax = axes[1, 1]
ax.barh(imp.index, imp.values, color=BLUE, edgecolor=NAVY)
ax.set_title(f"Top 10 drivers — {champion_name}", color=NAVY)
ax.set_xlabel("Feature importance" if hasattr(clf_fit, "feature_importances_")
              else "|coefficient| (standardized)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_png = DOCS / "model1_results.png"
plt.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")
print("Model 1 training complete.")
