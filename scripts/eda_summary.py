"""Generate EDA summary visualizations for Phase 2 deliverable."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FEAT = BASE / "data" / "features"
FULL = BASE / "data" / "full"
OUT = BASE / "docs"
OUT.mkdir(parents=True, exist_ok=True)

def load(folder, name):
    """Prefer Parquet (used on the Mac); fall back to CSV (sandbox)."""
    pq = folder / f"{name}.parquet"
    csv = folder / f"{name}.csv"
    try:
        return pd.read_parquet(pq)
    except Exception:
        return pd.read_csv(csv)

m1 = load(FEAT, "model1_delinquency_features")
m2 = load(FEAT, "model2_reserve_features")
m3 = load(FEAT, "model3_anomaly_features")
assoc = load(FULL, "associations")
reserves = load(FULL, "reserve_funds")
fraud = pd.read_csv(FULL/"_fraud_key.csv")

NAVY="#1B3A6B"; BLUE="#2E75B6"; GREEN="#1F6B3A"; ORANGE="#8B4500"; RED="#8B1A1A"

fig = plt.figure(figsize=(16, 11))
fig.suptitle("COMPASS — Exploratory Data Analysis (Phase 2 Feature Engineering)",
             fontsize=17, fontweight='bold', color=NAVY, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.28,
                       top=0.91, bottom=0.06, left=0.06, right=0.97)

# 1. Association type distribution
ax1 = fig.add_subplot(gs[0,0])
assoc["association_type"].value_counts().plot(kind="bar", ax=ax1, color=BLUE, edgecolor='black', linewidth=0.5)
ax1.set_title("Association Types", fontweight='bold', color=NAVY)
ax1.set_ylabel("Count"); ax1.tick_params(axis='x', rotation=0)

# 2. Geographic distribution (top 8)
ax2 = fig.add_subplot(gs[0,1])
assoc["state"].value_counts().head(8).plot(kind="bar", ax=ax2, color=NAVY, edgecolor='black', linewidth=0.5)
ax2.set_title("Top 8 States", fontweight='bold', color=NAVY)
ax2.set_ylabel("Count"); ax2.tick_params(axis='x', rotation=0)

# 3. Units per association
ax3 = fig.add_subplot(gs[0,2])
ax3.hist(assoc["total_units"], bins=30, color=BLUE, edgecolor='black', linewidth=0.5)
ax3.set_title("Units per Association", fontweight='bold', color=NAVY)
ax3.set_xlabel("Total Units"); ax3.set_ylabel("Count")

# 4. Model 1 target balance
ax4 = fig.add_subplot(gs[1,0])
m1["target_delinquency"].value_counts().sort_index().plot(kind="bar", ax=ax4, color=[GREEN,RED], edgecolor='black', linewidth=0.5)
ax4.set_title("Model 1: Delinquency Target\n(0=No Risk, 1=At Risk)", fontweight='bold', color=GREEN)
ax4.set_ylabel("Count"); ax4.tick_params(axis='x', rotation=0)
ax4.set_xticklabels(['No Risk','At Risk'])

# 5. Model 1 key feature separation
ax5 = fig.add_subplot(gs[1,1])
for t,c,l in [(0,GREEN,'No Risk'),(1,RED,'At Risk')]:
    ax5.hist(m1[m1["target_delinquency"]==t]["avg_delinq_rate_12mo"], bins=20, alpha=0.6, color=c, label=l, edgecolor='black', linewidth=0.3)
ax5.set_title("Avg Delinquency Rate by Target", fontweight='bold', color=GREEN)
ax5.set_xlabel("Avg Delinquency Rate (12mo, %)"); ax5.set_ylabel("Count"); ax5.legend(fontsize=8)

# 6. Model 2 target balance
ax6 = fig.add_subplot(gs[1,2])
m2["target_reserve_failure"].value_counts().sort_index().plot(kind="bar", ax=ax6, color=[GREEN,RED], edgecolor='black', linewidth=0.5)
ax6.set_title("Model 2: Reserve Failure Target", fontweight='bold', color=ORANGE)
ax6.set_ylabel("Count"); ax6.tick_params(axis='x', rotation=0)
ax6.set_xticklabels(['Healthy','Will Fail'])

# 7. Model 2 reserve funding distribution
ax7 = fig.add_subplot(gs[2,0])
ax7.hist(m2["avg_funded_pct"], bins=25, color=ORANGE, edgecolor='black', linewidth=0.5)
ax7.axvline(25, color=RED, linestyle='--', linewidth=2, label='25% failure threshold')
ax7.set_title("Reserve Fund % Funded (bimodal)", fontweight='bold', color=ORANGE)
ax7.set_xlabel("Avg % Funded"); ax7.set_ylabel("Count"); ax7.legend(fontsize=8)

# 8. Model 2 deferred maintenance vs target
ax8 = fig.add_subplot(gs[2,1])
for t,c,l in [(0,GREEN,'Healthy'),(1,RED,'Will Fail')]:
    sub = m2[m2["target_reserve_failure"]==t]
    ax8.scatter(sub["avg_funded_pct"], sub["critical_deferred_count"], alpha=0.5, color=c, label=l, s=18)
ax8.set_title("Reserve % vs Critical Deferred WOs", fontweight='bold', color=ORANGE)
ax8.set_xlabel("Avg % Funded"); ax8.set_ylabel("Critical Deferred WO Count"); ax8.legend(fontsize=8)

# 9. Model 3 anomaly separation
ax9 = fig.add_subplot(gs[2,2])
m3["is_fraud"] = m3["caid"].isin(fraud["caid"]).astype(int)
normal = m3[m3["is_fraud"]==0]
fraud_pts = m3[m3["is_fraud"]==1]
ax9.scatter(normal["vendor_hhi"], normal["collection_gap"], alpha=0.4, color=BLUE, label='Normal', s=18)
ax9.scatter(fraud_pts["vendor_hhi"], fraud_pts["collection_gap"], alpha=0.9, color=RED, label='Embedded Fraud', s=60, marker='X', edgecolor='black', linewidth=0.5)
ax9.set_title("Model 3: Anomaly Feature Space", fontweight='bold', color=RED)
ax9.set_xlabel("Vendor HHI (concentration)"); ax9.set_ylabel("Collection Gap"); ax9.legend(fontsize=8)

plt.savefig(OUT/"eda_summary.png", dpi=160, bbox_inches='tight', facecolor='white')
print(f"EDA summary saved to {OUT/'eda_summary.png'}")
