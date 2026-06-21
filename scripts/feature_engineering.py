"""
COMPASS — Feature Engineering Pipeline
Phase 2 (Weeks 3-4) Deliverable

Transforms the raw CDES synthetic dataset into model-ready feature tables with
engineered features and target labels for all three predictive models:

  Model 1 — Delinquency Risk    (binary: will assoc exceed 10% delinquency in 90d)
  Model 2 — Reserve Fund Failure (binary: will reserve fall below 25% in 24mo)
  Model 3 — Anomaly Detection    (unsupervised feature matrix)

Output: feature tables in data/features/
Author: Diego Avella
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FULL = BASE / "data" / "full"
OUT = BASE / "data" / "features"
OUT.mkdir(parents=True, exist_ok=True)

PERIOD_MONTHS = 18
DECISION_POINT_MONTH = 12  # use months 1-12 as features, 13-18 as outcome window

def load(name):
    """Parquet first, CSV fallback (CSV path re-parses date columns)."""
    try:
        return pd.read_parquet(FULL / f"{name}.parquet")
    except Exception:
        df = pd.read_csv(FULL / f"{name}.csv")
        for col in df.columns:
            if col.endswith("_date") or col in ("snapshot_date", "transaction_date"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

print("="*64)
print("COMPASS Feature Engineering Pipeline — Phase 2")
print("="*64)
print("  Loading raw tables...")

assoc = load("associations")
units = load("units")
fees = load("fee_schedules")
ledger = load("payment_ledger")
delinq = load("delinquency_records")
reserves = load("reserve_funds")
work_orders = load("work_orders")
violations = load("violations")

# parse dates
for df, col in [(ledger,"transaction_date"),(delinq,"snapshot_date"),
                (reserves,"snapshot_date"),(work_orders,"created_date"),
                (violations,"issue_date")]:
    df[col] = pd.to_datetime(df[col])

PERIOD = sorted(delinq["snapshot_date"].dt.to_period("M").unique())
split_period = PERIOD[DECISION_POINT_MONTH-1]   # month 12
print(f"  Feature window: months 1-{DECISION_POINT_MONTH}; outcome window: months {DECISION_POINT_MONTH+1}-{PERIOD_MONTHS}")

# unit counts per association (actual generated)
unit_counts = units.groupby("caid").size().rename("generated_units")

# =============================================================================
# MODEL 1 — DELINQUENCY RISK FEATURES
# =============================================================================
print("\n  [Model 1] Engineering delinquency risk features...")

def delinquency_rate_by_month(delinq, units):
    """Monthly delinquency rate per association."""
    d = delinq.copy()
    d["period"] = d["snapshot_date"].dt.to_period("M")
    # count delinquent units (61+ days = serious) per assoc per month
    serious = d[d["aging_bucket"].isin(["DAYS_61_90","DAYS_91_120","DAYS_OVER_120"])]
    delinq_units = serious.groupby(["caid","period"])["unit_id"].nunique().rename("delinq_units")
    rate = delinq_units.reset_index().merge(unit_counts.reset_index(), on="caid")
    rate["delinq_rate"] = 100.0 * rate["delinq_units"] / rate["generated_units"]
    # Phase 3 enrichment: late-stage (91+ days) rate — captures severity depth
    late = d[d["aging_bucket"].isin(["DAYS_91_120","DAYS_OVER_120"])]
    late_units = late.groupby(["caid","period"])["unit_id"].nunique().rename("late_units")
    rate = rate.merge(late_units.reset_index(), on=["caid","period"], how="left")
    rate["late_units"] = rate["late_units"].fillna(0)
    rate["late_stage_rate"] = 100.0 * rate["late_units"] / rate["generated_units"]
    return rate

drate = delinquency_rate_by_month(delinq, units)

# FEATURE WINDOW: months 1-12
feat_window = drate[drate["period"] <= split_period]
# OUTCOME WINDOW: months 13-18
out_window = drate[drate["period"] > split_period]

m1_rows = []
for cid in assoc["caid"]:
    fw = feat_window[feat_window["caid"]==cid].sort_values("period")
    ow = out_window[out_window["caid"]==cid]

    # ledger-based features (months 1-12)
    cl = ledger[(ledger["caid"]==cid) & (ledger["transaction_date"].dt.to_period("M")<=split_period)]
    charges = cl[cl["transaction_type"]=="ASSESSMENT_CHARGE"]["amount"].abs().sum()
    payments = cl[cl["transaction_type"]=="PAYMENT"]["amount"].sum()
    collection_ratio = (payments / charges) if charges > 0 else 1.0

    # delinquency trajectory features
    rates = fw["delinq_rate"].tolist() if len(fw) else [0.0]
    avg_delinq = np.mean(rates) if rates else 0.0
    max_delinq = np.max(rates) if rates else 0.0
    last_delinq = rates[-1] if rates else 0.0
    # trend: slope of last 3 months
    if len(rates) >= 3:
        trend = np.polyfit(range(3), rates[-3:], 1)[0]
    else:
        trend = 0.0

    # --- Phase 3 feature enrichment (persistence & severity depth) ---
    # persistence: fraction of the 12 feature-window months with serious
    # delinquency above 5% (chronic vs one-off stress)
    pct_months_serious = sum(1 for r in rates if r > 5.0) / 12.0
    # full-window trend (12-month slope), more stable than 3-month slope
    trend_12 = np.polyfit(range(len(rates)), rates, 1)[0] if len(rates) >= 4 else 0.0
    # volatility of the monthly serious-delinquency rate
    volatility = float(np.std(rates)) if len(rates) >= 2 else 0.0
    # late-stage (91+ days) severity: mean monthly rate in the window
    late_rates = fw["late_stage_rate"].tolist() if len(fw) else [0.0]
    avg_late_stage = np.mean(late_rates) if late_rates else 0.0

    # lien/attorney signals
    cd = delinq[(delinq["caid"]==cid) & (delinq["snapshot_date"].dt.to_period("M")<=split_period)]
    lien_rate = cd["lien_filed"].mean() if len(cd) else 0.0
    atty_rate = cd["attorney_referred"].mean() if len(cd) else 0.0

    # occupancy / rental exposure
    au = units[units["caid"]==cid]
    rented_pct = (au["occupancy_status"]=="RENTED").mean() if len(au) else 0.0
    vacant_pct = (au["occupancy_status"]=="VACANT").mean() if len(au) else 0.0

    # TARGET: does delinq rate exceed 10% at any point in outcome window?
    target = int((ow["delinq_rate"] > 10.0).any()) if len(ow) else 0

    a = assoc[assoc["caid"]==cid].iloc[0]
    m1_rows.append({
        "caid": cid,
        "association_type": a["association_type"],
        "state": a["state"],
        "total_units": a["total_units"],
        "coastal_flag": int(a["coastal_flag"]),
        "avg_delinq_rate_12mo": round(avg_delinq,3),
        "max_delinq_rate_12mo": round(max_delinq,3),
        "last_delinq_rate": round(last_delinq,3),
        "delinq_trend_3mo": round(trend,4),
        "pct_months_serious_12mo": round(pct_months_serious,4),
        "delinq_trend_12mo": round(trend_12,4),
        "delinq_volatility_12mo": round(volatility,4),
        "avg_late_stage_rate_12mo": round(avg_late_stage,4),
        "collection_ratio_12mo": round(collection_ratio,4),
        "lien_rate": round(lien_rate,4),
        "attorney_referral_rate": round(atty_rate,4),
        "rented_pct": round(rented_pct,4),
        "vacant_pct": round(vacant_pct,4),
        "target_delinquency": target,
    })

m1 = pd.DataFrame(m1_rows)
print(f"        {len(m1)} rows, {m1['target_delinquency'].sum()} positive ({100*m1['target_delinquency'].mean():.1f}%)")

# =============================================================================
# MODEL 2 — RESERVE FUND FAILURE FEATURES
# =============================================================================
print("  [Model 2] Engineering reserve fund failure features...")

m2_rows = []
for cid in assoc["caid"]:
    a = assoc[assoc["caid"]==cid].iloc[0]
    rf = reserves[reserves["caid"]==cid].sort_values("snapshot_date")
    if len(rf) < 2:
        continue

    # split reserve snapshots into feature window (first half) and outcome (second half)
    n = len(rf)
    feat_rf = rf.iloc[:max(1,n//2)]
    out_rf = rf.iloc[n//2:]

    pcts = feat_rf["percent_funded"].tolist()
    avg_funded = np.mean(pcts)
    last_funded = pcts[-1]
    funding_trend = np.polyfit(range(len(pcts)), pcts, 1)[0] if len(pcts)>=2 else 0.0

    # deferred maintenance load
    wo = work_orders[work_orders["caid"]==cid]
    deferred = wo[wo["status"].isin(["DEFERRED","OPEN"])]
    critical_deferred = deferred[deferred["category"].isin(["STRUCTURAL","ROOF","HVAC","ELEVATOR","PLUMBING"])]
    avg_deferred_days = deferred["deferred_days"].mean() if len(deferred) else 0.0
    open_cost = deferred["estimated_cost"].sum()

    # building age
    year_built = a["year_built"] if pd.notna(a["year_built"]) else 1995
    building_age = 2025 - int(year_built)

    # annual contribution adequacy
    avg_contrib = feat_rf["annual_contribution"].mean()
    avg_target = feat_rf["fully_funded_target"].mean()
    contrib_ratio = (avg_contrib / avg_target) if avg_target>0 else 0.0

    # TARGET: does reserve fall below 25% in outcome window?
    target = int((out_rf["percent_funded"] < 25.0).any())

    m2_rows.append({
        "caid": cid,
        "association_type": a["association_type"],
        "coastal_flag": int(a["coastal_flag"]),
        "building_age": building_age,
        "avg_funded_pct": round(avg_funded,3),
        "last_funded_pct": round(last_funded,3),
        "funding_trend": round(funding_trend,4),
        "contribution_ratio": round(contrib_ratio,5),
        "total_open_workorders": len(deferred),
        "critical_deferred_count": len(critical_deferred),
        "avg_deferred_days": round(avg_deferred_days,1),
        "open_workorder_cost": round(open_cost,2),
        "target_reserve_failure": target,
    })

m2 = pd.DataFrame(m2_rows)
print(f"        {len(m2)} rows, {m2['target_reserve_failure'].sum()} positive ({100*m2['target_reserve_failure'].mean():.1f}%)")

# =============================================================================
# MODEL 3 — ANOMALY DETECTION FEATURES (unsupervised)
# =============================================================================
print("  [Model 3] Engineering anomaly detection features...")

def hhi(series):
    """Herfindahl-Hirschman Index of concentration (0-10000)."""
    total = series.sum()
    if total <= 0: return 0.0
    shares = (series/total)*100
    return float((shares**2).sum())

m3_rows = []
for cid in assoc["caid"]:
    a = assoc[assoc["caid"]==cid].iloc[0]
    cl = ledger[ledger["caid"]==cid]

    # vendor payment concentration (ADJUSTMENT/WIRE = vendor-like outflows)
    outflows = cl[cl["transaction_type"]=="ADJUSTMENT"]
    n_outflow = len(outflows)
    outflow_total = outflows["amount"].abs().sum()
    # simulate vendor concentration: bucket outflows into pseudo-vendors by amount band
    if n_outflow > 0:
        bands = (outflows["amount"].abs()//5000).astype(int)
        vendor_hhi = hhi(bands.value_counts())
    else:
        vendor_hhi = 0.0

    # collection gap vs occupancy
    au = units[units["caid"]==cid]
    occupied_pct = (au["occupancy_status"]!="VACANT").mean() if len(au) else 1.0
    charges = cl[cl["transaction_type"]=="ASSESSMENT_CHARGE"]["amount"].abs().sum()
    payments = cl[cl["transaction_type"]=="PAYMENT"]["amount"].sum()
    collection_rate = (payments/charges) if charges>0 else 1.0
    collection_gap = occupied_pct - collection_rate   # high = suspicious

    # reserve volatility
    rf = reserves[reserves["caid"]==cid].sort_values("snapshot_date")
    if len(rf) >= 2:
        reserve_volatility = rf["percent_funded"].pct_change().abs().max()
        reserve_max_drop = (rf["percent_funded"].diff().min())
    else:
        reserve_volatility = 0.0
        reserve_max_drop = 0.0

    m3_rows.append({
        "caid": cid,
        "vendor_hhi": round(vendor_hhi,2),
        "outflow_count": n_outflow,
        "outflow_total": round(outflow_total,2),
        "occupied_pct": round(occupied_pct,4),
        "collection_rate": round(collection_rate,4),
        "collection_gap": round(collection_gap,4),
        "reserve_volatility": round(float(reserve_volatility) if pd.notna(reserve_volatility) else 0.0,4),
        "reserve_max_drop": round(float(reserve_max_drop) if pd.notna(reserve_max_drop) else 0.0,3),
    })

m3 = pd.DataFrame(m3_rows)
print(f"        {len(m3)} rows, {m3.shape[1]-1} features")

# =============================================================================
# WRITE OUTPUTS
# =============================================================================
print("-"*64)
try: m1.to_parquet(OUT/"model1_delinquency_features.parquet", index=False)
except Exception: pass
m1.to_csv(OUT/"model1_delinquency_features.csv", index=False)
try: m2.to_parquet(OUT/"model2_reserve_features.parquet", index=False)
except Exception: pass
m2.to_csv(OUT/"model2_reserve_features.csv", index=False)
try: m3.to_parquet(OUT/"model3_anomaly_features.parquet", index=False)
except Exception: pass
m3.to_csv(OUT/"model3_anomaly_features.csv", index=False)

print(f"  Model 1 features: {m1.shape}")
print(f"  Model 2 features: {m2.shape}")
print(f"  Model 3 features: {m3.shape}")
print(f"  OUTPUT: {OUT}")
print("="*64)
