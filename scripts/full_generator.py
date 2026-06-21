"""
COMPASS — Full Synthetic Dataset Generator
Week 2 Deliverable

Generates the complete CDES-conformant synthetic dataset used to train and
validate the three predictive models. Risk-profile-driven: each association is
assigned a profile (HEALTHY / MODERATE / STRESSED) that governs all of its
downstream financial, maintenance, and compliance behavior, producing realistic
correlated risk patterns.

Calibrated against CAI 2024 and HUD benchmarks (see synthetic_dataset_spec.json).

Output: Parquet + CSV files in data/full/
Author: Diego Avella
"""

import json
import random
import string
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from faker import Faker
except ImportError:
    # ── Minimal built-in fallback (cosmetic fields only) ─────────────────
    # Faker is used solely for display names / cities / zips. It maintains
    # its OWN seeded RNG, fully independent of the `random` and `numpy`
    # streams that drive every numeric value in the dataset — so swapping
    # in this shim leaves all amounts, dates, targets, and the 19.0%
    # positive-class check byte-for-byte reproducible.
    import random as _rnd
    class Faker:  # noqa: N801 — mirrors the real class name
        _rng = _rnd.Random(0)
        _LAST = ["Alvarez","Brooks","Castillo","Delgado","Everett","Fuentes",
                 "Grayson","Herrera","Ibarra","Jenkins","Kowalski","Lozano",
                 "Mendez","Navarro","Ortega","Pruitt","Quintero","Ramsey",
                 "Santos","Trujillo","Underwood","Vasquez","Whitfield","Young"]
        _CITY = ["Lakeside","Fairview","Riverton","Oakdale","Summerfield",
                 "Crestwood","Maplewood","Brookhaven","Stonebridge","Palmetto",
                 "Cypress Grove","Harbor Springs","Eagle Point","Westhaven"]
        def __init__(self, locale="en_US"): pass
        @classmethod
        def seed(cls, s): cls._rng = _rnd.Random(s)
        def last_name(self): return self._rng.choice(self._LAST)
        def city(self): return self._rng.choice(self._CITY)
        def zipcode(self): return f"{self._rng.randint(10000, 99999)}"
        def name(self):
            return f"{self._rng.choice(['Alex','Jordan','Morgan','Casey','Riley','Taylor'])} {self._rng.choice(self._LAST)}"

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

BASE = Path(__file__).resolve().parent.parent
SPEC = json.load(open(BASE / "data" / "synthetic_dataset_spec.json"))
OUT = BASE / "data" / "full"
OUT.mkdir(parents=True, exist_ok=True)

# ── Period setup ─────────────────────────────────────────────────────────────
START = date(2024, 1, 1)
MONTHS = SPEC["scale"]["time_period_months"]  # 18
N_ASSOC = SPEC["scale"]["total_associations"]  # 300

def month_dates(start, n):
    out = []
    y, m = start.year, start.month
    for _ in range(n):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1; y += 1
    return out

PERIOD = month_dates(START, MONTHS)

# ── ID helpers ───────────────────────────────────────────────────────────────
def ra(n): return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))
def caid(state, i): return f"CA-{state}-{i:08d}"
def uid(): return f"U-{ra(12)}"
def feeid(): return f"FS-{ra(12)}"
def leid(): return f"LE-{ra(16)}"
def dlid(): return f"DL-{ra(16)}"
def rfid(): return f"RF-{ra(12)}"
def vlid(): return f"VL-{ra(12)}"
def woid(): return f"WO-{ra(12)}"
def sirsid(): return f"SIRS-{ra(10)}"

# ── Weighted choice ──────────────────────────────────────────────────────────
def wchoice(d):
    keys = list(d.keys()); weights = list(d.values())
    return random.choices(keys, weights=weights, k=1)[0]

# ── 1. ASSOCIATIONS ──────────────────────────────────────────────────────────
def gen_associations():
    rows = []
    states_dist = SPEC["geographic_distribution"]["states"]
    type_mix = SPEC["association_profile"]["type_mix"]
    up = SPEC["association_profile"]["units_per_assoc"]
    yb = SPEC["association_profile"]["year_built"]["ranges"]
    coastal_states = set(SPEC["association_profile"]["coastal_states_priority"])
    coastal_pct = SPEC["association_profile"]["coastal_flag_pct"]
    profiles = SPEC["delinquency_generation"]["association_risk_profiles"]

    state_counter = {}
    for i in range(1, N_ASSOC + 1):
        st = wchoice(states_dist)
        state_counter[st] = state_counter.get(st, 0) + 1
        atype = wchoice(type_mix)

        # Units: lognormal, clamped
        n_units = int(np.clip(np.random.lognormal(up["mu"], up["sigma"]), up["min"], up["max"]))

        # Year built: weighted range pick
        yr_range = random.choices(yb, weights=[r["weight"] for r in yb], k=1)[0]
        year_built = random.randint(yr_range["min"], yr_range["max"])

        # Coastal: higher prob in coastal states
        base_coastal = coastal_pct * (2.2 if st in coastal_states else 0.3)
        coastal = random.random() < min(base_coastal, 0.85)

        # Risk profile assignment
        profile = random.choices(
            list(profiles.keys()),
            weights=[profiles[p]["share"] for p in profiles], k=1)[0]

        # year_built missing
        if random.random() < SPEC["data_quality_targets"]["completeness"]["associations.year_built_missing_pct"]:
            year_built_out = None
        else:
            year_built_out = year_built

        rows.append({
            "caid": caid(st, i),
            "legal_name": f"{fake.last_name()} {random.choice(['Community','Villas','Estates','Gardens','Pointe','Towers','Commons','Park','Place','Ridge'])} {random.choice(['HOA','Condominium Association','Owners Association','Community Association'])}",
            "association_type": atype,
            "state": st,
            "city": fake.city(),
            "zip": fake.zipcode(),
            "registration_date": (START - timedelta(days=random.randint(30, 720))).isoformat(),
            "status": "ACTIVE",
            "total_units": n_units,
            "year_built": year_built_out,
            "building_count": 1 if atype in ("HOA","PUD") else random.randint(1,6),
            "coastal_flag": coastal,
            # internal (not in CDES output, used by generator):
            "_profile": profile,
            "_year_built_actual": year_built,
        })
    return pd.DataFrame(rows)

# ── 2. UNITS ─────────────────────────────────────────────────────────────────
def gen_units(assoc):
    rows = []
    occ = SPEC["occupancy_distribution"]
    occ_clean = {k:v for k,v in occ.items() if isinstance(v,(int,float))}
    sqft_missing = SPEC["data_quality_targets"]["completeness"]["units.square_footage_missing_pct"]
    parcel_missing = SPEC["data_quality_targets"]["completeness"]["units.parcel_id_missing_pct"]

    for _, a in assoc.iterrows():
        # cap units generated for tractability while preserving total_units field
        gen_count = min(a["total_units"], 60)
        for u in range(gen_count):
            atype = a["association_type"]
            utype = ("CONDO_UNIT" if atype == "CONDO"
                     else "TOWNHOME" if atype == "PUD"
                     else random.choice(["SINGLE_FAMILY","TOWNHOME"]))
            rows.append({
                "unit_id": uid(),
                "caid": a["caid"],
                "unit_number": f"{random.randint(1,40)}{random.choice('ABCDEFGH')}" if atype=="CONDO" else str(100+u),
                "unit_type": utype,
                "square_footage": None if random.random()<sqft_missing else random.randint(650, 3200),
                "bedrooms": random.choices([1,2,3,4,5],[0.1,0.35,0.35,0.15,0.05])[0],
                "occupancy_status": wchoice(occ_clean),
                "parcel_id": None if random.random()<parcel_missing else f"{random.randint(10,99)}-{random.randint(1000,9999)}-{random.randint(100,999)}",
            })
    return pd.DataFrame(rows)

# ── 3. FEE SCHEDULES ─────────────────────────────────────────────────────────
def gen_fees(assoc):
    rows = []
    fee_cfg = SPEC["fee_schedule"]["monthly_assessment"]
    for _, a in assoc.iterrows():
        cfg = fee_cfg[a["association_type"]]
        amt = float(np.clip(np.random.normal(cfg["mean"], cfg["std"]), cfg["min"], cfg["max"]))
        rows.append({
            "fee_schedule_id": feeid(),
            "caid": a["caid"],
            "effective_date": START.isoformat(),
            "expiration_date": None,
            "monthly_assessment": round(amt, 2),
            "frequency": "MONTHLY",
            "late_fee_amount": SPEC["fee_schedule"]["late_fee_amount"],
            "late_fee_threshold_days": SPEC["fee_schedule"]["late_fee_threshold_days"],
        })
    return pd.DataFrame(rows)

# ── 4. PAYMENT LEDGER + DELINQUENCY (the core time-series) ───────────────────
def gen_ledger_and_delinquency(assoc, units, fees):
    ledger = []
    delinq = []
    profiles = SPEC["delinquency_generation"]["association_risk_profiles"]
    seasonal = SPEC["delinquency_generation"]["seasonal_multipliers"]
    recovery_p = SPEC["delinquency_generation"]["recovery_probability_per_month"]
    lien_days = SPEC["delinquency_generation"]["lien_threshold_days"]
    atty_days = SPEC["delinquency_generation"]["attorney_referral_days"]

    fee_map = dict(zip(fees["caid"], fees["monthly_assessment"]))
    units_by_assoc = units.groupby("caid")

    def aging_bucket(days):
        if days == 0: return "CURRENT"
        if days <= 30: return "DAYS_1_30"
        if days <= 60: return "DAYS_31_60"
        if days <= 90: return "DAYS_61_90"
        if days <= 120: return "DAYS_91_120"
        return "DAYS_OVER_120"

    for _, a in assoc.iterrows():
        cid = a["caid"]
        if cid not in units_by_assoc.groups:
            continue
        au = units_by_assoc.get_group(cid)
        monthly_fee = fee_map[cid]
        base_miss = profiles[a["_profile"]]["monthly_miss_pct"]

        # track per-unit delinquency day counters
        unit_days = {u: 0 for u in au["unit_id"]}
        unit_balance = {u: 0.0 for u in au["unit_id"]}

        for mi, mdate in enumerate(PERIOD):
            season = seasonal[str(mdate.month)]
            miss_p = min(base_miss * season, 0.95)

            for u in au["unit_id"]:
                # assessment charge
                unit_balance[u] += monthly_fee
                ledger.append({
                    "ledger_entry_id": leid(), "caid": cid, "unit_id": u,
                    "transaction_date": mdate.isoformat(),
                    "transaction_type": "ASSESSMENT_CHARGE",
                    "amount": -round(monthly_fee,2),
                    "balance_after": -round(unit_balance[u],2),
                    "payment_method": None,
                })

                # do they pay this month?
                currently_behind = unit_days[u] > 0
                pays = (random.random() > miss_p) if not currently_behind \
                       else (random.random() < recovery_p)

                if pays:
                    pay_amt = unit_balance[u]
                    if pay_amt > 0:
                        unit_balance[u] = 0.0
                        unit_days[u] = 0
                        ledger.append({
                            "ledger_entry_id": leid(), "caid": cid, "unit_id": u,
                            "transaction_date": (mdate+timedelta(days=random.randint(1,20))).isoformat(),
                            "transaction_type": "PAYMENT",
                            "amount": round(pay_amt,2),
                            "balance_after": 0.0,
                            "payment_method": random.choice(["ACH","CHECK","CARD"]),
                        })
                else:
                    unit_days[u] += 30
                    # late fee
                    if unit_days[u] >= SPEC["fee_schedule"]["late_fee_threshold_days"]:
                        unit_balance[u] += SPEC["fee_schedule"]["late_fee_amount"]

                # monthly delinquency snapshot (only if behind)
                if unit_days[u] > 0:
                    delinq.append({
                        "delinquency_id": dlid(), "caid": cid,
                        "snapshot_date": mdate.isoformat(), "unit_id": u,
                        "days_delinquent": unit_days[u],
                        "amount_owed": round(unit_balance[u],2),
                        "aging_bucket": aging_bucket(unit_days[u]),
                        "lien_filed": unit_days[u] >= lien_days,
                        "attorney_referred": unit_days[u] >= atty_days,
                    })
    return pd.DataFrame(ledger), pd.DataFrame(delinq)

# ── 5. RESERVE FUNDS (quarterly snapshots) ───────────────────────────────────
def gen_reserves(assoc):
    rows = []
    clusters = SPEC["reserve_fund_generation"]["percent_funded_distribution"]
    freq = SPEC["reserve_fund_generation"]["snapshot_frequency_months"]
    study_missing = SPEC["data_quality_targets"]["completeness"]["reserve_funds.last_study_date_missing_pct"]

    # map risk profile to reserve cluster tendency
    profile_to_cluster = {
        "HEALTHY":  "HEALTHY_CLUSTER",
        "MODERATE": "MODERATE_CLUSTER",
        "STRESSED": "CRITICAL_CLUSTER",
    }
    for _, a in assoc.iterrows():
        cl_name = profile_to_cluster[a["_profile"]]
        cl = clusters[cl_name]
        base_pct = float(np.clip(np.random.normal(cl["mean"], cl["std"]), cl["min"], cl["max"]))
        target = round(random.uniform(300_000, 5_000_000), 2)

        # trend: stressed associations decline over time
        trend = SPEC["reserve_fund_generation"].get(
            "quarterly_trend_by_profile",
            {"HEALTHY": 0.4, "MODERATE": -0.3, "STRESSED": -1.1})[a["_profile"]]

        for qi, mi in enumerate(range(0, MONTHS, freq)):
            mdate = PERIOD[mi]
            pct = float(np.clip(base_pct + trend*qi + np.random.normal(0,1.5), 0, 150))
            rows.append({
                "reserve_id": rfid(), "caid": a["caid"],
                "snapshot_date": mdate.isoformat(),
                "current_balance": round(target*pct/100, 2),
                "fully_funded_target": target,
                "percent_funded": round(pct, 2),
                "annual_contribution": round(target*random.uniform(0.03,0.08), 2),
                "last_study_date": None if random.random()<study_missing else (mdate - timedelta(days=random.randint(180, 1460))).isoformat(),
                "study_type": "SIRS" if (a["state"]=="FL" and a["association_type"]=="CONDO") else random.choice(["STANDARD","INFORMAL"]),
            })
    return pd.DataFrame(rows)

# ── 6. WORK ORDERS ───────────────────────────────────────────────────────────
def gen_work_orders(assoc):
    rows = []
    wo_cfg = SPEC["work_orders_generation"]
    cat_dist = wo_cfg["category_distribution"]
    for _, a in assoc.iterrows():
        rate = wo_cfg["orders_per_assoc_per_month"][a["_profile"]]
        defer_p = wo_cfg["deferral_probability"][a["_profile"]]
        for mdate in PERIOD:
            n = max(0, int(np.random.normal(rate["mean"], rate["std"])))
            for _ in range(n):
                cat = wchoice(cat_dist)
                deferred = random.random() < defer_p
                created = mdate + timedelta(days=random.randint(0,27))
                if deferred:
                    status = random.choice(["DEFERRED","OPEN"])
                    ddays = int(np.random.lognormal(wo_cfg["deferred_days_lognormal"]["mu"], wo_cfg["deferred_days_lognormal"]["sigma"]))
                    completion = None
                    actual = None
                else:
                    status = "COMPLETED"
                    ddays = random.randint(1, 21)
                    completion = (created + timedelta(days=ddays)).isoformat()
                    actual = round(random.uniform(150, 12000), 2)
                est = round(random.uniform(200, 15000), 2)
                rows.append({
                    "work_order_id": woid(), "caid": a["caid"], "unit_id": None,
                    "created_date": created.isoformat(),
                    "category": cat,
                    "priority": random.choices(["EMERGENCY","HIGH","MEDIUM","LOW"],[0.05,0.20,0.45,0.30])[0],
                    "status": status,
                    "estimated_cost": est,
                    "actual_cost": actual,
                    "completion_date": completion,
                    "deferred_days": ddays,
                })
    return pd.DataFrame(rows)

# ── 7. VIOLATIONS ────────────────────────────────────────────────────────────
def gen_violations(assoc, units):
    rows = []
    v_cfg = SPEC["violations_generation"]
    cat_dist = v_cfg["category_distribution"]
    units_by_assoc = units.groupby("caid")
    for _, a in assoc.iterrows():
        if a["caid"] not in units_by_assoc.groups: continue
        au = units_by_assoc.get_group(a["caid"])
        # annual rate scaled to 18 months, scaled by profile
        prof_mult = {"HEALTHY":0.6, "MODERATE":1.0, "STRESSED":1.8}[a["_profile"]]
        n_viol = int(len(au) * v_cfg["annual_violations_per_unit"]["mean"] * 1.5 * prof_mult)
        for _ in range(n_viol):
            u = au.sample(1).iloc[0]["unit_id"]
            issue = START + timedelta(days=random.randint(0, MONTHS*30))
            cured = random.random() < v_cfg["cure_rate"]
            rows.append({
                "violation_id": vlid(), "caid": a["caid"], "unit_id": u,
                "issue_date": issue.isoformat(),
                "violation_category": wchoice(cat_dist),
                "cure_deadline": (issue+timedelta(days=30)).isoformat(),
                "fine_amount": round(np.random.lognormal(v_cfg["fine_amount_lognormal"]["mu"], v_cfg["fine_amount_lognormal"]["sigma"]),2),
                "status": random.choice(["CURED","CURED","DISMISSED"]) if cured else random.choice(["OPEN","ESCALATED","FINED","HEARING_SCHEDULED"]),
                "resolution_date": (issue+timedelta(days=random.randint(10,60))).isoformat() if cured else None,
            })
    return pd.DataFrame(rows)

# ── 8. SIRS FILINGS ──────────────────────────────────────────────────────────
def gen_sirs(assoc):
    rows = []
    s_cfg = SPEC["sirs_filings_generation"]
    find_dist = s_cfg["findings_distribution"]
    for _, a in assoc.iterrows():
        applies = (a["state"] in s_cfg["applicability"]["states_required"]
                   and a["association_type"] in s_cfg["applicability"]["association_types_required"]
                   and a["_year_built_actual"] < s_cfg["applicability"]["min_year_built_for_exempt"])
        if not applies:
            continue
        if random.random() > 0.70:  # only ~70% of applicable have filed
            continue
        filing = START - timedelta(days=random.randint(0, 900))
        overdue = random.random() < s_cfg["overdue_rate"]
        rows.append({
            "sirs_id": sirsid(), "caid": a["caid"],
            "filing_date": filing.isoformat(),
            "study_engineer": fake.name() + ", P.E.",
            "engineer_license": f"PE-{random.randint(10000,99999)}",
            "structural_findings": wchoice(find_dist),
            "estimated_repair_cost": round(random.uniform(0, 2_500_000),2),
            "next_due_date": (filing+timedelta(days=3650)).isoformat(),
            "compliance_status": "OVERDUE" if overdue else "COMPLIANT",
        })
    return pd.DataFrame(rows)

# ── FRAUD INJECTION ──────────────────────────────────────────────────────────
def inject_fraud(assoc, ledger, reserves):
    """Embed fraud patterns into specific associations for Model 3 evaluation."""
    fraud_log = []
    healthy = assoc[assoc["_profile"]=="HEALTHY"]["caid"].tolist()
    random.shuffle(healthy)
    picks = healthy[:6]

    # Pattern 1: vendor payment concentration (2 assoc) — simulated via ADJUSTMENT entries
    for c in picks[0:2]:
        for _ in range(40):
            md = random.choice(PERIOD)
            ledger = pd.concat([ledger, pd.DataFrame([{
                "ledger_entry_id": leid(), "caid": c, "unit_id": None,
                "transaction_date": md.isoformat(), "transaction_type": "ADJUSTMENT",
                "amount": -round(random.uniform(8000,25000),2), "balance_after": 0.0,
                "payment_method": "WIRE",
            }])], ignore_index=True)
        fraud_log.append({"caid": c, "pattern": "vendor_payment_concentration"})

    # Pattern 2: collection gap (2 assoc) — suppress payments to create gap
    for c in picks[2:4]:
        # Remove ~40% of payment records to simulate phantom-occupancy / misstatement
        pay_mask = (ledger["caid"]==c) & (ledger["transaction_type"]=="PAYMENT")
        pay_idx = ledger[pay_mask].index.tolist()
        random.shuffle(pay_idx)
        drop = pay_idx[:int(len(pay_idx)*0.40)]
        ledger = ledger.drop(index=drop).reset_index(drop=True)
        fraud_log.append({"caid": c, "pattern": "collection_gap_anomaly"})

    # Pattern 3: reserve balance irregularity (2 assoc) — inject a sharp unexplained drop
    for c in picks[4:6]:
        mask = reserves["caid"]==c
        idx = reserves[mask].index
        if len(idx) >= 2:
            drop_idx = idx[len(idx)//2:]
            reserves.loc[drop_idx, "percent_funded"] = (reserves.loc[drop_idx, "percent_funded"] * 0.25).round(2)
            reserves.loc[drop_idx, "current_balance"] = (reserves.loc[drop_idx, "fully_funded_target"] * reserves.loc[drop_idx, "percent_funded"]/100).round(2)
        fraud_log.append({"caid": c, "pattern": "reserve_balance_irregularity"})

    return ledger, reserves, pd.DataFrame(fraud_log)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("="*64); print("COMPASS Full Synthetic Dataset Generator — Week 2"); print("="*64)

    print("  [1/8] Generating associations...")
    assoc = gen_associations()
    print(f"        {len(assoc)} associations")

    print("  [2/8] Generating units...")
    units = gen_units(assoc)
    print(f"        {len(units)} units")

    print("  [3/8] Generating fee schedules...")
    fees = gen_fees(assoc)

    print("  [4/8] Generating payment ledger + delinquency (time-series)...")
    ledger, delinq = gen_ledger_and_delinquency(assoc, units, fees)
    print(f"        {len(ledger)} ledger entries, {len(delinq)} delinquency snapshots")

    print("  [5/8] Generating reserve funds...")
    reserves = gen_reserves(assoc)

    print("  [6/8] Generating work orders...")
    work_orders = gen_work_orders(assoc)
    print(f"        {len(work_orders)} work orders")

    print("  [7/8] Generating violations + SIRS...")
    violations = gen_violations(assoc, units)
    sirs = gen_sirs(assoc)
    print(f"        {len(violations)} violations, {len(sirs)} SIRS filings")

    print("  [8/8] Injecting fraud patterns...")
    ledger, reserves, fraud_log = inject_fraud(assoc, ledger, reserves)
    print(f"        {len(fraud_log)} fraud cases embedded")

    # Strip internal columns from associations before export
    assoc_export = assoc.drop(columns=["_profile","_year_built_actual"])
    # Keep a private profile key file for later evaluation (not part of CDES)
    profile_key = assoc[["caid","_profile"]].rename(columns={"_profile":"risk_profile"})

    tables = {
        "associations": assoc_export, "units": units, "fee_schedules": fees,
        "payment_ledger": ledger, "delinquency_records": delinq,
        "reserve_funds": reserves, "work_orders": work_orders,
        "violations": violations, "sirs_filings": sirs,
    }

    print("-"*64); print("  Writing files...")
    total = 0
    for name, df in tables.items():
        try:
            df.to_parquet(OUT / f"{name}.parquet", index=False)
        except Exception:
            print(f"    (parquet engine unavailable -> CSV only for {name})")
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"    {name:22s} {len(df):>8,} rows")
        total += len(df)

    # write keys for evaluation
    profile_key.to_csv(OUT / "_risk_profile_key.csv", index=False)
    fraud_log.to_csv(OUT / "_fraud_key.csv", index=False)

    print("-"*64)
    print(f"  TOTAL ROWS: {total:,}")
    print(f"  OUTPUT: {OUT}")
    print("="*64)

if __name__ == "__main__":
    main()
