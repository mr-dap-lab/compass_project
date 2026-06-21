"""
COMPASS — Preview Synthetic Data Generator
Week 1 Proof-of-Concept

Generates a minimal sample (5 associations) to demonstrate that the CDES
schema, ID format conventions, and entity relationships are internally
consistent and ready for full-scale generation in Week 2.

Output: CSV files written to data/sample/

Author: Diego Avella
"""

import json
import csv
import os
import random
import string
from datetime import date, timedelta
from pathlib import Path

# Reproducibility — same seed as the Week 2 full generator will use
random.seed(42)

# Paths
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)
SPEC = json.load(open(BASE / "data" / "synthetic_dataset_spec.json"))


# ── ID helpers (match CDES patterns in JSON Schema) ──────────────────────────

def rand_alphanum(n):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))

def make_caid(state, idx):
    return f"CA-{state}-{idx:08d}"

def make_unit_id():
    return f"U-{rand_alphanum(12)}"

def make_fee_id():
    return f"FS-{rand_alphanum(12)}"

def make_ledger_id():
    return f"LE-{rand_alphanum(16)}"

def make_delinq_id():
    return f"DL-{rand_alphanum(16)}"

def make_reserve_id():
    return f"RF-{rand_alphanum(12)}"


# ── Generation ───────────────────────────────────────────────────────────────

def generate_preview():
    associations = []
    units = []
    fee_schedules = []
    payment_ledger = []
    delinquency_records = []
    reserve_funds = []

    states = ["FL", "CA", "TX", "GA", "NV"]
    types = ["CONDO", "HOA", "HOA", "PUD", "MIXED"]
    coastal = [True, True, False, False, False]
    units_count = [120, 240, 80, 180, 60]
    years_built = [1985, 2005, 2015, 1998, 1978]

    for i in range(5):
        caid = make_caid(states[i], i + 1)
        associations.append({
            "caid":               caid,
            "legal_name":         f"Preview Association {i+1}",
            "association_type":   types[i],
            "state":              states[i],
            "city":               "Sample City",
            "zip":                "00000",
            "registration_date":  "2024-01-15",
            "status":             "ACTIVE",
            "total_units":        units_count[i],
            "year_built":         years_built[i],
            "building_count":     1 + (i % 3),
            "coastal_flag":       coastal[i]
        })

        # Generate first 3 units per association (preview only — not all units)
        for u in range(3):
            unit_id = make_unit_id()
            units.append({
                "unit_id":            unit_id,
                "caid":               caid,
                "unit_number":        f"{100 + u}",
                "unit_type":          "CONDO_UNIT" if types[i] == "CONDO" else "SINGLE_FAMILY",
                "occupancy_status":   random.choice(["OWNER_OCCUPIED", "OWNER_OCCUPIED", "RENTED"])
            })

        # One fee schedule per association
        type_fee_map = {"HOA": 285, "CONDO": 575, "PUD": 220, "MIXED": 320}
        fee_schedules.append({
            "fee_schedule_id":          make_fee_id(),
            "caid":                     caid,
            "effective_date":           "2024-01-01",
            "monthly_assessment":       type_fee_map[types[i]],
            "frequency":                "MONTHLY",
            "late_fee_amount":          25,
            "late_fee_threshold_days":  15
        })

        # One reserve fund snapshot per association — varies by risk profile
        pct_funded = [72, 45, 22, 60, 18][i]
        target = 500_000 + 10_000 * units_count[i]
        reserve_funds.append({
            "reserve_id":            make_reserve_id(),
            "caid":                  caid,
            "snapshot_date":         "2024-12-31",
            "current_balance":       round(target * pct_funded / 100, 2),
            "fully_funded_target":   target,
            "percent_funded":        pct_funded,
            "annual_contribution":   round(target * 0.05, 2),
            "study_type":            "SIRS" if states[i] == "FL" and types[i] == "CONDO" else "STANDARD"
        })

        # A few payment ledger entries per association
        for ent in range(3):
            unit_id = units[-3 + ent]["unit_id"]
            payment_ledger.append({
                "ledger_entry_id":   make_ledger_id(),
                "caid":              caid,
                "unit_id":           unit_id,
                "transaction_date":  f"2024-{6 + ent:02d}-01",
                "transaction_type":  "ASSESSMENT_CHARGE",
                "amount":            -type_fee_map[types[i]],
                "balance_after":     -type_fee_map[types[i]] * (ent + 1),
                "payment_method":    None
            })

        # One delinquency record per association
        for u_idx in range(3):
            unit_id = units[-3 + u_idx]["unit_id"]
            days = [0, 45, 95][u_idx]
            bucket = ["CURRENT", "DAYS_31_60", "DAYS_91_120"][u_idx]
            delinquency_records.append({
                "delinquency_id":     make_delinq_id(),
                "caid":               caid,
                "snapshot_date":      "2024-12-31",
                "unit_id":            unit_id,
                "days_delinquent":    days,
                "amount_owed":        type_fee_map[types[i]] * (days // 30 + 1) if days > 0 else 0,
                "aging_bucket":       bucket,
                "lien_filed":         days >= 90,
                "attorney_referred":  False
            })

    return {
        "associations":         associations,
        "units":                units,
        "fee_schedules":        fee_schedules,
        "payment_ledger":       payment_ledger,
        "delinquency_records":  delinquency_records,
        "reserve_funds":        reserve_funds
    }


def write_csv(name, rows):
    if not rows:
        return
    path = OUT / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    data = generate_preview()
    print("=" * 64)
    print("COMPASS Preview Synthetic Data Generator — Week 1")
    print("=" * 64)
    total = 0
    for name, rows in data.items():
        path = write_csv(name, rows)
        print(f"  {name:25s} → {len(rows):4d} rows → {path.name}")
        total += len(rows)
    print("-" * 64)
    print(f"  TOTAL ROWS WRITTEN: {total}")
    print(f"  OUTPUT DIRECTORY:   {OUT}")
    print("=" * 64)
    print("\nSummary by entity:")
    for caid in [a["caid"] for a in data["associations"]]:
        units_n = len([u for u in data["units"] if u["caid"] == caid])
        reserve = next(r for r in data["reserve_funds"] if r["caid"] == caid)
        print(f"  {caid}: {units_n} units, reserve funded at {reserve['percent_funded']:>3}%")


if __name__ == "__main__":
    main()
