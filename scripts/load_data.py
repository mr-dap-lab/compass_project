#!/usr/bin/env python3
"""
COMPASS — PostgreSQL data loader (psycopg2)

Loads the synthetic CSV files into the CDES schema over a normal database
connection. No psql, no \\copy, no path-relative-to-psql surprises. Handles the
pandas float-to-int columns (year_built, square_footage) automatically.

PREREQUISITES
  1. PostgreSQL running, with the database and schema already created:
        psql -U postgres -f db/00_create_database.sql
        psql -U postgres -d compass -f db/01_schema.sql
  2. pip install psycopg2-binary pandas
  3. The CSV files (associations.csv, units.csv, ...) in one folder.

USAGE
  python db/load_data.py --csv-dir "/path/to/your/csv/folder"

  Common options (all optional; sensible defaults shown):
    --host localhost  --port 5432  --dbname compass  --user postgres
    --password ****** (or set PGPASSWORD env var, or rely on .pgpass)
    --truncate / --no-truncate   (default: --truncate, clears tables first)

EXAMPLES
  # macOS / Linux
  python3 db/load_data.py --csv-dir ~/Downloads/compass_project/data/full --user postgres

  # Windows
  python db\\load_data.py --csv-dir "C:\\Users\\me\\compass_project\\data\\full" --user postgres

Author: Diego Avella
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    sys.exit("psycopg2 not installed. Run:  pip install psycopg2-binary")

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed. Run:  pip install pandas")


# Tables in parent-first load order, with their exact column lists (must match the schema).
TABLES = {
    "associations": ["caid", "legal_name", "association_type", "state", "city", "zip",
                     "registration_date", "status", "total_units", "year_built",
                     "building_count", "coastal_flag"],
    "units": ["unit_id", "caid", "unit_number", "unit_type", "square_footage",
              "bedrooms", "occupancy_status", "parcel_id"],
    "fee_schedules": ["fee_schedule_id", "caid", "effective_date", "expiration_date",
                      "monthly_assessment", "frequency", "late_fee_amount",
                      "late_fee_threshold_days"],
    "payment_ledger": ["ledger_entry_id", "caid", "unit_id", "transaction_date",
                       "transaction_type", "amount", "balance_after", "payment_method"],
    "delinquency_records": ["delinquency_id", "caid", "snapshot_date", "unit_id",
                            "days_delinquent", "amount_owed", "aging_bucket",
                            "lien_filed", "attorney_referred"],
    "reserve_funds": ["reserve_id", "caid", "snapshot_date", "current_balance",
                      "fully_funded_target", "percent_funded", "annual_contribution",
                      "last_study_date", "study_type"],
    "work_orders": ["work_order_id", "caid", "unit_id", "created_date", "category",
                    "priority", "status", "estimated_cost", "actual_cost",
                    "completion_date", "deferred_days"],
    "violations": ["violation_id", "caid", "unit_id", "issue_date", "violation_category",
                   "cure_deadline", "fine_amount", "status", "resolution_date"],
    "sirs_filings": ["sirs_id", "caid", "filing_date", "study_engineer",
                     "engineer_license", "structural_findings", "estimated_repair_cost",
                     "next_due_date", "compliance_status"],
}
LOAD_ORDER = list(TABLES.keys())

# Nullable integer columns that pandas exports in float form (e.g. 1987.0).
# We round and cast them to Python int (or None) before inserting.
FLOAT_INT_COLS = {
    "associations": ["year_built", "building_count", "total_units"],
    "units": ["square_footage", "bedrooms"],
    "fee_schedules": ["late_fee_threshold_days"],
    "delinquency_records": ["days_delinquent"],
    "work_orders": ["deferred_days"],
}


def clean_frame(table, df):
    """Coerce types so psycopg2 inserts cleanly: NaN -> None, float-ints -> ints."""
    # Make sure every expected column exists and is ordered correctly
    cols = TABLES[table]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table}.csv is missing columns: {missing}")
    df = df[cols].copy()

    # Round/cast float-encoded integer columns
    for c in FLOAT_INT_COLS.get(table, []):
        if c in df.columns:
            df[c] = df[c].apply(lambda v: None if pd.isna(v) else int(round(float(v))))

    # Convert all remaining NaN to None (NULL) across the frame
    df = df.astype(object).where(pd.notnull(df), None)
    return df


def load_table(cur, table, csv_dir, truncate):
    path = Path(csv_dir) / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df = clean_frame(table, df)

    if truncate:
        # CASCADE on the first (parent) truncate handles children; safe to call per-table
        cur.execute(f"TRUNCATE {table} RESTART IDENTITY CASCADE;")

    cols = TABLES[table]
    collist = ", ".join(cols)
    rows = [tuple(r) for r in df.to_numpy()]
    if rows:
        sql = f"INSERT INTO {table} ({collist}) VALUES %s"
        execute_values(cur, sql, rows, page_size=5000)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Load COMPASS CSVs into PostgreSQL.")
    ap.add_argument("--csv-dir", required=True, help="Folder containing the *.csv files")
    ap.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    ap.add_argument("--port", default=os.environ.get("PGPORT", "5432"))
    ap.add_argument("--dbname", default=os.environ.get("PGDATABASE", "compass"))
    ap.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    ap.add_argument("--password", default=os.environ.get("PGPASSWORD"))
    ap.add_argument("--no-truncate", dest="truncate", action="store_false",
                    help="Do not clear tables before loading (default: clear them)")
    ap.set_defaults(truncate=True)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir).expanduser()
    if not csv_dir.is_dir():
        sys.exit(f"--csv-dir is not a folder: {csv_dir}")

    password = args.password
    if password is None:
        # Prompt only if not provided and not using .pgpass; empty string is allowed.
        try:
            password = getpass.getpass(f"Password for {args.user}@{args.host} (blank if none): ") or None
        except Exception:
            password = None

    print(f"Connecting to {args.dbname} at {args.host}:{args.port} as {args.user} ...")
    try:
        conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname,
                                user=args.user, password=password)
    except Exception as e:
        sys.exit(f"Connection failed: {e}")

    conn.autocommit = False
    total = 0
    try:
        with conn.cursor() as cur:
            # Defer FK checks within the transaction so order is forgiving
            cur.execute("SET CONSTRAINTS ALL DEFERRED;")
            for table in LOAD_ORDER:
                n = load_table(cur, table, csv_dir, args.truncate)
                total += n
                print(f"  loaded {table:22s} {n:>7,} rows")
        conn.commit()
        print(f"\nDone. Loaded {total:,} rows across {len(LOAD_ORDER)} tables.")
    except Exception as e:
        conn.rollback()
        sys.exit(f"\nLoad failed (rolled back): {e}")
    finally:
        # Verification counts
        try:
            with conn.cursor() as cur:
                print("\nRow counts in database:")
                for table in LOAD_ORDER:
                    cur.execute(f"SELECT count(*) FROM {table};")
                    print(f"  {table:22s} {cur.fetchone()[0]:>7,}")
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()
