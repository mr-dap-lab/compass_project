-- COMPASS Data Exchange Standard (CDES) v0.1.0 — PostgreSQL DDL
-- Capstone scope: Registry, Financial, Compliance domains + audit log.
-- Author: Diego Avella

CREATE TABLE associations (
    caid TEXT PRIMARY KEY,
    legal_name TEXT NOT NULL,
    association_type TEXT NOT NULL,
    state TEXT NOT NULL,
    city TEXT NOT NULL,
    zip TEXT NOT NULL,
    registration_date DATE NOT NULL,
    status TEXT NOT NULL,
    total_units INTEGER NOT NULL,
    year_built INTEGER,
    building_count INTEGER NOT NULL,
    coastal_flag BOOLEAN NOT NULL,
    CHECK (total_units >= 0)
);

CREATE TABLE units (
    unit_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    unit_number TEXT NOT NULL,
    unit_type TEXT NOT NULL,
    square_footage INTEGER,
    bedrooms INTEGER NOT NULL,
    occupancy_status TEXT NOT NULL,
    parcel_id TEXT,
    FOREIGN KEY (caid) REFERENCES associations(caid)
);

CREATE INDEX idx_units_caid ON units(caid);

CREATE TABLE fee_schedules (
    fee_schedule_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    effective_date DATE NOT NULL,
    expiration_date DATE,
    monthly_assessment NUMERIC(14,2) NOT NULL,
    frequency TEXT NOT NULL,
    late_fee_amount NUMERIC(14,2) NOT NULL,
    late_fee_threshold_days INTEGER NOT NULL,
    FOREIGN KEY (caid) REFERENCES associations(caid)
);

CREATE INDEX idx_fee_schedules_caid ON fee_schedules(caid);

CREATE TABLE payment_ledger (
    ledger_entry_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    unit_id TEXT,
    transaction_date DATE NOT NULL,
    transaction_type TEXT NOT NULL,
    amount NUMERIC(14,2) NOT NULL,
    balance_after NUMERIC(14,2) NOT NULL,
    payment_method TEXT,
    FOREIGN KEY (caid) REFERENCES associations(caid),
    FOREIGN KEY (unit_id) REFERENCES units(unit_id)
);

CREATE INDEX idx_payment_ledger_caid ON payment_ledger(caid);

CREATE TABLE delinquency_records (
    delinquency_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    unit_id TEXT NOT NULL,
    days_delinquent INTEGER NOT NULL,
    amount_owed NUMERIC(14,2) NOT NULL,
    aging_bucket TEXT NOT NULL,
    lien_filed BOOLEAN NOT NULL,
    attorney_referred BOOLEAN NOT NULL,
    FOREIGN KEY (caid) REFERENCES associations(caid),
    FOREIGN KEY (unit_id) REFERENCES units(unit_id)
);

CREATE INDEX idx_delinquency_records_caid ON delinquency_records(caid);

CREATE TABLE reserve_funds (
    reserve_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    current_balance NUMERIC(14,2) NOT NULL,
    fully_funded_target NUMERIC(14,2) NOT NULL,
    percent_funded NUMERIC(14,2) NOT NULL,
    annual_contribution NUMERIC(14,2) NOT NULL,
    last_study_date DATE,
    study_type TEXT NOT NULL,
    FOREIGN KEY (caid) REFERENCES associations(caid),
    CHECK (percent_funded >= 0)
);

CREATE INDEX idx_reserve_funds_caid ON reserve_funds(caid);

CREATE TABLE work_orders (
    work_order_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    unit_id TEXT,
    created_date DATE NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    estimated_cost NUMERIC(14,2) NOT NULL,
    actual_cost NUMERIC(14,2),
    completion_date DATE,
    deferred_days INTEGER NOT NULL,
    FOREIGN KEY (caid) REFERENCES associations(caid),
    FOREIGN KEY (unit_id) REFERENCES units(unit_id)
);

CREATE INDEX idx_work_orders_caid ON work_orders(caid);

CREATE TABLE violations (
    violation_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    issue_date DATE NOT NULL,
    violation_category TEXT NOT NULL,
    cure_deadline DATE NOT NULL,
    fine_amount NUMERIC(14,2) NOT NULL,
    status TEXT NOT NULL,
    resolution_date DATE,
    FOREIGN KEY (caid) REFERENCES associations(caid),
    FOREIGN KEY (unit_id) REFERENCES units(unit_id)
);

CREATE INDEX idx_violations_caid ON violations(caid);

CREATE TABLE sirs_filings (
    sirs_id TEXT PRIMARY KEY,
    caid TEXT NOT NULL,
    filing_date DATE NOT NULL,
    study_engineer TEXT NOT NULL,
    engineer_license TEXT NOT NULL,
    structural_findings TEXT NOT NULL,
    estimated_repair_cost NUMERIC(14,2) NOT NULL,
    next_due_date DATE NOT NULL,
    compliance_status TEXT NOT NULL,
    FOREIGN KEY (caid) REFERENCES associations(caid)
);

CREATE INDEX idx_sirs_filings_caid ON sirs_filings(caid);

-- Immutable audit trail (FedRAMP High AU-2 / AU-3 / AU-12)
CREATE TABLE audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target_entity TEXT,
    target_id TEXT,
    detail JSONB
);

-- Analytical views consumed by the predictive models
CREATE VIEW v_delinquency_monthly AS
    SELECT caid, date_trunc('month', snapshot_date) AS month,
           count(*) FILTER (WHERE days_delinquent >= 61) AS serious_units,
           count(*) AS total_snapshots
    FROM delinquency_records GROUP BY caid, date_trunc('month', snapshot_date);

CREATE VIEW v_reserve_latest AS
    SELECT DISTINCT ON (caid) caid, snapshot_date, percent_funded, current_balance
    FROM reserve_funds ORDER BY caid, snapshot_date DESC;

CREATE VIEW v_open_workorders AS
    SELECT caid, count(*) AS open_count,
           sum(estimated_cost) AS open_cost,
           count(*) FILTER (WHERE priority IN ('EMERGENCY','HIGH') AND status='DEFERRED') AS critical_deferred
    FROM work_orders WHERE status IN ('OPEN','DEFERRED') GROUP BY caid;
