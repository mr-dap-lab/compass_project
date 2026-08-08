# COMPASS: Community & Property Accountability System for States

**Capstone Project for Master of Science in Cybersecurity & Analytics**
**St. Thomas University · Summer 2026**

**Author:** Diego Avella
**Advisor:** Dr. Bernard Parenteau

---

## Project Summary

COMPASS is a proof-of-concept analytics and cybersecurity platform addressing a gap in United States housing policy: the absence of a national information system for monitoring, analyzing, and predicting risk across the ~365,000 community associations governing ~75.5 million residents and ~$12.2 trillion in property value.

This capstone delivers:

1. **CDES**: the COMPASS Data Exchange Standard: a published data schema (JSON Schema + PostgreSQL DDL) for community-association data interoperability. This is a first iteration open to scale.
2. **Three predictive models**: Delinquency Risk (binary classifier), Reserve Fund Failure (calibrated probability scorer), Financial Anomaly Detection (unsupervised Isolation Forest).
3. **Security architecture**: STRIDE threat model + FedRAMP High control mapping (NIST SP 800-53 Rev. 5) + Privacy Impact Assessment (Phase 4).
4. **Streamlit prototype dashboard**: end-to-end live demo (Phase 5). PoC in this repo: https://github.com/mr-dap-lab/compass-portal-defense

All data is **fully synthetic** (no real PII), generated with a fixed seed (42) and calibrated to CAI 2024 and HUD benchmarks.

---

## Repository Structure

```
compass/
├── README.md
├── HOW_TO_RUN_LOCALLY.md       #  Unix based systems, or windows based system running guide
├── requirements.txt            # Python libraries required for running
├── schema/
│   ├── cdes_v0.1.0.json              # CDES JSON Schema (draft-07)
│   └── cdes_v0.1.0_postgres.sql      # CDES PostgreSQL DDL (10 tables + audit log + 3 views)
├── data/
│   ├── synthetic_dataset_spec.json   # All generation parameters
│   ├── sample/                       # 5-association preview
│   ├── full/                         # Full generated dataset (CSV) + evaluation keys
│   └── features/                     # 3 model-ready feature tables
├── data/models/                      # Trained models (.pkl) + CV/OOF prediction CSVs
├── docs/                             # Generated figures (ERD, EDA, model results)
└── scripts/                          # 7 pipeline scripts (see run order below)
```

---

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 scripts/preview_generator.py     # 1  schema proof (5 assoc)
python3 scripts/render_erd.py            # 2  ERD figure
python3 scripts/full_generator.py        # 3  full dataset  (REQUIRED before running the feature engineering and then the models)
python3 scripts/feature_engineering.py   # 4  features  -> prints "57 positive (19.0%)"
python3 scripts/eda_summary.py           # 5  EDA figure
python3 scripts/train_model1.py          # 6  Model 1 + figure
python3 scripts/train_model2.py          # 7  Model 2 + figure
python3 scripts/train_model3.py          # 8  Model 3 (anomaly) + figure
python3 scripts/generate_security_diagram.py  # 9  Security architecture diagram
```

Go to `HOW_TO_RUN_LOCALLY.md` for a detailed, step-by-step walkthrough

---

## Verified results (Phases 1–3)

| Component | Result |
|---|---|
| Dataset | 300 associations, 12,689 units, ~484K rows; 6 embedded fraud cases |
| Model 1: Delinquency | Logistic Regression champion: F1 = 0.765, ROC-AUC = 0.928 (both targets met); oracle ceiling 0.811 (94.4%) |
| Model 2: Reserve failure | Random Forest (Platt-calibrated): Brier = 0.0119, ROC-AUC = 0.970 (both targets met) |
| Model 3: Anomaly Detection | Isolation Forest: Precision@6 = 0.833 (target ≥ 0.70 met); ROC-AUC = 0.999 |
| Reproducibility check | Feature pipeline prints **57 positive (19.0%)** for Model 1 |

---

## Citation

> Avella, D. (2026). *COMPASS: Community & Property Accountability System for States: A predictive analytics and cybersecurity platform for U.S. community association oversight* [Master's capstone project]. St. Thomas University.
