# COMPASS Project: How to Run Locally (Linux, macOS and Windows)

This guide walks you through reproducing every output of the COMPASS capstone, in
order, on **either macOS or Windows**. Follow the steps top to bottom. Total time:
about 5–8 minutes (most of it is the data-generation step).

Everything is reproducible because all randomness uses a fixed seed (42) so you will
get the exact same numbers shown in the progress reports, on both operating system.

> **Mac vs. Windows:** the only differences are how you open a terminal, how you
> activate the virtual environment, and that Windows uses `python` where macOS uses
> `python3`. Every step below shows **both**. Pick the line for your OS.

---

## STEP 0 — One-time setup (only do this once)

### 0.1 Check that you have Python 3.10+

**macOS** — open the **Terminal** app (Cmd+Space, type "Terminal", Enter):
```bash
python3 --version
```

**Windows** — open **PowerShell** (Start menu, type "PowerShell", Enter):
```powershell
python --version
```

You should see `Python 3.10.x` or higher. If not, install from
https://www.python.org/downloads/.
**Windows users:** on the first screen of the installer, check the box
**"Add python.exe to PATH"** before clicking Install — this avoids most problems.

### 0.2 Unzip the project and navigate into it

Unzip `COMPASS_Project_Phases1-4.zip`. You'll get a `compass_project` folder.

**macOS:** type `cd ` (with a space), drag the unzipped folder from Finder into
Terminal, press Enter. Example:
```bash
cd ~/Downloads/compass_project
```

**Windows:** in File Explorer, open the unzipped folder, click the address bar,
copy the path, then in PowerShell:
```powershell
cd "C:\Users\YourName\Downloads\compass_project"
```

Confirm you're in the right place — you should see `schema`, `scripts`, `data`,
`docs`, `README.md`, `requirements.txt`:
```bash
ls          # works on both macOS and Windows PowerShell
```

### 0.3 Create a virtual environment (isolated sandbox for this project's libraries)

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

After activating, your prompt starts with `(venv)`. **You must run the activate
command every time you open a new terminal to work on this project.**

> **Windows note:** if you see "running scripts is disabled on this system" when
> activating, run this once, then try the activate line again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> (Alternatively use `venv\Scripts\activate.bat` from Command Prompt instead of
> PowerShell.)

### 0.4 (Windows only) Enable UTF-8 output

The scripts print symbols like ✓ and ≥. Some Windows terminals use an older
encoding that chokes on these. Set this once per terminal session (harmless on Mac):
```powershell
$env:PYTHONUTF8 = "1"
```
To make it permanent, you can instead set `PYTHONUTF8=1` in your system environment
variables. (macOS needs nothing here.)

### 0.5 Install the required libraries

With the environment active (you see `(venv)`):

**macOS:**
```bash
pip install -r requirements.txt
```
**Windows:**
```powershell
pip install -r requirements.txt
```

This downloads pandas, numpy, Faker, matplotlib, pyarrow, scikit-learn, and shap.
Takes a minute or two the first time. You only do this once.

---

## The pipeline — run these in order

After Step 0, run the scripts below in sequence. **Use `python3` on macOS and
`python` on Windows** (shown together as `python3`/`python`).

### STEP 1 — Preview sample (Phase 1 proof-of-concept)
```bash
python3 scripts/preview_generator.py        # macOS
python  scripts/preview_generator.py        # Windows
```
**Expected:** 6 files written totaling 60 rows; 5 associations listed
(CA-FL-00000001 funded at 72%, etc.). Output lands in `data/sample/`.

### STEP 2 — ERD diagram (Phase 1)
```bash
python3 scripts/render_erd.py               # macOS
python  scripts/render_erd.py               # Windows
```
**Expected:** `ERD saved to: .../docs/cdes_erd.png`.

### STEP 3 — Full synthetic dataset (Phase 2)  ·  2–4 min
```bash
python3 scripts/full_generator.py           # macOS
python  scripts/full_generator.py           # Windows
```
**Expected:** progress log ending in ~484,000 rows across 9 tables.
Always the same counts: **300 associations, 12,689 units** (seed-locked).

### STEP 4 — Feature engineering (Phase 2)
```bash
python3 scripts/feature_engineering.py      # macOS
python  scripts/feature_engineering.py      # Windows
```
**Expected (the reproducibility check):** Model 1 prints
**`57 positive (19.0%)`** and Model 2 prints **`56 positive (18.7%)`**.

### STEP 5 — EDA visualization (Phase 2)
```bash
python3 scripts/eda_summary.py              # macOS
python  scripts/eda_summary.py              # Windows
```
**Expected:** `EDA summary saved to .../docs/eda_summary.png` (nine-panel figure).

### STEP 6 — Model 1: delinquency risk (Phase 3)  ·  1–2 min
```bash
python3 scripts/train_model1.py             # macOS
python  scripts/train_model1.py             # Windows
```
**Expected:** `CHAMPION: Logistic Regression | OOF F1 = 0.765 @ thr 0.56 |
ROC-AUC = 0.928`, oracle ceiling ≈ 0.811, both targets MET, and
`docs/model1_results.png`.

### STEP 7 — Model 2: reserve failure probability (Phase 3)
```bash
python3 scripts/train_model2.py             # macOS
python  scripts/train_model2.py             # Windows
```
**Expected:** champion Random Forest, Brier = 0.0119, ROC-AUC = 0.970, both targets
MET, and `docs/model2_results.png`.

### STEP 8 — Model 3: financial anomaly detection (Phase 4)
```bash
python3 scripts/train_model3.py             # macOS
python  scripts/train_model3.py             # Windows
```
**Expected:** `Precision@6 = 5/6 = 0.833`, target MET, ROC-AUC = 0.999, and
`docs/model3_results.png`.

### STEP 9 — Security architecture diagram (Phase 4)
```bash
python3 scripts/generate_security_diagram.py    # macOS
python  scripts/generate_security_diagram.py    # Windows
```
**Expected:** `Security architecture diagram saved: .../docs/security_architecture.png`.

---

## The correct order (copy-paste summary)

**macOS:**
```bash
source venv/bin/activate
python3 scripts/preview_generator.py
python3 scripts/render_erd.py
python3 scripts/full_generator.py
python3 scripts/feature_engineering.py
python3 scripts/eda_summary.py
python3 scripts/train_model1.py
python3 scripts/train_model2.py
python3 scripts/train_model3.py
python3 scripts/generate_security_diagram.py
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python scripts\preview_generator.py
python scripts\render_erd.py
python scripts\full_generator.py
python scripts\feature_engineering.py
python scripts\eda_summary.py
python scripts\train_model1.py
python scripts\train_model2.py
python scripts\train_model3.py
python scripts\generate_security_diagram.py
```

**Important:** Steps 3–9 must run in that order — feature engineering needs the full
dataset; EDA and all three model scripts need the features; Model 1 also reads
`data/full/_risk_profile_key.csv` for the evaluation-only oracle analysis; Model 3
reads `data/full/_fraud_key.csv` for evaluation.

---

## Troubleshooting

**"command not found: python3" (Mac) / "python is not recognized" (Windows)**
Python isn't installed or isn't on PATH. Reinstall from python.org; on Windows be
sure to check **"Add python.exe to PATH"** during install, then reopen the terminal.

**"No module named pandas" (or numpy, sklearn, shap, etc.)**
The virtual environment isn't active or libraries aren't installed. Activate it
(you should see `(venv)`), then `pip install -r requirements.txt`.

**Windows: "running scripts is disabled on this system"**
Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again;
or use `venv\Scripts\activate.bat` from Command Prompt.

**Windows: `UnicodeEncodeError` when a script prints**
Run `$env:PYTHONUTF8 = "1"` (PowerShell) before the script, or set `PYTHONUTF8=1`
as a permanent environment variable.

**"FileNotFoundError: .../data/full/..." when running feature_engineering.py**
You skipped Step 3. Run the full generator first.

**Figures look slightly different (fonts)**
Harmless. If a referenced font isn't installed, matplotlib substitutes a default.
The data and numbers are unaffected.

**"externally-managed-environment" error during pip install**
The virtual environment isn't active. Activate it first; the venv is the clean fix.

---

## When you're done

Leave the virtual environment with:
```bash
deactivate          # works on both macOS and Windows
```
Next session: navigate back to the folder and re-run the activate command before
running any scripts.
