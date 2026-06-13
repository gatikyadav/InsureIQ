# 🛡️ InsureIQ

An actuarially-grounded insurance risk modeling tool implementing the classical
frequency-severity pricing framework used by real insurance companies.

## Live Demo
🚀 [Coming soon]

## Overview

InsureIQ implements the industry-standard two-part actuarial pricing model:

**Pure Premium = E[Frequency | X] × E[Severity | X, N > 0]**

- **Frequency model** — Poisson GLM with log link and exposure offset
- **Severity model** — Gamma GLM with log link, fitted on claims-only rows
- **Benchmark** — XGBoost with Tweedie loss (compound Poisson-Gamma)
- **Dataset** — French Motor Third-Party Liability (MTPL), ~678K policies

This is the same modeling architecture used by personal lines insurers in the
US and Europe for auto and homeowners rate filings. The French MTPL dataset
is the standard academic benchmark for this class of models, used in peer-reviewed
actuarial literature (Noll, Salzmann, Wüthrich 2020).

## Key Findings

| Metric | GLM | XGBoost |
|--------|-----|---------|
| MSE | 204,042,720 | 203,848,178 |
| Gini Coefficient | 0.3093 | 0.1379 |

**XGBoost achieves marginally lower MSE (0.1% improvement) but the GLM
discriminates risk more than 2x better by Gini coefficient.** XGBoost also
showed instability on edge cases — predicting pure premiums up to €55,843 on
zero-claim policyholders vs the GLM's €31–€442 range. This illustrates why
the industry defaults to GLMs for regulatory rate filings: interpretability
and stability over marginal accuracy gains.

### Poisson GLM — Top Significant Coefficients

| Feature | Coef | Interpretation |
|---------|------|----------------|
| BonusMalus | +0.023 | Strongest predictor — tracks no-claims history |
| VehAge | -0.040 | Older vehicles file fewer claims |
| VehBrand_B12 | +0.149 | Riskiest brand in the dataset |
| Region_R83 | -0.273 | Lowest risk region in France |
| VehGas_Regular | +0.044 | Regular fuel slightly riskier than Diesel |

### XGBoost Feature Importance — Top 5
1. BonusMalus (~0.14 importance score — dominant predictor)
2. DrivAge (~0.08)
3. VehBrand_B11
4. Area_E
5. VehBrand_B4

### Rating Factor Insights
- **GLM** enforces a smooth monotonic relationship between DrivAge and pure
  premium — interpretable and regulator-friendly but rigid
- **XGBoost** captures non-linear patterns: young drivers (18–20) are high
  risk, dip at 30, rise again at 50–55, spike at 70–72 — more realistic but
  harder to justify in a rate filing

## Dashboard

| Tab | Description |
|-----|-------------|
| ⚡ Risk Scorer | Input policyholder features → claim frequency, severity, pure premium, risk tier, and gauge chart |
| 📊 Model Comparison | Lift curves, Gini, MSE, top disagreements, residual plots, XGBoost feature importance |
| 🔍 Rating Factor Explorer | GLM relativities vs XGBoost partial dependence per rating factor |
| 📁 Portfolio Simulator | Upload a CSV of policyholders → score entire portfolio, view loss distribution, export results |

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Data | `sklearn` / OpenML | `fetch_openml()` pulls MTPL directly |
| GLMs | `statsmodels` | Proper GLM with log link, offset, full coefficient table |
| Gradient Boosting | `xgboost` | Tweedie objective for compound loss |
| Visualization | `plotly`, `matplotlib` | Interactive charts + residual plots |
| UI | `streamlit` | Multi-tab dashboard |
| Deployment | Render / HuggingFace Spaces | Public URL |

## Setup

### Prerequisites
**Mac only:** XGBoost requires OpenMP:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenMP
brew install libomp
```

### Installation
```bash
git clone https://github.com/gatikyadav/InsureIQ.git
cd InsureIQ
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app trains all models on first load (~2 minutes). Subsequent loads use
Streamlit's cache and are instant.

## Dataset

**French Motor Third-Party Liability (MTPL)**
- ~678,013 policies (frequency table)
- ~26,000 claims (severity table)
- Source: `sklearn.datasets.fetch_openml('freMTPL2freq')` and `freMTPL2sev`
- Referenced in: Noll, Salzmann & Wüthrich (2020), *Case Study: French Motor
  Third-Party Liability Claims*

### Features
| Feature | Description |
|---------|-------------|
| Exposure | Fraction of year the policy was active |
| Area | Urban/rural area code (A–F) |
| VehPower | Vehicle engine power |
| VehAge | Vehicle age in years |
| DrivAge | Driver age |
| BonusMalus | No-claims discount score (50 = best, 150 = worst) |
| VehBrand | Vehicle brand category |
| VehGas | Fuel type (Diesel / Regular) |
| Density | Population density of driver's region |
| Region | French administrative region |

### Data Cleaning
- `VehGas` — stripped stray quotes from raw data (`'Regular'` → `Regular`)
- `BonusMalus` — capped at 150 to remove outliers
- `Density` — log-transformed (`log1p`) to handle right skew
- `ClaimAmount` — filled null with 0 for no-claim policies
- Zero/negative claim amounts filtered before Gamma GLM fit
- Exposure clipped at 1e-6 to avoid `log(0)` in offset

## Actuarial Framework

Insurance pricing decomposes expected loss into two components:

**Frequency** — how many claims will a policyholder generate?
Modeled as a Poisson GLM with log link and exposure offset. Appropriate
because claim counts are non-negative integers and exposure varies per policy.

**Severity** — what is the average cost per claim?
Modeled as a Gamma GLM with log link, fitted only on policies with at least
one claim. Appropriate because claim amounts are positive, continuous, and
right-skewed.

**Pure Premium** — the product of frequency and severity predictions.
This is the foundational quantity insurers use to set base rates. Both GLMs
use a log link, making the combined model multiplicative — coefficients are
interpretable as rating factor relativities, exactly as used in real US and
European personal lines pricing.