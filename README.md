# Texas Health — Houston Market Enrollment Targeting Model

> **Predicting HSA/FSA enrollment likelihood for Houston-area residents using an XGBoost + LightGBM ensemble model.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Internal%20Use%20Only-red)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Inputs](#3-data-inputs)
4. [How the Model Works](#4-how-the-model-works)
5. [Key Predictors of Enrollment](#5-key-predictors-of-enrollment)
6. [How to Use the Score](#6-how-to-use-the-score)
7. [Model Validation](#7-model-validation)
8. [Limitations & Caveats](#8-limitations--caveats)
9. [Quickstart](#9-quickstart)
10. [Recommended Next Steps](#10-recommended-next-steps)

---

## 1. Project Overview

Texas Health is expanding into the Houston media market. This project delivers a machine learning targeting model to maximize enrollment in the state-based **Health Savings Account (HSA)** and **Flexible Spending Account (FSA)** program by identifying which Houston residents should be prioritized for individual outreach.

Two plan types are available:

| Plan | Description |
|------|-------------|
| **Plan Red** | HSA combined with a Dependent Care FSA |
| **Plan Blue** | HSA only |

> **Key assumption:** No Houston residents have had prior access to this program, but their demographic behaviors are assumed to mirror the broader Texas population used to train the model.

---

## 2. Repository Structure

```
texas-health-houston-targeting/
│
├── data/                               # Input data files (not committed — see below)
│   ├── ds_test_training_dataset_2024.csv
│   ├── ds_test_houston_2024.csv
│   ├── tx_county_summary.csv
│   ├── tx_county_enrollment_rates.csv
│   └── county_to_media_region.csv
│
├── outputs/
│   └── houston_ranked_outreach.csv     # Final ranked outreach list (2,539 records)
│
├── targeting_model.py                  # Full modeling pipeline (train → score → rank)
├── Texas_Health_Houston_Targeting_Model.ipynb  # Jupyter notebook with full walkthrough
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

> ⚠️ Raw data files contain PII and are excluded from version control. Contact the Analytics Team for access.

---

## 3. Data Inputs

The model was built using five data sources:

| File | Records | Description |
|------|---------|-------------|
| `ds_test_training_dataset_2024.csv` | 7,462 | Individual TX residents with known enrollment outcomes (Plan Red / Plan Blue / Not Subscribed) — **the training set** |
| `ds_test_houston_2024.csv` | 2,539 | Houston-area residents with no prior enrollment history — **the scoring target** |
| `tx_county_summary.csv` | 254 counties | Census-derived demographics: income, education, ethnicity, housing tenure, employment |
| `tx_county_enrollment_rates.csv` | 254 counties | Observed county-level enrollment penetration rates from existing markets |
| `county_to_media_region.csv` | — | Lookup mapping county names to Texas media market regions |

The training data shows approximately a **50/50 split** between enrolled and non-enrolled residents, providing balanced signal for model training with no special class imbalance handling required.

---

## 4. How the Model Works

The model learns patterns from the 7,462 Texans with known enrollment outcomes, then applies those patterns to predict which Houston residents are most likely to enroll.

**Pipeline summary:**

```
Raw individual data  ──┐
                       ├──▶  Feature Engineering  ──▶  XGBoost ──┐
County demographics  ──┤                                           ├──▶  Ensemble Score  ──▶  Ranked Output
County enroll rates  ──┘                              LightGBM ──┘
```

**Two independent models** (XGBoost and LightGBM) were trained and their predicted probabilities averaged into a single ensemble score. This approach:

- Reduces prediction variance compared to either model alone
- Smooths out individual model overconfidence
- Is more robust to the distribution shift between statewide TX training data and the Houston scoring population

**Score interpretation:**

| Score | Meaning |
|-------|---------|
| `0.90` | 90% estimated probability of enrollment if contacted |
| `0.50` | Coin-flip likelihood |
| `0.10` | 10% probability — low priority |

---

## 5. Key Predictors of Enrollment

The following attributes were the most influential signals the model identified. These should inform how outreach messaging is tailored.

| Feature | Importance | Business Interpretation |
|---------|-----------|------------------------|
| Length of Residence | 🔴 High | Longer-tenure residents show higher intent to enroll |
| Latino Ethnicity | 🔴 High | Cultural-community effect on HSA/FSA adoption patterns |
| Political Donor Activity | 🔴 High | Civic engagement correlates with benefits participation |
| Income Level | 🔴 High | Higher income drives FSA/HSA eligibility and interest |
| Luxury Purchases | 🟡 Medium | Signals discretionary financial planning behavior |
| Donor to Health Orgs | 🟡 Medium | Existing health-conscious financial behavior |
| Homeownership | 🟡 Medium | Financial stability and benefit-awareness indicator |
| White Ethnicity | 🟡 Medium | Demographic segment with distinct enrollment patterns |
| Renter Status | 🟡 Medium | Contrasting housing patterns shape financial priorities |
| Age | 🟡 Medium | Age-band dynamics influence HSA vs FSA preference |

---

## 6. How to Use the Score

### Reading the Ranked File

`outputs/houston_ranked_outreach.csv` contains all 2,539 Houston individuals sorted from most likely to least likely to enroll.

| Column | Description |
|--------|-------------|
| `rank` | Outreach priority — **1 = contact first** |
| `id` | Unique person identifier for matching to your outreach system |
| `county_name` | County within the Houston market |
| `media_market` | Always `"houston tx"` in this file |
| `enrollment_score` | Model probability estimate (0 to 1) |

### Recommended Tiering Strategy

| Tier | Score Range | Recommended Action |
|------|-------------|-------------------|
| **Tier 1 — Hot** 🔥 | ≥ 0.75 | Personal outreach — direct mail, phone, or personalized digital touchpoint |
| **Tier 2 — Warm** 🌤️ | 0.40 – 0.74 | Targeted digital ads or mailer — efficient spend, decent conversion |
| **Tier 3 — Cold** 🧊 | < 0.40 | Broad awareness campaigns only — low expected return on individual outreach |

> **Tip:** If outreach capacity is fixed (e.g., a budget for 500 contacts), work from Rank 1 downward. Contacting the top-ranked individuals yields the highest expected enrollment count per outreach dollar.

---

## 7. Model Validation

### Cross-Validation Results

The model was evaluated using **5-fold stratified cross-validation** — each fold withholds 20% of training data, trains on the remaining 80%, and measures AUC on the held-out portion. This mimics real-world performance on unseen individuals like those in Houston.

| Model | 5-Fold CV AUC | Std. Deviation |
|-------|--------------|----------------|
| XGBoost | **0.718** | ±0.016 |
| LightGBM | **0.720** | ±0.018 |
| **Ensemble (Average)** | **0.720** | — |

> **AUC** (Area Under the ROC Curve) measures discrimination on a scale from 0.5 (random) to 1.0 (perfect). A score of **0.72** is a solid result for consumer behavioral targeting — substantially better than chance.

### Score Decile Lift Analysis

Training individuals were divided into 10 equal score groups (deciles) and actual enrollment rates were measured per group:

| Score Decile | Enrollment Rate | Lift vs. Average | Interpretation |
|-------------|----------------|-----------------|----------------|
| 10 (Highest) | 99.7% | **2.0×** | Prioritize first — near-certain enrollers |
| 9 | 94.4% | **1.9×** | Highly likely — strong outreach targets |
| 8 | 83.1% | **1.7×** | Likely — good ROI for outreach resources |
| 7 | 73.9% | **1.5×** | Moderate-high — worth including in campaigns |
| 6 | 56.8% | 1.1× | Near average — use if capacity allows |
| 5 | 43.3% | 0.9× | Below average — deprioritize |
| 4 | 25.1% | 0.5× | Low likelihood — not recommended |
| 3 | 13.4% | 0.3× | Very low — skip unless volume-constrained |
| 2 | 5.5% | 0.1× | Very low |
| 1 (Lowest) | 0.4% | ~0× | Do not contact |

> The top decile achieved a **2.0× lift** — individuals in that group enrolled at twice the average rate. Focusing outreach on the top three deciles yields enrollment rates of **83%–100%**.

---

## 8. Limitations & Caveats

- **Behavioral assumption:** The model assumes Houston residents behave like the broader Texas population used for training. If Houston has meaningfully different cultural or economic dynamics, performance may vary.

- **No ground-truth Houston labels:** Because no Houston enrollment data exists yet, model performance on this specific population cannot be directly measured until outreach occurs. Tracking actual enrollments post-campaign is strongly recommended.

- **Probabilistic, not deterministic:** A score of 0.85 does not guarantee enrollment — it indicates high likelihood based on observed patterns. Outreach quality, messaging, and timing also affect conversion.

- **Temporal drift:** If the program runs over many months, demographics may shift. Periodic model retraining every **6–12 months** is recommended.

---

## 9. Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
lightgbm>=3.3
matplotlib>=3.6
jupyter>=1.0
```

### Run the modeling pipeline

```bash
# Place data files in ./data/ then run:
python targeting_model.py
```

Output: `outputs/houston_ranked_outreach.csv`

### Explore the notebook

```bash
jupyter notebook Texas_Health_Houston_Targeting_Model.ipynb
```

The notebook walks through every step — data loading, merging, feature engineering, cross-validation, ensemble scoring, validation charts, and the final ranking — with Markdown explanations at each stage.

---

## 10. Recommended Next Steps

1. **Execute tiered outreach** using the ranked file, prioritizing Tier 1 (score ≥ 0.75) for high-touch engagement.
2. **Track enrollment outcomes** by outreach tier and link actual results to individual IDs in the ranked file.
3. **Post-campaign lift analysis** — compare enrollment rates across score deciles using actual outcomes to validate model performance in Houston.
4. **Retrain** incorporating Houston enrollment data once available, improving accuracy for future expansion phases.
5. **Plan sub-model** — develop a Plan Red vs. Plan Blue classifier to further personalize outreach messaging by plan type.

---

## Project Info

**Prepared by:** Analytics Team  
**Date:** April 2024  
**Classification:** Internal Use Only

---

*For questions about the model or data access, contact the Analytics Team.*
