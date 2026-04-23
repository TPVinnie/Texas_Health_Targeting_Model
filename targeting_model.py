"""
Texas Health – Houston Targeting Model
Manager I, Data Science Technical Assessment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
train = pd.read_csv("/mnt/user-data/uploads/ds_test_training_dataset_2024.csv")
houston = pd.read_csv("/mnt/user-data/uploads/ds_test_houston_2024.csv")
county_summary = pd.read_csv("/mnt/user-data/uploads/tx_county_summary.csv")
county_enrollment = pd.read_csv("/mnt/user-data/uploads/tx_county_enrollment_rates.csv")
county_to_region = pd.read_csv("/mnt/user-data/uploads/county_to_media_region.csv")

print(f"Training rows: {len(train)}")
print(f"Houston rows:  {len(houston)}")
print(f"Target distribution:\n{train['Plan Enrolled'].value_counts()}\n")

# ─────────────────────────────────────────
# 2. TARGET ENCODING
# ─────────────────────────────────────────
# Binary: Any enrollment vs None
train["enrolled"] = (train["Plan Enrolled"] != "Not Subscribed").astype(int)
print(f"Enrollment rate in training: {train['enrolled'].mean():.1%}\n")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
def encode_bbq(val):
    mapping = {"No Interest": 0, "Some Interest": 1, "Strong Interest": 2}
    return mapping.get(str(val).strip(), 0)

def build_features(df, county_summary, county_enrollment):
    df = df.copy()
    
    # Encode BBQ interest
    df["bbq_interest_score"] = df["Interest in Barbeque"].apply(encode_bbq)
    
    # Encode categorical features
    df["gender_enc"] = (df["gender"] == "Female").astype(int)
    
    party_map = {"Democrat": 0, "Republican": 1, "Independent": 2, "Unknown": 3}
    df["party_enc"] = df["political_party"].map(party_map).fillna(3)
    
    edu_map = {"No HS": 0, "HS": 1, "Some College": 2, "College": 3, "Post Grad": 4}
    df["edu_enc"] = df["education_area"].map(edu_map).fillna(2)
    
    # Derived features
    df["income_age_ratio"] = df["income"] / (df["age"] + 1)
    df["children_homeowner"] = df["has_children"] * df["is_homeowner"]
    df["financial_interest"] = df["interests_investing"] + df["donor_health_org"]
    df["donor_total"] = (df[["donor_political_org","donor_liberal_org","donor_conservative_org",
                               "donor_religious_org","donor_health_org"]].sum(axis=1))
    df["purchase_total"] = (df[["purchases_apparel","purchases_book","purchases_electronic",
                                  "purchases_boat","purchases_luxuryitems"]].sum(axis=1))
    df["interest_total"] = (df[["interests_environment","interests_outdoorgarden","interests_outdoorsport",
                                  "interests_guns","interests_golf","interests_investing",
                                  "interests_veteranaffairs"]].sum(axis=1))
    
    # Merge county enrollment rate (county-level proxy for market penetration)
    county_summary_clean = county_summary.copy()
    county_summary_clean["county_name_upper"] = county_summary_clean["county_name"].str.upper()
    df["county_name_upper"] = df["county_name"].str.upper()
    
    df = df.merge(county_summary_clean[["censuskey","county_name_upper",
                                          "c14_median_income","c14_pct_hispanic",
                                          "c14_pct_white","c14_pct_black","c14_pct_asian",
                                          "c14_pct_nohsdegree","c14_pct_collegedegree",
                                          "c14_pct_renter","c14_pct_owner",
                                          "c14_pop_density_sqmile","c14_pct_multitenant",
                                          "c14_pct_occupation_employedcivilian"]],
                  left_on="county_name_upper", right_on="county_name_upper", how="left")
    
    df = df.merge(county_enrollment, left_on="censuskey", right_on="county", how="left")
    
    return df

train_feat = build_features(train, county_summary, county_enrollment)
houston_feat = build_features(houston, county_summary, county_enrollment)

# ─────────────────────────────────────────
# 4. SELECT FEATURES
# ─────────────────────────────────────────
FEATURE_COLS = [
    "age","income","length_of_residence","number_of_children","has_children",
    "is_homeowner","is_renter","gender_female","gender_male",
    "maritalstatus_single","maritalstatus_married",
    "religion_catholic","religion_christian",
    "donor_political_org","donor_liberal_org","donor_conservative_org",
    "donor_religious_org","donor_health_org",
    "occupation_blue_collar","occupation_farmer","occupation_professional_technical","occupation_retired",
    "purchases_apparel","purchases_book","purchases_electronic","purchases_boat","purchases_luxuryitems",
    "has_a_cat",
    "interests_environment","interests_outdoorgarden","interests_outdoorsport",
    "interests_guns","interests_golf","interests_investing","interests_veteranaffairs",
    "ethnicity_afam","ethnicity_latino","ethnicity_asian","ethnicity_white","ethnicity_other",
    "bbq_interest_score","party_enc","edu_enc",
    "income_age_ratio","children_homeowner","financial_interest",
    "donor_total","purchase_total","interest_total",
    "c14_median_income","c14_pct_hispanic","c14_pct_white","c14_pct_black","c14_pct_asian",
    "c14_pct_nohsdegree","c14_pct_collegedegree","c14_pct_renter","c14_pct_owner",
    "c14_pop_density_sqmile","c14_pct_multitenant","c14_pct_occupation_employedcivilian",
    "pct.enrolled"
]

# Only keep features that exist in both datasets
FEATURE_COLS = [c for c in FEATURE_COLS if c in train_feat.columns and c in houston_feat.columns]

X_train = train_feat[FEATURE_COLS].fillna(0)
y_train = train_feat["enrolled"]
X_score = houston_feat[FEATURE_COLS].fillna(0)

print(f"Features used: {len(FEATURE_COLS)}")

# ─────────────────────────────────────────
# 5. MODEL TRAINING WITH CROSS-VALIDATION
# ─────────────────────────────────────────
print("\n--- Cross-Validation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
lgb_scores = cross_val_score(lgb_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

print(f"XGBoost  5-fold AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print(f"LightGBM 5-fold AUC: {lgb_scores.mean():.4f} ± {lgb_scores.std():.4f}")

# Train final models on all data
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Ensemble probabilities
xgb_proba_train = xgb_model.predict_proba(X_train)[:,1]
lgb_proba_train = lgb_model.predict_proba(X_train)[:,1]
ensemble_proba_train = 0.5 * xgb_proba_train + 0.5 * lgb_proba_train
train_auc = roc_auc_score(y_train, ensemble_proba_train)
print(f"\nEnsemble train AUC (in-sample): {train_auc:.4f}")

# ─────────────────────────────────────────
# 6. SCORE HOUSTON DATA
# ─────────────────────────────────────────
xgb_proba_h = xgb_model.predict_proba(X_score)[:,1]
lgb_proba_h = lgb_model.predict_proba(X_score)[:,1]
houston_proba = 0.5 * xgb_proba_h + 0.5 * lgb_proba_h

houston_feat["enrollment_score"] = houston_proba

# ─────────────────────────────────────────
# 7. DECILE ANALYSIS
# ─────────────────────────────────────────
# Validate score lift using training data
train_feat["score"] = ensemble_proba_train
train_feat["decile"] = pd.qcut(train_feat["score"], 10, labels=False, duplicates="drop")
decile_lift = (train_feat.groupby("decile")["enrolled"]
                .agg(["mean","count"])
                .sort_index(ascending=False)
                .rename(columns={"mean":"enrollment_rate","count":"n"}))
decile_lift["lift"] = decile_lift["enrollment_rate"] / y_train.mean()
print("\n--- Score Decile Analysis (Training Data) ---")
print(decile_lift.round(3))

# ─────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────
fi = pd.DataFrame({
    "feature": FEATURE_COLS,
    "xgb_importance": xgb_model.feature_importances_,
    "lgb_importance": lgb_model.feature_importances_ / lgb_model.feature_importances_.sum()
}).sort_values("xgb_importance", ascending=False)
print("\n--- Top 15 Features (XGBoost) ---")
print(fi.head(15)[["feature","xgb_importance"]].to_string(index=False))

# ─────────────────────────────────────────
# 9. PRODUCE RANKED OUTPUT FILE
# ─────────────────────────────────────────
ranked = (houston[["id","county_name","media_market"]]
          .copy()
          .assign(enrollment_score=houston_proba)
          .sort_values("enrollment_score", ascending=False)
          .reset_index(drop=True))
ranked["rank"] = ranked.index + 1
ranked = ranked[["rank","id","county_name","media_market","enrollment_score"]]

ranked.to_csv("/mnt/user-data/outputs/houston_ranked_outreach.csv", index=False)
print(f"\nRanked file saved: {len(ranked)} records")
print(ranked.head(10))

# ─────────────────────────────────────────
# 10. SAVE VALIDATION STATS FOR RELEASE NOTES
# ─────────────────────────────────────────
stats = {
    "training_n": len(train),
    "houston_n": len(houston),
    "enrollment_rate_train": float(y_train.mean()),
    "xgb_cv_auc_mean": float(xgb_scores.mean()),
    "xgb_cv_auc_std": float(xgb_scores.std()),
    "lgb_cv_auc_mean": float(lgb_scores.mean()),
    "lgb_cv_auc_std": float(lgb_scores.std()),
    "top_features": fi.head(10)["feature"].tolist(),
    "decile_analysis": decile_lift.reset_index().to_dict(orient="records")
}

import json
with open("/home/claude/model_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n✓ All outputs complete.")
