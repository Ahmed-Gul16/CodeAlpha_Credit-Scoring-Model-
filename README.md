# CodeAlpha_Credit-Scoring-Model-
A python notebook training 3 different classification models to check the credictworthiness of loan requester.

# Credit Risk Classification Pipeline

An end-to-end Machine Learning pipeline developed to predict customer creditworthiness using a comprehensive financial and demographic dataset (12,000 records, 41 features). This project transitions from data exploration and rigorous domain feature engineering to a comparative analysis of linear and tree-based classification models.

## Project Workflow & Highlights

### 1. Exploratory Data Analysis (EDA)
* Identified a stable **66:34 class distribution** (`creditworthy`), eliminating the need for aggressive synthetic resampling but guiding the switch to **F1-Score** and **ROC-AUC** as primary evaluation metrics.
* Isolated key domain variable (`credit_score`) showing distinct, statistically significant interquartile separation between target groups.
* Audited feature matrices for **multicollinearity** to protect linear decision boundaries.

### 2. Pipeline Feature Engineering
* **Structured Encodings:** Handled categorical data intelligently by mapping `education_level` ordinally to preserve academic hierarchy, while utilizing drop-first One-Hot Encoding for nominal features (`employment_status`, `housing_status`, `loan_purpose`, `state`).
* **Domain Metrics:** Synthesized raw features into aggregate, risk-reflective variables including:
  * `total_financial_assets` (Liquid wealth cushion across accounts)
  * `total_delinquencies` (Cumulative historical payment defaults)
  * `loan_to_asset_ratio` (Requested debt footprint relative to liquid holdings)
* **Data Leakage Safeguards:** Explicitly quarantined feature scaling transformations (`StandardScaler`) to fit strictly on training splits.

### 3. Progressive Model Evaluation
The notebook establishes a clear baseline before introducing higher-complexity architectures:

* **Baseline — Logistic Regression:** Built as an interpretable linear benchmark. Required robust feature scaling to safely handle varying feature magnitudes.
* **Non-Linear Splitter — Decision Tree:** Scaled to map explicit rule-based thresholds. Regularized with structural constraints (`max_depth=5`) to prevent overfitting.
* **Ensemble Champion — Random Forest:** Leveraged Bootstrap Aggregating (Bagging) over 100 decision estimators. Successfully flattened feature variance, spreading reliance safely across secondary indicators like `bankruptcies` and `annual_income`.

---

## Summary Performance Metrics

| Model | Accuracy | Recall (Class 1) | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 81% | 90% | 0.8694 |
| **Decision Tree** | 81% | 90% | 0.8618 |
| **Random Forest** | 81% | 91% | 0.8669 |

---

## Tech Stack & Requirements
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`



