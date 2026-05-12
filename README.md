# 💼 Employee Income Prediction

A full end-to-end machine learning pipeline that predicts whether an individual's annual income exceeds **$50K**, built on the classic UCI Adult Census dataset. The project covers the complete ML lifecycle — from raw data analysis to a live interactive GUI — and was developed as a college AI/ML graduation project.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Models & Evaluation](#models--evaluation)
- [GUI](#gui)
- [Installation](#installation)
- [Usage](#usage)
- [Team](#team)

---

## Overview

The task is a **binary classification problem**: given demographic and employment data about an individual, predict whether they earn `>50K` or `<=50K` per year.

What makes this solution stand out:

- **Signal-driven feature engineering** — we derived new features (log-transformed capital values, net capital flow, overtime flag, marriage flag) rather than feeding raw columns directly into models.
- **Leak-free imbalance handling** — SMOTE is integrated *inside* the sklearn/imbalanced-learn Pipeline, ensuring synthetic samples are never generated before cross-validation splits, which prevents data leakage that inflates scores.
- **Dual preprocessing strategy** — scaled pipelines for linear/distance-based models, passthrough pipelines for tree-based models — so each algorithm gets the representation it actually needs.
- **Comprehensive benchmarking** — 8 algorithms × 2 variants (with/without SMOTE) = 16 configurations evaluated under Stratified K-Fold cross-validation for fair, comparable results.
- **Deployed as a live GUI** — built with Gradio so any non-technical user can run predictions without touching code.

---

## Dataset

The project uses the **UCI Adult Census Income** dataset, provided as two separate CSV files (`Train.csv` / `Test.csv`).

| Feature | Type | Notes |
|---|---|---|
| age | Numerical | |
| workclass | Categorical | Contains `?` as hidden nulls |
| fnlwgt | Numerical | High variance; treated as noisy |
| education | Categorical | Redundant with `education-num` |
| education-num | Numerical | Used over raw education labels |
| marital-status | Categorical | |
| occupation | Categorical | Contains `?` as hidden nulls |
| relationship | Categorical | |
| race | Categorical | Dropped after feature engineering |
| sex | Categorical | |
| capital-gain | Numerical | Highly skewed; log-transformed |
| capital-loss | Numerical | Highly skewed; log-transformed |
| hours-per-week | Numerical | Outliers up to 99 hrs |
| native-country | Categorical | High cardinality; dominated by US |
| **salary** | **Target** | `<=50K` (0) or `>50K` (1) |

**Class distribution:** ~75% `<=50K` / ~25% `>50K` — addressed with SMOTE.

---

## Project Structure

```
employee-income-prediction/
│
├── employees-salary-prediction-Final.ipynb   # Main notebook (full pipeline)
├── Train.csv                                  # Training data
├── Test.csv                                   # Test data
├── requirements.txt                           # Python dependencies
└── README.md
```

---

## Pipeline Architecture

```
Raw Data
   │
   ├── Data Cleaning
   │     ├── Replace "?" with NaN → impute with mode
   │     ├── Drop duplicates
   │     └── IQR-based outlier capping (robust to skewed distributions)
   │
   ├── Feature Engineering
   │     ├── capital_gain_log      = log1p(capital-gain)
   │     ├── capital_loss_log      = log1p(capital-loss)
   │     ├── capital_net_log       = capital_gain_log - capital_loss_log
   │     ├── is_any_capital        = 1 if any capital activity
   │     ├── works_overtime        = 1 if hours-per-week > 40
   │     ├── is_us                 = 1 if native-country == United-States
   │     ├── is_married            = 1 if Married-civ-spouse or Married-AF-spouse
   │     └── education-num         = ordinal mapping from education label
   │
   ├── Preprocessing (per model type)
   │     ├── Scaled pipeline       → OneHotEncoder + StandardScaler  (LR, KNN, SVM)
   │     ├── Tree pipeline         → OneHotEncoder + passthrough      (RF, ET, XGB, SMOTE models)
   │     └── Boost pipeline        → passthrough + passthrough        (CatBoost native)
   │
   ├── SMOTE (inside Pipeline — no leakage)
   │     └── k_neighbors=5, random_state=30
   │
   └── Model Training & Evaluation
         └── StratifiedKFold(n_splits=5) → Accuracy, Precision, Recall, F1
```

---

## Models & Evaluation

All models were evaluated using **Stratified 5-Fold Cross-Validation**. Each algorithm was tested in two configurations: standard and with SMOTE integrated in-pipeline.

| Model | Variant |
|---|---|
| Logistic Regression | Standard / + SMOTE |
| K-Nearest Neighbors | Standard / + SMOTE |
| Support Vector Machine | Standard / + SMOTE |
| Decision Tree | Standard / + SMOTE |
| Random Forest | Standard / + SMOTE |
| Extra Trees | Standard / + SMOTE |
| XGBoost | Standard / + SMOTE |
| CatBoost | Standard / + SMOTE |

Metrics reported: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix** for the final selected model.

> The best-performing model was selected based on F1-score to balance precision and recall given the class imbalance.

---

## GUI

The interactive GUI was built with **Gradio** and exposes all 12 model input features through dropdowns, sliders, and number fields.

**Input fields:**
- Age (slider), Hours per Week (slider)
- Capital Gain, Capital Loss (numeric inputs)
- Education, Workclass, Occupation, Marital Status, Relationship, Sex, Race, Native Country (dropdowns)

**Output:** Instant prediction — `>50K` or `<=50K`

To launch the GUI, run the final cell in the notebook after training the model.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/employee-income-prediction.git
cd employee-income-prediction
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch Jupyter**
```bash
jupyter notebook employees-salary-prediction-Final.ipynb
```

---

## Usage

1. Run all cells in order from top to bottom.
2. The notebook will train all 16 model configurations and display cross-validation results.
3. The final cell launches the **Gradio GUI** — a local URL will appear in the output (e.g., `http://127.0.0.1:7860`).
4. Open the URL in your browser, fill in the fields, and click **Predict Income 🚀**.

---

## Team

This project was developed by a team of 6 as a college AI/ML graduation project.

| Name |
|---|
| *(Your Name)* |
| Adham |
| Abdelrahman |
| Wegdan |
| Rawan |
| Yasmeen |

---

## License

This project is open-source and available under the [MIT License](LICENSE).
