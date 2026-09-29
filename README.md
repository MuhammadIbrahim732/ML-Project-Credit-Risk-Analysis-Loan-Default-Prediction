# 💳 Credit Risk Analysis & Loan Default Prediction

> An end-to-end Machine Learning project for predicting **loan default risk** using customer financial, employment, credit-history, and loan-related attributes.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC4E20)
![SHAP](https://img.shields.io/badge/SHAP-Model%20Explainability-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Credit risk assessment is an important machine learning application in the financial sector. The goal of this project is to develop a classification system capable of identifying applicants who are more likely to experience **loan default**.

The project follows a complete machine learning workflow:

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Validation
     ↓
Duplicate & Invalid Record Handling
     ↓
Feature / Target Separation
     ↓
Stratified Train-Test Split
     ↓
Class Imbalance Handling
     ↓
Feature Preprocessing
     ↓
5-Fold Stratified Cross-Validation
     ↓
Logistic Regression Baseline
     ↓
XGBoost Model
     ↓
Hyperparameter Optimization
     ↓
Probability Calibration
     ↓
Threshold Optimization
     ↓
SHAP Explainability
     ↓
False Positive / False Negative Analysis
     ↓
Model Serialization
```

The final workflow produces a calibrated XGBoost-based model together with an optimized classification threshold for practical prediction.

---

# 🎯 Objectives

The main objectives of this project are to:

* Analyze customer and loan-related data.
* Identify data-quality issues and invalid records.
* Explore class imbalance in loan default status.
* Build a strong preprocessing pipeline.
* Establish Logistic Regression as a baseline model.
* Train and evaluate an XGBoost classifier.
* Handle class imbalance using class weighting.
* Perform stratified cross-validation.
* Optimize XGBoost hyperparameters.
* Evaluate classification performance using multiple metrics.
* Calibrate predicted probabilities.
* Optimize the classification threshold using F1-score.
* Interpret model predictions using SHAP.
* Analyze false positives and false negatives.
* Save the trained model for future inference.

---

# 📊 Dataset

The project uses a credit-risk dataset containing **32,581 records and 12 features** before cleaning.

### Dataset Features

| Feature                      | Description                            | Type        |
| ---------------------------- | -------------------------------------- | ----------- |
| `person_age`                 | Applicant's age                        | Numerical   |
| `person_income`              | Applicant's annual income              | Numerical   |
| `person_home_ownership`      | Home ownership status                  | Categorical |
| `person_emp_length`          | Employment length                      | Numerical   |
| `loan_intent`                | Purpose of the loan                    | Categorical |
| `loan_grade`                 | Loan grade                             | Categorical |
| `loan_amnt`                  | Requested loan amount                  | Numerical   |
| `loan_int_rate`              | Loan interest rate                     | Numerical   |
| `loan_status`                | Target variable indicating loan status | Binary      |
| `loan_percent_income`        | Loan amount as a percentage of income  | Numerical   |
| `cb_person_default_on_file`  | Historical default indicator           | Categorical |
| `cb_person_cred_hist_length` | Length of credit history               | Numerical   |

### Target Variable

The target is:

```text
loan_status
```

where:

```text
0 → No default
1 → Default
```

The original dataset contains:

* **25,473** samples with class `0`
* **7,108** samples with class `1`

This demonstrates a significant **class imbalance**, with default cases representing a smaller portion of the dataset.

---

# 🔎 Exploratory Data Analysis

The EDA stage examines:

* Dataset dimensions
* Data types
* Missing values
* Target distribution
* Statistical summaries
* Numerical feature distributions
* Potential outliers
* Correlation between numerical variables

### Dataset Shape

```text
Rows:    32,581
Columns: 12
```

### Missing Values

The main missing values were found in:

```text
person_emp_length    895
loan_int_rate       3116
```

Rather than dropping large numbers of observations, missing numerical values are handled later through the preprocessing pipeline.

---

# 🧹 Data Validation & Cleaning

The project performs data validation before model training.

### 1. Duplicate Removal

Original dataset:

```text
32,581 rows
```

After removing duplicates:

```text
32,416 rows
```

Therefore:

```text
165 duplicate rows
```

were removed.

### 2. Age Validation

Records with unrealistic ages were identified.

The project applies:

```text
18 ≤ person_age ≤ 100
```

This removes invalid age records such as applicants above 100 years old.

### 3. Employment Length Validation

Employment length is constrained to:

```text
person_emp_length ≤ 60
```

### 4. Income Validation

Rows with zero or negative income are removed:

```text
person_income > 0
```

### 5. Loan Amount Validation

Rows with zero or negative loan amounts are removed:

```text
loan_amnt > 0
```

### Important Design Decision

Not every statistical outlier was automatically removed.

The project distinguishes between:

```text
Statistical outlier
        ≠
Invalid business/data value
```

For example, a high income or large loan amount may be unusual but still represent a valid applicant.

---

# 🧩 Feature & Target Preparation

The target column is separated from the predictors:

```python
X = df_copy.drop("loan_status", axis=1)
y = df_copy["loan_status"]
```

The dataset is divided using a stratified train-test split:

```text
Training set → 67%
Testing set  → 33%
```

with:

```python
random_state = 42
stratify = y
```

Stratification ensures that the class distribution remains approximately consistent between training and testing data.

---

# ⚖️ Handling Class Imbalance

The target distribution is imbalanced.

The training data contains approximately:

```text
Class 0 → 16,558
Class 1 → 4,561
```

The project calculates the positive-class weight as:

```python
scale_weight = negative_samples / positive_samples
```

Result:

```text
scale_weight ≈ 3.63
```

This weighting is incorporated into the models to give greater importance to the minority/default class.

### Logistic Regression

Uses:

```python
class_weight="balanced"
```

### XGBoost

Uses:

```python
scale_pos_weight=3.63
```

---

# ⚙️ Feature Preprocessing

The project uses `Pipeline` and `ColumnTransformer` to keep preprocessing consistent and avoid applying transformations manually.

## Numerical Features

Numerical features:

```text
person_age
person_income
person_emp_length
loan_amnt
loan_int_rate
loan_percent_income
cb_person_cred_hist_length
```

### Logistic Regression

Numerical preprocessing:

```text
Missing values
      ↓
Median Imputation
      ↓
Standard Scaling
```

Implemented using:

* `SimpleImputer(strategy="median")`
* `StandardScaler()`

---

## XGBoost

XGBoost does not require standard scaling for numerical features.

Therefore:

```text
Missing values
      ↓
Median Imputation
```

is applied without `StandardScaler`.

---

## Categorical Features

Categorical features:

```text
person_home_ownership
loan_intent
loan_grade
cb_person_default_on_file
```

Preprocessing:

```text
Missing values
      ↓
Constant Imputation
      ↓
One-Hot Encoding
```

Using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This also ensures that unseen categories during inference do not break the pipeline.

---

# 🔄 Stratified 5-Fold Cross-Validation

To obtain a more reliable estimate of model performance, the project uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The following metrics are evaluated:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

# 🧪 Baseline Model — Logistic Regression

Logistic Regression is used as the baseline classification model.

Configuration:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

### Cross-Validation Performance

| Metric    |      Score |
| --------- | ---------: |
| ROC-AUC   | **0.8713** |
| Accuracy  | **0.8110** |
| Recall    | **0.7792** |
| Precision | **0.5436** |
| F1-score  | **0.6404** |

### Test Set Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **0.8163** |
| Precision | **0.5530** |
| Recall    | **0.7787** |
| F1-score  | **0.6467** |
| ROC-AUC   | **0.8718** |

The baseline provides a useful reference point for evaluating the more complex XGBoost model.

---

# 🚀 XGBoost Model

XGBoost is used as the primary tree-based classification model.

The initial model incorporates class imbalance handling through:

```python
scale_pos_weight = 3.63
```

### Initial XGBoost Test Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **0.9229** |
| Precision | **0.8425** |
| Recall    | **0.7907** |
| F1-score  | **0.8158** |

> Note: The ROC-AUC value displayed in the notebook's initial XGBoost evaluation uses the baseline Logistic Regression probability variable. Therefore, that particular ROC-AUC value should not be interpreted as the XGBoost ROC-AUC. The later tuned XGBoost evaluation reports a ROC-AUC of approximately **0.9459**.

---

# 🎛️ Hyperparameter Optimization

The project uses:

```python
RandomizedSearchCV
```

instead of exhaustively evaluating every possible parameter combination.

### Search Configuration

```text
Iterations: 150
Cross-validation: 5 folds
Scoring: Average Precision
Parallel processing: n_jobs=-1
```

This results in:

```text
150 × 5 = 750 model fits
```

### Parameters Tuned

| Parameter          | Search Space |
| ------------------ | ------------ |
| `n_estimators`     | 100–500      |
| `max_depth`        | 3–9          |
| `learning_rate`    | 0.01–0.30    |
| `subsample`        | 0.60–1.00    |
| `colsample_bytree` | 0.60–1.00    |
| `min_child_weight` | 1–9          |
| `gamma`            | 0–5          |

### Best Parameters Found

```text
n_estimators       = 173
max_depth          = 5
learning_rate      = 0.1754
subsample          = 0.9855
colsample_bytree   = 0.6168
min_child_weight   = 6
gamma              = 1.1729
```

Best cross-validation average-precision score:

```text
≈ 0.90
```

---

# 📈 Cross-Validation Model Comparison

The initial cross-validation results show the difference between the baseline and tree-based model.

| Model               | ROC-AUC | Accuracy | Recall | Precision |     F1 |
| ------------------- | ------: | -------: | -----: | --------: | -----: |
| Logistic Regression |  0.8713 |   0.8110 | 0.7792 |    0.5436 | 0.6404 |
| XGBoost             |  0.9401 |   0.9114 | 0.7886 |    0.7989 | 0.7936 |

These results demonstrate that the XGBoost model captures nonlinear relationships and feature interactions that the linear baseline may not capture as effectively.

---

# 🎯 Probability Calibration

For a credit-risk application, the predicted probability can be as important as the final class label.

Instead of treating:

```text
0.51 → Default
0.49 → No Default
```

as the complete output, probability calibration attempts to make predicted probabilities more representative of observed frequencies.

The project applies:

```python
CalibratedClassifierCV(
    best_model,
    cv=CV,
    method="sigmoid"
)
```

The calibrated model is then used to generate:

```python
calibrated_prob = calibrated_model.predict_proba(X_test)[:, 1]
```

Calibration curves are plotted to compare:

* Uncalibrated probabilities
* Calibrated probabilities
* Perfect calibration reference

---

# 🎚️ Classification Threshold Optimization

The default classification threshold is generally:

```text
0.50
```

However, the optimal threshold depends on the application's objective.

This project searches for a threshold that maximizes the **F1-score** using the calibrated probabilities.

The optimized threshold found in the notebook is:

```text
Best Threshold ≈ 0.6064
```

Therefore, prediction can be performed conceptually as:

```python
prediction = probability >= 0.6064
```

This allows the decision boundary to be adjusted instead of blindly relying on `0.50`.

---

# 🔍 Model Explainability with SHAP

Machine learning models used in financial applications should ideally provide interpretable reasoning.

This project uses **SHAP (SHapley Additive exPlanations)** to investigate model predictions.

### Global Explainability

A SHAP summary plot is used to understand:

* Which features influence predictions most strongly.
* The direction of their contribution.
* The distribution of feature effects across test samples.

### Local Explainability

A SHAP waterfall plot is generated for an individual test sample.

It shows how individual feature values contribute toward increasing or decreasing the model's prediction.

This provides both:

```text
Global model interpretation
```

and

```text
Individual prediction explanation
```

---

# ❌ False Positive & False Negative Analysis

The project also investigates prediction errors instead of relying only on aggregate metrics.

### False Positive

A false positive occurs when:

```text
Actual = 0
Predicted = 1
```

Meaning the model predicted default risk when the actual class was non-default.

### False Negative

A false negative occurs when:

```text
Actual = 1
Predicted = 0
```

Meaning the model failed to identify an actual default case.

The notebook explicitly extracts these cases from the test predictions for further analysis.

For the analyzed XGBoost predictions:

```text
False Positives: 461
False Negatives: 446
```

This error analysis is useful because different mistakes can have different business consequences in credit-risk applications.

---

# 💾 Model Serialization

The final calibrated model and optimized threshold are saved using `joblib`.

### Saved Model

```text
Credit_Risk_Model.pkl
```

### Saved Threshold

```text
Best_Threshold.pkl
```

These artifacts can later be loaded for inference without retraining the model.

Example:

```python
import joblib

model = joblib.load("Credit_Risk_Model.pkl")
threshold = joblib.load("Best_Threshold.pkl")
```

Prediction:

```python
probability = model.predict_proba(X_new)[:, 1]

prediction = (probability >= threshold).astype(int)
```

---

# 🧠 Machine Learning Workflow

The complete implementation follows this architecture:

```text
                    ┌──────────────────┐
                    │ Credit Risk Data │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       EDA        │
                    │ Missing Values   │
                    │ Class Balance    │
                    │ Correlation      │
                    │ Outliers         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Validation  │
                    │ Duplicates       │
                    │ Invalid Ages     │
                    │ Invalid Income   │
                    │ Invalid Loans    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Train/Test Split │
                    │   Stratified     │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌──────────────────────────┐
                 │ Class Imbalance Handling │
                 │   scale_pos_weight       │
                 └────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │    Preprocessing Pipeline │
                ├───────────────────────────┤
                │ Numerical → Imputation    │
                │ Categorical → One-Hot     │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │ Stratified 5-Fold CV      │
                └─────────────┬─────────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │ Logistic        │     │    XGBoost      │
        │ Regression      │     │                 │
        │ Baseline        │     │ Main Model      │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 │                       ▼
                 │              ┌─────────────────┐
                 │              │ Hyperparameter  │
                 │              │    Tuning       │
                 │              └────────┬────────┘
                 │                       │
                 │                       ▼
                 │              ┌─────────────────┐
                 │              │   Calibration   │
                 │              └────────┬────────┘
                 │                       │
                 │                       ▼
                 │              ┌─────────────────┐
                 │              │ Threshold       │
                 │              │ Optimization    │
                 │              └────────┬────────┘
                 │                       │
                 └───────────────┬───────┘
                                 ▼
                       ┌──────────────────┐
                       │ Model Evaluation │
                       └────────┬─────────┘
                                │
                ┌───────────────┼────────────────┐
                ▼               ▼                ▼
           SHAP Analysis   FP/FN Analysis   Saved Model
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Explainability

* SHAP

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* Google Colab

---

# 📁 Project Structure

Recommended GitHub repository structure:

```text
credit-risk-analysis/
│
├── 📓 ML Project - Credit Risk Analysis.ipynb
│
├── 📊 data/
│   └── credit_risk_dataset.csv
│
├── 🤖 models/
│   ├── Credit_Risk_Model.pkl
│   └── Best_Threshold.pkl
│
├── 📈 images/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   └── calibration_curve.png
│
├── 📄 README.md
│
└── 📜 requirements.txt
```

> Avoid committing large datasets or sensitive financial data directly to a public repository. If the dataset has redistribution restrictions, provide instructions for obtaining it instead.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/credit-risk-analysis.git
```

Navigate into the project:

```bash
cd credit-risk-analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
joblib
jupyter
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
ML Project - Credit Risk Analysis.ipynb
```

Run the notebook sequentially from data loading through model serialization.

---

# 🔮 Making Predictions

After training, the saved model can be loaded:

```python
import joblib

model = joblib.load("Credit_Risk_Model.pkl")
threshold = joblib.load("Best_Threshold.pkl")
```

Generate default probabilities:

```python
probability = model.predict_proba(X_new)[:, 1]
```

Apply the optimized threshold:

```python
prediction = (probability >= threshold).astype(int)
```

Output interpretation:

```text
0 → Lower predicted default class
1 → Higher predicted default class
```

The probability should be interpreted as a model output rather than a guaranteed statement about whether an applicant will default.

---

# 📊 Evaluation Metrics

The project uses several complementary metrics.

### Accuracy

Measures the overall proportion of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive/default cases were actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual positive/default cases were identified.

```text
Recall = TP / (TP + FN)
```

### F1-Score

Harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) /
     (Precision + Recall)
```

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Average Precision

Used during XGBoost hyperparameter optimization and is particularly informative for imbalanced classification problems.

---

# 💡 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Exploratory Data Analysis
* Data validation
* Duplicate detection
* Missing-value handling
* Outlier analysis
* Feature engineering/preparation
* Train-test splitting
* Stratification
* Class imbalance
* Class weighting
* Feature scaling
* One-hot encoding
* Pipelines
* ColumnTransformer
* Cross-validation
* Logistic Regression
* XGBoost
* Hyperparameter tuning
* RandomizedSearchCV
* Average Precision
* Probability calibration
* Classification threshold optimization
* SHAP explainability
* False-positive analysis
* False-negative analysis
* Model serialization

---

# ⚠️ Important Considerations

This project is intended as a **machine learning portfolio and educational project**.

A production credit-risk system would require considerably more work, including:

* Data governance
* Robust data validation
* Bias and fairness evaluation
* Model monitoring
* Drift detection
* Regulatory compliance
* Security controls
* Auditability
* Explainability requirements
* Independent model validation
* Proper probability calibration validation
* Business-specific cost-sensitive threshold selection
* Production deployment infrastructure

Therefore, model performance on this dataset should not be interpreted as evidence that the system is ready for real-world lending decisions.

---

# 🔬 Future Improvements

Potential improvements include:

### 1. Cost-Sensitive Threshold Optimization

Instead of optimizing only F1-score, define explicit business costs for:

```text
False Positive
False Negative
```

and select a threshold based on those costs.

### 2. Better Probability Evaluation

Add:

* Brier Score
* Expected Calibration Error
* Reliability diagrams
* Calibration comparison before/after tuning

### 3. Model Comparison

Experiment with additional classifiers:

```text
Random Forest
LightGBM
CatBoost
SVM
HistGradientBoosting
```

### 4. Advanced Feature Engineering

Potential features include:

```text
Income-to-loan ratio
Employment-to-age ratio
Credit-history-to-age ratio
Interest-rate categories
Loan amount categories
```

### 5. Production API

Deploy the model using:

```text
FastAPI
```

and expose an endpoint such as:

```text
POST /predict
```

### 6. Web Application

Build an interactive prediction interface using:

```text
Streamlit
```

### 7. MLOps

The project can be extended with:

```text
MLflow
Docker
GitHub Actions
Model Monitoring
CI/CD
```

---

# 📌 Project Highlights

### Data

```text
32,581 original records
12 features
Binary classification
Class imbalance
Missing values
Duplicate records
Invalid business values
```

### Modeling

```text
Logistic Regression Baseline
        ↓
XGBoost
        ↓
Randomized Hyperparameter Search
        ↓
Probability Calibration
        ↓
Threshold Optimization
```

### Explainability

```text
SHAP Global Explanation
SHAP Local Explanation
False Positive Analysis
False Negative Analysis
```

### Model Artifacts

```text
Credit_Risk_Model.pkl
Best_Threshold.pkl
```

---

# 🏆 Results Summary

The project establishes Logistic Regression as a baseline and then develops an XGBoost-based approach.

The cross-validation results were:

| Model               | ROC-AUC | Accuracy | Precision | Recall |     F1 |
| ------------------- | ------: | -------: | --------: | -----: | -----: |
| Logistic Regression |  0.8713 |   0.8110 |    0.5436 | 0.7792 | 0.6404 |
| XGBoost             |  0.9401 |   0.9114 |    0.7989 | 0.7886 | 0.7936 |

The tuned XGBoost search achieved an average-precision score of approximately:

```text
0.90
```

The final workflow additionally incorporates probability calibration and an optimized threshold of approximately:

```text
0.6064
```

for converting calibrated default probabilities into binary predictions.

---

# 👨‍💻 Author

**Muhammad Ibrahim**

Software Engineering | Machine Learning | AI

📍 Faisalabad, Pakistan

---

# ⭐ If You Found This Project Useful

If this project helped you understand credit-risk classification, machine learning pipelines, or model explainability, consider giving the repository a ⭐.

---

## 📄 License

This project is available under the MIT License.
