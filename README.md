# Construction Project Risk Intelligence

A machine learning system for assessing construction project risk using planning-stage project information.

The system predicts three risk levels:

- Low
- Medium
- High

It also provides risk probabilities and detects unusual project planning patterns using Isolation Forest.

## Project Overview

Construction projects can face different risks related to budget, duration, project complexity, team size, contractor experience, permits, and subcontractors.

This project uses machine learning to assess construction project risk using information available during the planning stage.

The system includes:

- Data cleaning and preprocessing
- Feature engineering
- Machine learning model comparison
- Hyperparameter tuning
- Random Forest risk classification
- Isolation Forest anomaly detection
- SHAP-based model explainability
- Error analysis
- Streamlit web application

### Important Scope

This project performs early risk assessment using planning-stage information.

The target variable is a retrospectively defined historical risk outcome. Therefore, the system should be interpreted as an early risk assessment model, not as proof of forecasting an independently observed future risk event.

---

## Key Features

### 1. Risk Classification

The system predicts:

- Low Risk
- Medium Risk
- High Risk

The final selected model is a tuned Random Forest Classifier.

### 2. Risk Probabilities

The system provides probability estimates for each risk level.

Example:

Low: 0.15  
Medium: 0.24  
High: 0.61

This provides more information than showing only the predicted class.

### 3. Anomaly Detection

An Isolation Forest model is used separately from the risk classifier.

It identifies projects with unusual planning-stage feature patterns.

Output:

- Normal
- Anomalous

Risk classification and anomaly detection are separate signals.

### 4. Explainability

SHAP is used to understand how different features contributed to individual model predictions.

SHAP explanations describe model contribution and should not be interpreted as causal relationships.

### 5. Error Analysis

The final model was evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1
- ROC-AUC
- Confusion patterns
- High-risk errors
- Prediction confidence
- Feature importance

---

## Machine Learning Pipeline

Planning-stage project information

↓

Data Preprocessing

↓

Feature Engineering

↓

Tuned Random Forest

↓

Risk Prediction

↓

Low / Medium / High

↓

Risk Probabilities


Separate branch:

Planning Features

↓

Isolation Forest

↓

Normal / Anomalous


Explainability:

Final Random Forest

↓

SHAP

↓

Feature Contributions

---

## Input Features

The application uses the following planning-stage inputs:

| Feature | Description |
|---|---|
| Project Type | Type of construction project |
| Region | Project region |
| Planned Budget | Planned project budget in USD |
| Planned Duration | Expected project duration in days |
| Team Size | Planned project team size |
| Contractor Experience | Contractor experience in years |
| Complexity Rating | Project complexity rating |
| Permits Required | Number of permits required |
| Subcontractors | Number of subcontractors |

---

## Feature Engineering

Seven additional features were created from the planning-stage variables.

### Budget per Team Member

planned_budget_usd / team_size

### Team Density

team_size / planned_duration_days

### Subcontractor Ratio

num_subcontractors / team_size

### Permit Complexity

num_permits_required / complexity_rating

### Experience-Complexity Ratio

contractor_experience_years / complexity_rating

### Budget-Duration Ratio

planned_budget_usd / planned_duration_days

### Team-Complexity Interaction

team_size × complexity_rating

These engineered features use only planning-stage information.

---

## Data Preparation

The dataset contains 1,200 cleaned project records.

Data preparation included:

- Duplicate detection
- Duplicate removal
- Missing-value inspection
- Data-type validation
- Feature selection
- Target validation

Missing values were handled during model preprocessing using training-data-fitted preprocessing steps.

This helps prevent information from the test set influencing preprocessing.

---

## Data Leakage Prevention

Several variables in the original dataset describe project outcomes rather than planning-stage information.

The following variables were excluded from the model inputs:

- Actual Cost
- Cost Overrun
- Actual Duration
- Schedule Slippage
- Safety Incidents
- Rework Cost
- Material Cost Variance
- Labour Overtime
- Weather Delay
- Inspection Failures
- Change Orders

Project ID was also excluded from model features.

The final model uses 16 planning-stage and engineered features.

---

## Model Development

Several machine learning approaches were evaluated during development.

### Logistic Regression

Used as a baseline linear classification model.

### Random Forest

Used as a nonlinear tree-based classification model.

### Gradient Boosting

Used as another nonlinear classification approach.

### Hyperparameter Tuning

The models were tuned using training data and cross-validation.

The held-out test set was not used for hyperparameter selection.

### Hybrid Model

Soft-voting combinations of the tuned models were also tested.

The tested hybrid models did not improve the final hard-classification performance over the selected Random Forest.

Therefore, the final application uses the tuned Random Forest.

---

## Final Model

The final risk classifier is a tuned Random Forest Classifier.

Main configuration:

- n_estimators = 300
- max_depth = 8
- min_samples_split = 5
- min_samples_leaf = 4
- max_features = sqrt
- class_weight = balanced
- random_state = 42

The model artifact contains the preprocessing and model pipeline required for inference.

---

## Evaluation

The final model was evaluated on a held-out test set containing 240 projects.

Final results:

| Metric | Result |
|---|---:|
| Accuracy | 65.83% |
| Macro F1 | 56.59% |
| High-risk Precision | 34.48% |
| High-risk Recall | 45.45% |

Macro F1 is emphasized because the dataset contains imbalanced risk classes.

Target distribution:

| Risk Level | Percentage |
|---|---:|
| Medium | 63.4% |
| Low | 27.3% |
| High | 9.2% |

Therefore, accuracy alone does not fully describe model performance.

---

## Feature Importance

Feature importance was investigated using permutation importance and SHAP.

The strongest recurring features included:

1. Experience-complexity ratio
2. Contractor experience
3. Team-complexity interaction
4. Complexity rating

These results describe model behavior.

They do not establish causal relationships between the features and construction risk.

---

## Error Analysis

The final test set contained:

- 240 projects
- 158 correct predictions
- 82 incorrect predictions

A major source of error was confusion between Medium and High risk.

Several historical High-risk projects were classified as Medium.

The analysis found that some missed High-risk projects had planning profiles resembling Medium-risk projects, particularly with lower complexity and lower team-complexity interaction values than High-risk projects that were successfully identified.

This indicates that some retrospective High-risk outcomes cannot be reliably distinguished using the available planning-stage features.

### Model Limitation

The model does not reliably identify every High-risk project from planning-stage information alone.

Therefore, the system should be treated as a decision-support tool rather than a replacement for professional construction risk management.

---

## Anomaly Detection

An Isolation Forest model was trained separately using planning-stage features.

On the held-out test set:

- Normal projects: 211
- Anomalous projects: 29

Anomaly detection is not equivalent to risk classification.

A project can be both:

- High Risk and Normal
- High Risk and Anomalous
- Medium Risk and Normal
- Medium Risk and Anomalous
- Low Risk and Normal
- Low Risk and Anomalous

This allows the system to provide two separate signals:

Risk Level

+

Anomaly Status

---

## Explainability

The project includes SHAP-based model explanation.

SHAP values can be used to identify features that contributed toward or away from a predicted class.

For example:

Predicted Risk: High

Possible contributing features may include:

- Higher project complexity
- Lower experience-complexity ratio
- Higher team-complexity interaction

These are model contributions and should not be interpreted as causal effects.

---

## Streamlit Application

The project includes a lightweight Streamlit web application.

Users can enter:

- Project type
- Region
- Planned budget
- Planned duration
- Team size
- Contractor experience
- Complexity rating
- Permits required
- Number of subcontractors

The application returns:

- Risk Level
- Risk Probabilities
- Anomaly Status

---

## Project Structure

```text
Construction-Project-Risk-Intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   └── processed/
│       └── construction_projects_features.csv
│
├── experiments/
│   ├── anomaly_detection.py
│   ├── baseline_models.py
│   ├── data_cleaning.py
│   ├── error_analysis.py
│   ├── feature_engineering.py
│   ├── final_validation.py
│   ├── gradient_boosting.py
│   ├── hybrid_model.py
│   ├── model_tuning.py
│   ├── random_forest.py
│   └── risk_explanation.py
│
├── models/
│   ├── final_risk_model.joblib
│   └── isolation_forest.joblib
│
├── src/
│   ├── README.md
│   ├── anomaly.py
│   ├── explain.py
│   ├── predict.py
│   └── preprocessing.py
│
└── tests/
    ├── test_anomaly.py
    ├── test_explanation.py
    ├── test_prediction.py
    └── test_preprocessing.py

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Joblib
- Streamlit
- Matplotlib
- Seaborn

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/sinegaharini06-crypto/Construction-Project-Risk-Intelligence.git
