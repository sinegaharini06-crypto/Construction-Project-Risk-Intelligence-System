import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.preprocessing import engineer_features


DATA_PATH = "data/processed/construction_projects_features.csv"
MODEL_PATH = "models/final_risk_model.joblib"


FEATURES = [
    "project_type",
    "region",
    "planned_budget_usd",
    "planned_duration_days",
    "team_size",
    "contractor_experience_years",
    "complexity_rating",
    "num_subcontractors",
    "num_permits_required",
    "budget_per_team_member",
    "team_density",
    "subcontractor_ratio",
    "permit_complexity",
    "experience_complexity_ratio",
    "budget_duration_ratio",
    "team_complexity_interaction",
]


def main():

    print("===================================")
    print(" FINAL MODEL VALIDATION")
    print("===================================")

    # -----------------------------
    # Check dataset
    # -----------------------------

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            DATA_PATH
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:", df.shape)

    # -----------------------------
    # Check project_id leakage
    # -----------------------------

    if "project_id" in FEATURES:
        print(
            "WARNING: project_id is included!"
        )
    else:
        print(
            "PASS: project_id excluded."
        )

    # -----------------------------
    # Feature engineering
    # -----------------------------

    df = engineer_features(df)

    X = df[FEATURES]
    y = df["risk_level"]

    # -----------------------------
    # Same final split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Load frozen model
    # -----------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            MODEL_PATH
        )

    model = joblib.load(MODEL_PATH)

    print(
        "\nFrozen model loaded:",
        type(model).__name__
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    predictions = model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    high_true = y_test == "High"
    high_pred = predictions == "High"

    high_precision = precision_score(
        high_true,
        high_pred,
        zero_division=0
    )

    high_recall = recall_score(
        high_true,
        high_pred,
        zero_division=0
    )

    print("\nFinal metrics:")

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Macro F1:",
        round(macro_f1, 4)
    )

    print(
        "High Precision:",
        round(high_precision, 4)
    )

    print(
        "High Recall:",
        round(high_recall, 4)
    )

    # -----------------------------
    # Expected final results
    # -----------------------------

    print("\nExpected approximate results:")

    print("Accuracy: 0.6583")
    print("Macro F1: 0.5659")
    print("High Precision: ~0.3448")
    print("High Recall: 0.4545")

    # -----------------------------
    # Feature check
    # -----------------------------

    outcome_features = [
        "actual_cost_usd",
        "cost_overrun_pct",
        "actual_duration_days",
        "schedule_slippage_pct",
        "safety_incidents",
        "rework_cost_pct",
        "material_cost_variance_pct",
        "labour_overtime_hours",
        "weather_delay_days",
        "inspection_failures",
        "num_change_orders"
    ]

    leakage_features_present = [
        feature
        for feature in outcome_features
        if feature in FEATURES
    ]

    print("\nLeakage feature check:")

    if leakage_features_present:
        print(
            "WARNING:",
            leakage_features_present
        )
    else:
        print(
            "PASS: Outcome/post-event features excluded."
        )

    print("\n===================================")
    print(" VALIDATION COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()