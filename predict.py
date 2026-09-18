import joblib
import pandas as pd

MODEL_PATH = "models/final_risk_model.joblib"

model = joblib.load(MODEL_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["budget_per_team_member"] = (
        df["planned_budget_usd"] / df["team_size"]
    )

    df["team_density"] = (
        df["team_size"] / df["planned_duration_days"]
    )

    df["subcontractor_ratio"] = (
        df["num_subcontractors"] / df["team_size"]
    )

    df["permit_complexity"] = (
        df["num_permits_required"] / df["complexity_rating"]
    )

    df["experience_complexity_ratio"] = (
        df["contractor_experience_years"] / df["complexity_rating"]
    )

    df["budget_duration_ratio"] = (
        df["planned_budget_usd"] / df["planned_duration_days"]
    )

    df["team_complexity_interaction"] = (
        df["team_size"] * df["complexity_rating"]
    )

    return df


def predict_risk(project_data: dict) -> dict:
    df = pd.DataFrame([project_data])

    df = engineer_features(df)

    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]

    class_names = model.classes_

    probability_dict = {
        str(class_name): float(probability)
        for class_name, probability in zip(
            class_names,
            probabilities
        )
    }

    return {
        "risk_level": str(prediction),
        "probabilities": probability_dict,
    }