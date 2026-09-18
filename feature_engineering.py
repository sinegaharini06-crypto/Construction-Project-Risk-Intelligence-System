import pandas as pd
import numpy as np

DATA_PATH = "data/processed/construction_projects_features.csv"


PLANNING_FEATURES = [
    "project_type",
    "region",
    "planned_budget_usd",
    "planned_duration_days",
    "team_size",
    "contractor_experience_years",
    "complexity_rating",
    "num_subcontractors",
    "num_permits_required",
]


ENGINEERED_FEATURES = [
    "budget_per_team_member",
    "team_density",
    "subcontractor_ratio",
    "permit_complexity",
    "experience_complexity_ratio",
    "budget_duration_ratio",
    "team_complexity_interaction",
]


def engineer_features(df):
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
        df["contractor_experience_years"]
        / df["complexity_rating"]
    )

    df["budget_duration_ratio"] = (
        df["planned_budget_usd"]
        / df["planned_duration_days"]
    )

    df["team_complexity_interaction"] = (
        df["team_size"] * df["complexity_rating"]
    )

    return df


def main():
    df = pd.read_csv(DATA_PATH)

    df = engineer_features(df)

    print("Engineered features:")

    for feature in ENGINEERED_FEATURES:
        print(feature)

    print("\nCorrelation with risk_level cannot be calculated directly")
    print("because risk_level is categorical.")

    print("\nNaN values in engineered features:")

    print(df[ENGINEERED_FEATURES].isnull().sum())

    print("\nInfinite values:")

    print(
        np.isinf(
            df[ENGINEERED_FEATURES]
            .select_dtypes(include=np.number)
        ).sum()
    )

    print("\nFeature engineering completed.")


if __name__ == "__main__":
    main()