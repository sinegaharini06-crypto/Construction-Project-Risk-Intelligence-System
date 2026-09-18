import pandas as pd


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


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)

    feature_columns = PLANNING_FEATURES + [
        "budget_per_team_member",
        "team_density",
        "subcontractor_ratio",
        "permit_complexity",
        "experience_complexity_ratio",
        "budget_duration_ratio",
        "team_complexity_interaction",
    ]

    return df[feature_columns]