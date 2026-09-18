import pandas as pd

from src.preprocessing import engineer_features


def test_engineered_features_are_created():

    data = {
        "planned_budget_usd": [800000],
        "planned_duration_days": [90],
        "team_size": [15],
        "contractor_experience_years": [10],
        "complexity_rating": [6],
        "num_subcontractors": [4],
        "num_permits_required": [3],
    }

    df = pd.DataFrame(data)

    result = engineer_features(df)

    expected_features = [
        "budget_per_team_member",
        "team_density",
        "subcontractor_ratio",
        "permit_complexity",
        "experience_complexity_ratio",
        "budget_duration_ratio",
        "team_complexity_interaction",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_engineered_values():

    data = {
        "planned_budget_usd": [800000],
        "planned_duration_days": [90],
        "team_size": [15],
        "contractor_experience_years": [10],
        "complexity_rating": [6],
        "num_subcontractors": [4],
        "num_permits_required": [3],
    }

    df = pd.DataFrame(data)

    result = engineer_features(df)

    assert result["budget_per_team_member"].iloc[0] == 800000 / 15

    assert result["team_density"].iloc[0] == 15 / 90

    assert result["subcontractor_ratio"].iloc[0] == 4 / 15

    assert result["permit_complexity"].iloc[0] == 3 / 6

    assert result["experience_complexity_ratio"].iloc[0] == 10 / 6

    assert result["budget_duration_ratio"].iloc[0] == 800000 / 90

    assert result["team_complexity_interaction"].iloc[0] == 15 * 6