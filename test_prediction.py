from src.predict import predict_risk


def sample_project():

    return {
        "project_type": "Commercial",
        "region": "South",
        "planned_budget_usd": 800000,
        "planned_duration_days": 90,
        "team_size": 15,
        "contractor_experience_years": 10,
        "complexity_rating": 6,
        "num_subcontractors": 4,
        "num_permits_required": 3,
    }


def test_prediction_returns_risk_level():

    result = predict_risk(sample_project())

    assert "risk_level" in result

    assert result["risk_level"] in [
        "Low",
        "Medium",
        "High"
    ]


def test_prediction_returns_probabilities():

    result = predict_risk(sample_project())

    assert "probabilities" in result

    probabilities = result["probabilities"]

    assert "Low" in probabilities
    assert "Medium" in probabilities
    assert "High" in probabilities


def test_probabilities_sum_to_one():

    result = predict_risk(sample_project())

    probabilities = result["probabilities"]

    total = sum(probabilities.values())

    assert abs(total - 1.0) < 0.000001