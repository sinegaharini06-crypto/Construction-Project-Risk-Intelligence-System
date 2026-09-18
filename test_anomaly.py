from src.anomaly import detect_anomaly


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


def test_anomaly_returns_status():

    result = detect_anomaly(sample_project())

    assert "is_anomalous" in result

    assert "status" in result

    assert result["status"] in [
        "NORMAL",
        "ANOMALOUS"
    ]


def test_anomaly_returns_score():

    result = detect_anomaly(sample_project())

    assert "anomaly_score" in result

    assert isinstance(
        result["anomaly_score"],
        float
    )