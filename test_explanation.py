import os
import joblib


SHAP_PATH = "models/shap_explanations.joblib"


def test_shap_artifact_exists():

    assert os.path.exists(SHAP_PATH)


def test_shap_artifact_can_be_loaded():

    artifact = joblib.load(SHAP_PATH)

    assert isinstance(
        artifact,
        dict
    )


def test_shap_artifact_contains_required_fields():

    artifact = joblib.load(SHAP_PATH)

    assert "feature_names" in artifact

    assert "class_order" in artifact

    assert "shap_values" in artifact

    assert "explainer_expected_value" in artifact