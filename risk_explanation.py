import joblib
import os


MODEL_PATH = "models/final_risk_model.joblib"
SHAP_PATH = "models/shap_explanations.joblib"


def main():

    print("=== Risk Explanation Validation ===")

    if not os.path.exists(MODEL_PATH):
        print("Final model not found.")
        return

    if not os.path.exists(SHAP_PATH):
        print("SHAP artifact not found.")
        return

    model = joblib.load(MODEL_PATH)
    shap_artifact = joblib.load(SHAP_PATH)

    print("\nFinal model loaded successfully.")

    print(
        "Model type:",
        type(model).__name__
    )

    print("\nSHAP artifact loaded successfully.")

    print(
        "Classes:",
        shap_artifact.get("class_order")
    )

    feature_names = shap_artifact.get(
        "feature_names",
        []
    )

    print(
        "Number of transformed features:",
        len(feature_names)
    )

    print("\nImportant limitation:")

    print(
        "The saved SHAP artifact contains explanations "
        "for previously evaluated projects."
    )

    print(
        "It should not be reused as a live explanation "
        "for an arbitrary new project."
    )

    print(
        "\nLive SHAP generation requires a compatible "
        "SHAP/Numba environment."
    )


if __name__ == "__main__":
    main()