import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest


from src.preprocessing import engineer_features


DATA_PATH = "data/processed/construction_projects_features.csv"


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


NUMERIC_FEATURES = FEATURES[2:]

CATEGORICAL_FEATURES = [
    "project_type",
    "region"
]


def main():

    df = pd.read_csv(DATA_PATH)

    df = engineer_features(df)

    X = df[FEATURES]
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            CATEGORICAL_FEATURES
        )
    ])

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "isolation_forest",
            IsolationForest(
                contamination=0.10,
                n_estimators=300,
                random_state=42,
                n_jobs=1
            )
        )
    ])

    model.fit(X_train)

    predictions = model.predict(X_test)

    anomaly_status = predictions == -1

    result = pd.DataFrame({
        "risk_level": y_test.values,
        "anomaly": anomaly_status
    })

    print("\n=== Anomaly Detection ===")

    print(
        result["anomaly"]
        .value_counts()
        .rename({
            False: "Normal",
            True: "Anomalous"
        })
    )

    print("\nAnomaly rate by risk level:")

    print(
        result.groupby("risk_level")["anomaly"]
        .mean()
        .round(3)
    )


if __name__ == "__main__":
    main()