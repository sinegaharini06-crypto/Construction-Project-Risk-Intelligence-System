import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import os


DATA_PATH = "data/processed/construction_projects_features.csv"

REPORT_PATH = "reports/stage9_tuning_comparison.csv"


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


def create_preprocessor(scale=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer([
        (
            "numeric",
            Pipeline(numeric_steps),
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


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    high_mask = y_test == "High"
    predicted_high = predictions == "High"

    high_precision = precision_score(
        high_mask,
        predicted_high,
        zero_division=0
    )

    high_recall = recall_score(
        high_mask,
        predicted_high,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "High Precision": high_precision,
        "High Recall": high_recall,
        "Macro F1": macro_f1
    }


def main():

    df = pd.read_csv(DATA_PATH)

    from src.preprocessing import engineer_features

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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {

        "LogReg Tuned": Pipeline([
            (
                "preprocessor",
                create_preprocessor(scale=True)
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42
                )
            )
        ]),

        "RF Tuned": Pipeline([
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1
                )
            )
        ]),

        "GB Tuned": Pipeline([
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.85,
                    random_state=42
                )
            )
        ])
    }

    results = []

    for name, model in models.items():

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro"
        )

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        result["CV Macro F1 Mean"] = cv_scores.mean()

        results.append(result)

        print("\n", name)
        print("CV Macro F1:", round(cv_scores.mean(), 3))
        print("Test Macro F1:", round(result["Macro F1"], 3))
        print("High Recall:", round(result["High Recall"], 3))

    os.makedirs("reports", exist_ok=True)

    pd.DataFrame(results).to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nSaved:", REPORT_PATH)


if __name__ == "__main__":
    main()