import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.base import clone

from experiments.model_tuning import (
    create_preprocessor,
    FEATURES
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.pipeline import Pipeline

from src.preprocessing import engineer_features


DATA_PATH = "data/processed/construction_projects_features.csv"


def build_models():

    lr = Pipeline([
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
    ])

    rf = Pipeline([
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
    ])

    gb = Pipeline([
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

    return lr, rf, gb


def evaluate(name, y_test, predictions):

    print(f"\n=== {name} ===")

    print(
        "Accuracy:",
        round(
            accuracy_score(y_test, predictions),
            3
        )
    )

    print(
        "Macro F1:",
        round(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ),
            3
        )
    )

    high_true = y_test == "High"
    high_pred = predictions == "High"

    print(
        "High Precision:",
        round(
            precision_score(
                high_true,
                high_pred,
                zero_division=0
            ),
            3
        )
    )

    print(
        "High Recall:",
        round(
            recall_score(
                high_true,
                high_pred,
                zero_division=0
            ),
            3
        )
    )


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

    lr, rf, gb = build_models()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    lr_prob = lr.predict_proba(X_test)
    rf_prob = rf.predict_proba(X_test)
    gb_prob = gb.predict_proba(X_test)

    classes = rf.classes_

    weights = np.array([
        0.32,
        0.35,
        0.33
    ])

    hybrid_prob = (
        weights[0] * lr_prob +
        weights[1] * rf_prob +
        weights[2] * gb_prob
    )

    hybrid_prediction = classes[
        np.argmax(
            hybrid_prob,
            axis=1
        )
    ]

    evaluate(
        "Hybrid Weighted",
        y_test,
        hybrid_prediction
    )


if __name__ == "__main__":
    main()