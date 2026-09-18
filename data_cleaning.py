import pandas as pd
import os


INPUT_PATH = "data/processed/construction_projects_features.csv"


def clean_dataset():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)

    # Remove exact duplicate rows
    duplicate_count = df.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    df = df.drop_duplicates().reset_index(drop=True)

    print("Shape after removing duplicates:", df.shape)

    # Display missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Display target distribution
    if "risk_level" in df.columns:
        print("\nRisk distribution:")
        print(df["risk_level"].value_counts())

        print("\nRisk distribution (%):")
        print(
            df["risk_level"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

    # Save cleaned dataset
    df.to_csv(INPUT_PATH, index=False)

    print("\nCleaned dataset saved successfully.")
    print("Final shape:", df.shape)


if __name__ == "__main__":
    clean_dataset()