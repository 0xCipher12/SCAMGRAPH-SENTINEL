import sys
import types
import pandas as pd

# Compatibility layer for old pandas pickle files
numeric_module = types.ModuleType("pandas.core.indexes.numeric")

numeric_module.Int64Index = pd.Index
numeric_module.Float64Index = pd.Index
numeric_module.NumericIndex = pd.Index

sys.modules["pandas.core.indexes.numeric"] = numeric_module
import os
import glob
import pickle
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "simulated-data-raw-main",
    "simulated-data-raw-main",
    "data"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ALL TRANSACTION FILES
# ============================================================

def load_all_data():

    files = sorted(
        glob.glob(os.path.join(RAW_DATA_DIR, "*.pkl"))
    )

    if not files:
        raise FileNotFoundError(
            f"No .pkl files found in:\n{RAW_DATA_DIR}"
        )

    print("=" * 60)
    print(f"Found {len(files)} dataset files")
    print("=" * 60)

    frames = []

    for i, file in enumerate(files, 1):

        print(f"Loading {i}/{len(files)}: {os.path.basename(file)}")

        with open(file, "rb") as f:
            df = pickle.load(f)

        frames.append(df)

    print("\nCombining datasets...")

    df = pd.concat(
        frames,
        ignore_index=True
    )

    print(f"Total transactions: {len(df):,}")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    print("\nStarting feature engineering...")

    df = df.copy()

    # --------------------------------------------------------
    # DATETIME
    # --------------------------------------------------------

    df["TX_DATETIME"] = pd.to_datetime(
        df["TX_DATETIME"]
    )

    df["HOUR"] = df["TX_DATETIME"].dt.hour

    df["DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek

    df["DAY_OF_MONTH"] = df["TX_DATETIME"].dt.day

    df["MONTH"] = df["TX_DATETIME"].dt.month

    # Weekend

    df["IS_WEEKEND"] = (
        df["DAY_OF_WEEK"] >= 5
    ).astype(int)

    # Night transactions

    df["IS_NIGHT"] = (
        (df["HOUR"] < 6) |
        (df["HOUR"] >= 22)
    ).astype(int)


    # --------------------------------------------------------
    # CUSTOMER BEHAVIOR
    # --------------------------------------------------------

    customer_stats = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .agg(
            CUSTOMER_TX_COUNT="count",
            CUSTOMER_AVG_AMOUNT="mean",
            CUSTOMER_MAX_AMOUNT="max"
        )
        .reset_index()
    )

    df = df.merge(
        customer_stats,
        on="CUSTOMER_ID",
        how="left"
    )

    # Amount compared with customer's average

    df["AMOUNT_TO_CUSTOMER_AVG"] = (
        df["TX_AMOUNT"] /
        (df["CUSTOMER_AVG_AMOUNT"] + 1e-6)
    )


    # --------------------------------------------------------
    # TERMINAL BEHAVIOR
    # --------------------------------------------------------

    terminal_stats = (
        df.groupby("TERMINAL_ID")["TX_AMOUNT"]
        .agg(
            TERMINAL_TX_COUNT="count",
            TERMINAL_AVG_AMOUNT="mean"
        )
        .reset_index()
    )

    df = df.merge(
        terminal_stats,
        on="TERMINAL_ID",
        how="left"
    )

    # Amount compared with terminal average

    df["AMOUNT_TO_TERMINAL_AVG"] = (
        df["TX_AMOUNT"] /
        (df["TERMINAL_AVG_AMOUNT"] + 1e-6)
    )


    # --------------------------------------------------------
    # HIGH VALUE TRANSACTION
    # --------------------------------------------------------

    amount_threshold = df["TX_AMOUNT"].quantile(0.95)

    df["HIGH_AMOUNT"] = (
        df["TX_AMOUNT"] >= amount_threshold
    ).astype(int)


    # --------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------------

    numeric_columns = [
        "TX_AMOUNT",
        "TX_TIME_SECONDS",
        "TX_TIME_DAYS",
        "CUSTOMER_TX_COUNT",
        "CUSTOMER_AVG_AMOUNT",
        "CUSTOMER_MAX_AMOUNT",
        "AMOUNT_TO_CUSTOMER_AVG",
        "TERMINAL_TX_COUNT",
        "TERMINAL_AVG_AMOUNT",
        "AMOUNT_TO_TERMINAL_AVG"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df[numeric_columns] = (
        df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


    # --------------------------------------------------------
    # FINAL FEATURE DATASET
    # --------------------------------------------------------

    feature_columns = [

        "TX_AMOUNT",
        "TX_TIME_SECONDS",
        "TX_TIME_DAYS",

        "HOUR",
        "DAY_OF_WEEK",
        "DAY_OF_MONTH",
        "MONTH",

        "IS_WEEKEND",
        "IS_NIGHT",

        "CUSTOMER_TX_COUNT",
        "CUSTOMER_AVG_AMOUNT",
        "CUSTOMER_MAX_AMOUNT",
        "AMOUNT_TO_CUSTOMER_AVG",

        "TERMINAL_TX_COUNT",
        "TERMINAL_AVG_AMOUNT",
        "AMOUNT_TO_TERMINAL_AVG",

        "HIGH_AMOUNT",

        "TX_FRAUD"
    ]

    feature_columns = [
        col
        for col in feature_columns
        if col in df.columns
    ]

    df_features = df[feature_columns].copy()


    # --------------------------------------------------------
    # FINAL CLEANUP
    # --------------------------------------------------------

    df_features = df_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df_features = df_features.fillna(0)


    print("\nOriginal shape:")
    print(df.shape)

    print("\nFeature dataset shape:")
    print(df_features.shape)

    print("\nFeatures:")

    for column in df_features.columns:
        print(f" + {column}")


    print("\nFraud distribution:")

    print(
        df_features["TX_FRAUD"]
        .value_counts()
    )


    print("\nSample:")

    print(
        df_features.head()
    )


    return df_features


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df_features):

    print("\nSaving engineered dataset...")

    # Pickle does NOT require pyarrow/fastparquet
    output_path = os.path.join(
        OUTPUT_DIR,
        "engineered_features.pkl"
    )

    with open(output_path, "wb") as f:
        pickle.dump(
            df_features,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print("\nDataset saved successfully!")

    print(
        f"Location:\n{output_path}"
    )

    print(
        f"Size: {len(df_features):,} rows"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SCAMGRAPH SENTINEL - FEATURE ENGINEERING")
    print("=" * 60)

    df = load_all_data()

    df_features = engineer_features(df)

    save_dataset(df_features)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)