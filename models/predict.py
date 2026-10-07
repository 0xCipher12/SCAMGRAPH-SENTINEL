import joblib
import pandas as pd
import numpy as np


# ============================================================
# SCAMGRAPH SENTINEL - FRAUD PREDICTION
# ============================================================

MODEL_PATH = r".\models\fraud_model.joblib"

print("=" * 60)
print("SCAMGRAPH SENTINEL - FRAUD PREDICTION")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fraud detection model...")

model_data = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model keys:", list(model_data.keys()))


# Extract saved model components
weights = np.asarray(model_data["weights"], dtype=float)
mean = np.asarray(model_data["mean"], dtype=float)
std = np.asarray(model_data["std"], dtype=float)
features = list(model_data["features"])
threshold = float(model_data["threshold"])


print("Number of features:", len(features))
print("Number of weights :", len(weights))
print("Threshold         :", threshold)


# ============================================================
# CHECK MODEL DIMENSIONS
# ============================================================

print("\nChecking model dimensions...")

if len(weights) == len(features) + 1:
    print("Detected bias/intercept weight.")

    bias = weights[0]
    feature_weights = weights[1:]

elif len(weights) == len(features):
    print("No separate bias weight detected.")

    bias = 0.0
    feature_weights = weights

else:
    raise ValueError(
        "\nMODEL DIMENSION ERROR\n"
        f"Features = {len(features)}\n"
        f"Weights  = {len(weights)}\n"
        "The saved model is inconsistent."
    )


if len(mean) != len(features):
    raise ValueError(
        f"Mean has {len(mean)} values but "
        f"there are {len(features)} features."
    )


if len(std) != len(features):
    raise ValueError(
        f"Std has {len(std)} values but "
        f"there are {len(features)} features."
    )


if len(feature_weights) != len(features):
    raise ValueError(
        f"Feature weights = {len(feature_weights)}, "
        f"features = {len(features)}"
    )


print("Model dimensions OK!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_transaction(transaction):

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(transaction, dict):
        df = pd.DataFrame([transaction])

    elif isinstance(transaction, pd.DataFrame):
        df = transaction.copy()

    else:
        raise TypeError(
            "Transaction must be a dictionary "
            "or pandas DataFrame."
        )


    # --------------------------------------------------------
    # Add missing features
    # --------------------------------------------------------

    for feature in features:

        if feature not in df.columns:
            df[feature] = 0


    # --------------------------------------------------------
    # Keep exactly the training features
    # --------------------------------------------------------

    df = df[features]


    # --------------------------------------------------------
    # Convert values to numeric
    # --------------------------------------------------------

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X = df.to_numpy(dtype=float)


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    safe_std = np.where(
        std == 0,
        1,
        std
    )

    X_scaled = (
        X - mean
    ) / safe_std


    # --------------------------------------------------------
    # Calculate model score
    # --------------------------------------------------------

    score = (
        bias
        + np.dot(
            X_scaled,
            feature_weights
        )
    )


    # --------------------------------------------------------
    # Sigmoid → probability
    # --------------------------------------------------------

    probability = 1 / (
        1 + np.exp(
            -np.clip(
                score,
                -500,
                500
            )
        )
    )


    # --------------------------------------------------------
    # Fraud decision
    # --------------------------------------------------------

    prediction = (
        probability >= threshold
    ).astype(int)


    return (
        int(prediction[0]),
        float(probability[0])
    )


# ============================================================
# TEST TRANSACTION
# ============================================================

test_transaction = {

    "TX_AMOUNT": 250.00,

    "TX_TIME_SECONDS": 500,

    "TX_TIME_DAYS": 1,

    "HOUR": 14,

    "DAY_OF_WEEK": 2,

    "DAY_OF_MONTH": 6,

    "MONTH": 10,

    "IS_WEEKEND": 0,

    "IS_NIGHT": 0,

    "AMOUNT_LOG": np.log1p(250.00),

    "CUSTOMER_TX_COUNT": 5,

    "CUSTOMER_AVG_AMOUNT": 100.0,

    "CUSTOMER_MAX_AMOUNT": 300.0,

    "AMOUNT_TO_CUSTOMER_AVG": 2.5,

    "TERMINAL_TX_COUNT": 10,

    "TERMINAL_AVG_AMOUNT": 120.0,

    "AMOUNT_TO_TERMINAL_AVG": 2.08,

    "HIGH_AMOUNT": 0
}


# ============================================================
# RUN TEST
# ============================================================

print("\nTesting transaction...")


prediction, probability = predict_transaction(
    test_transaction
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)


if prediction == 1:

    print("Result       : FRAUD")

else:

    print("Result       : LEGITIMATE")


print(
    "Prediction   :",
    prediction
)

print(
    f"Fraud Score  : {probability:.6f}"
)

print(
    f"Fraud Chance : {probability * 100:.2f}%"
)

print(
    f"Threshold    : {threshold:.6f}"
)


print("=" * 60)
print("FRAUD PREDICTION COMPLETE")
print("=" * 60)