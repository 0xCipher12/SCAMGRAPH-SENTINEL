from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os


# ============================================================
# SCAMGRAPH SENTINEL
# FRAUD DETECTION API
# ============================================================

app = Flask(
    __name__,
    template_folder="dashboard",
    static_folder="dashboard"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_model.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SCAMGRAPH SENTINEL - FRAUD DETECTION API")
print("=" * 70)

print("\nLoading trained fraud detection model...")

try:

    model_data = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")
    print("Model type:", type(model_data))

    MODEL_FEATURES = list(
        model_data["features"]
    )

    print("\nModel features:")

    for i, feature in enumerate(
        MODEL_FEATURES,
        start=1
    ):
        print(
            f"{i:02d}. {feature}"
        )

    print(
        f"\nTotal model features: "
        f"{len(MODEL_FEATURES)}"
    )

except Exception as e:

    print(
        "\nERROR loading model:"
    )

    print(str(e))

    model_data = None
    MODEL_FEATURES = []


# ============================================================
# SAFE NUMBER CONVERSION
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (
        ValueError,
        TypeError
    ):

        return default


def safe_int(
    value,
    default=0
):

    try:

        return int(float(value))

    except (
        ValueError,
        TypeError
    ):

        return default


# ============================================================
# FEATURE BUILDER
# ============================================================

def build_features(data):

    """
    Convert dashboard JSON into the exact
    17 features expected by the trained model.
    """

    # ========================================================
    # BASIC TRANSACTION FEATURES
    # ========================================================

    amount = safe_float(
        data.get(
            "TX_AMOUNT",
            data.get("amount", 0)
        )
    )


    tx_time_seconds = safe_float(
        data.get(
            "TX_TIME_SECONDS",
            data.get("tx_time", 100)
        )
    )


    tx_time_days = safe_float(
        data.get(
            "TX_TIME_DAYS",
            data.get("tx_days", 0)
        )
    )


    hour = safe_int(
        data.get(
            "HOUR",
            data.get("hour", 12)
        )
    )


    day_of_week = safe_int(
        data.get(
            "DAY_OF_WEEK",
            data.get("day", 0)
        )
    )


    day_of_month = safe_int(
        data.get(
            "DAY_OF_MONTH",
            data.get("day_month", 1)
        )
    )


    month = safe_int(
        data.get(
            "MONTH",
            data.get("month", 1)
        )
    )


    # ========================================================
    # VALIDATE BASIC RANGES
    # ========================================================

    hour = max(
        0,
        min(23, hour)
    )

    day_of_week = max(
        0,
        min(6, day_of_week)
    )

    day_of_month = max(
        1,
        min(31, day_of_month)
    )

    month = max(
        1,
        min(12, month)
    )


    # ========================================================
    # TIME RISK FEATURES
    # ========================================================

    calculated_weekend = int(
        day_of_week >= 5
    )

    calculated_night = int(
        hour < 6 or hour >= 22
    )


    # If frontend explicitly sends the values,
    # use those values. Otherwise calculate them.

    is_weekend = safe_int(
        data.get(
            "IS_WEEKEND",
            calculated_weekend
        )
    )

    is_night = safe_int(
        data.get(
            "IS_NIGHT",
            calculated_night
        )
    )


    # ========================================================
    # CUSTOMER BEHAVIOR
    # ========================================================

    customer_tx_count = safe_float(
        data.get(
            "CUSTOMER_TX_COUNT",
            data.get(
                "customer_transaction_count",
                1
            )
        )
    )


    customer_avg_amount = safe_float(
        data.get(
            "CUSTOMER_AVG_AMOUNT",
            data.get(
                "customer_average_amount",
                amount
            )
        )
    )


    customer_max_amount = safe_float(
        data.get(
            "CUSTOMER_MAX_AMOUNT",
            data.get(
                "customer_maximum_amount",
                amount
            )
        )
    )


    # ========================================================
    # CUSTOMER AMOUNT RATIO
    # ========================================================

    amount_to_customer_avg = (
        amount /
        max(
            customer_avg_amount,
            1e-6
        )
    )


    # ========================================================
    # TERMINAL BEHAVIOR
    # ========================================================

    terminal_tx_count = safe_float(
        data.get(
            "TERMINAL_TX_COUNT",
            data.get(
                "terminal_transaction_count",
                1
            )
        )
    )


    terminal_avg_amount = safe_float(
        data.get(
            "TERMINAL_AVG_AMOUNT",
            data.get(
                "terminal_average_amount",
                amount
            )
        )
    )


    # ========================================================
    # TERMINAL AMOUNT RATIO
    # ========================================================

    amount_to_terminal_avg = (
        amount /
        max(
            terminal_avg_amount,
            1e-6
        )
    )


    # ========================================================
    # HIGH AMOUNT
    # ========================================================

    # IMPORTANT:
    #
    # During training, HIGH_AMOUNT was generated using
    # the 95th percentile of the complete dataset.
    #
    # For the dashboard, the user can explicitly select
    # the High Amount indicator.
    #
    # If the frontend does not send it, we use a simple
    # fallback of amount >= 1000.

    high_amount = safe_int(
        data.get(
            "HIGH_AMOUNT",
            data.get(
                "high_amount",
                int(amount >= 1000)
            )
        )
    )


    # ========================================================
    # EXACT 17 MODEL FEATURES
    # ========================================================

    features = {

        "TX_AMOUNT":
            amount,

        "TX_TIME_SECONDS":
            tx_time_seconds,

        "TX_TIME_DAYS":
            tx_time_days,

        "HOUR":
            hour,

        "DAY_OF_WEEK":
            day_of_week,

        "DAY_OF_MONTH":
            day_of_month,

        "MONTH":
            month,

        "IS_WEEKEND":
            is_weekend,

        "IS_NIGHT":
            is_night,

        "CUSTOMER_TX_COUNT":
            customer_tx_count,

        "CUSTOMER_AVG_AMOUNT":
            customer_avg_amount,

        "CUSTOMER_MAX_AMOUNT":
            customer_max_amount,

        "AMOUNT_TO_CUSTOMER_AVG":
            amount_to_customer_avg,

        "TERMINAL_TX_COUNT":
            terminal_tx_count,

        "TERMINAL_AVG_AMOUNT":
            terminal_avg_amount,

        "AMOUNT_TO_TERMINAL_AVG":
            amount_to_terminal_avg,

        "HIGH_AMOUNT":
            high_amount
    }


    return features


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_transaction(data):

    if model_data is None:

        raise ValueError(
            "Fraud detection model is not loaded."
        )


    # ========================================================
    # LOAD MODEL PARAMETERS
    # ========================================================

    weights = np.asarray(
        model_data["weights"],
        dtype=float
    )


    mean = np.asarray(
        model_data["mean"],
        dtype=float
    )


    std = np.asarray(
        model_data["std"],
        dtype=float
    )


    features = list(
        model_data["features"]
    )


    threshold = float(
        model_data.get(
            "threshold",
            0.5
        )
    )


    # ========================================================
    # BUILD FEATURE DICTIONARY
    # ========================================================

    feature_values = build_features(
        data
    )


    # ========================================================
    # BUILD VECTOR IN EXACT MODEL ORDER
    # ========================================================

    values = []

    for feature in features:

        value = feature_values.get(
            feature,
            0.0
        )

        values.append(
            safe_float(value)
        )


    X = np.asarray(
        values,
        dtype=float
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(X) != len(mean):

        raise ValueError(
            "Feature count mismatch: "
            f"model expects {len(mean)}, "
            f"received {len(X)}."
        )


    if len(X) != len(std):

        raise ValueError(
            "Standard deviation vector "
            "does not match feature count."
        )


    # ========================================================
    # CLEAN VALUES
    # ========================================================

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # ========================================================
    # STANDARDIZATION
    # ========================================================

    std_safe = np.where(
        std == 0,
        1.0,
        std
    )


    X_scaled = (
        X - mean
    ) / std_safe


    # ========================================================
    # MODEL SCORE
    # ========================================================

    # Your training script adds a bias column before training.
    #
    # Therefore the saved weight vector can contain:
    #
    #   bias + 17 feature weights
    #
    # Handle both possible formats safely.

    if len(weights) == len(features) + 1:

        bias = float(
            weights[0]
        )

        feature_weights = (
            weights[1:]
        )

        score = float(
            bias +
            np.dot(
                X_scaled,
                feature_weights
            )
        )

    elif len(weights) == len(features):

        score = float(
            np.dot(
                X_scaled,
                weights
            )
        )

    else:

        raise ValueError(
            "Model weight count mismatch: "
            f"model has {len(weights)} weights "
            f"but {len(features)} features."
        )


    # ========================================================
    # SIGMOID PROBABILITY
    # ========================================================

    score_clipped = np.clip(
        score,
        -30,
        30
    )


    probability = (
        1.0 /
        (
            1.0 +
            np.exp(
                -score_clipped
            )
        )
    )


    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    prediction = int(
        probability >= threshold
    )


    result = (
        "FRAUD"
        if prediction == 1
        else "LEGITIMATE"
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success":
            True,

        "prediction":
            prediction,

        "result":
            result,

        "fraud_probability":
            round(
                float(
                    probability
                ),
                6
            ),

        "fraud_percentage":
            round(
                float(
                    probability * 100
                ),
                2
            ),

        "threshold":
            round(
                threshold,
                4
            ),

        "score":
            round(
                score,
                6
            ),

        "features_used":
            feature_values
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy",

        "model_loaded":
            model_data is not None,

        "feature_count":
            len(MODEL_FEATURES),

        "model_features":
            MODEL_FEATURES

    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # READ JSON
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success":
                    False,

                "error":
                    "No JSON data received."

            }), 400


        print("\n" + "=" * 70)

        print(
            "NEW TRANSACTION ANALYSIS"
        )

        print("=" * 70)


        print(
            "\nIncoming data:"
        )


        for key, value in data.items():

            print(
                f"{key}: {value}"
            )


        # ----------------------------------------------------
        # RUN MODEL
        # ----------------------------------------------------

        result = predict_transaction(
            data
        )


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            "\nPrediction:",
            result["result"]
        )

        print(
            "Probability:",
            result["fraud_percentage"],
            "%"
        )

        print(
            "Score:",
            result["score"]
        )

        print(
            "Threshold:",
            result["threshold"]
        )

        print(
            "=" * 70
        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "\nPrediction error:"
        )

        print(
            str(e)
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting SCAMGRAPH SENTINEL..."
    )

    print(
        "Dashboard:"
        " http://127.0.0.1:5000"
    )

    print(
        "Health:"
        " http://127.0.0.1:5000/health"
    )

    print(
        "Prediction API:"
        " http://127.0.0.1:5000/predict"
    )

    print(
        "\nPress CTRL+C to stop the server."
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )