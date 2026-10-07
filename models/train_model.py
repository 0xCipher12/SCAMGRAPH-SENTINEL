import os
import joblib
import numpy as np
import pandas as pd


print("=" * 65)
print("SCAMGRAPH SENTINEL - NUMPY FRAUD MODEL TRAINING")
print("=" * 65)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "engineered_features.pkl"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fraud_model.joblib"
)

METRICS_PATH = os.path.join(
    ARTIFACT_DIR,
    "model_metrics.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading engineered dataset...")

df = pd.read_pickle(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# TARGET
# ============================================================

if "TX_FRAUD" not in df.columns:
    raise ValueError("TX_FRAUD column not found!")

y = df["TX_FRAUD"].to_numpy(
    dtype=np.float32
)

X_df = df.drop(
    columns=["TX_FRAUD"]
)


# ============================================================
# KEEP NUMERIC FEATURES ONLY
# ============================================================

X_df = X_df.select_dtypes(
    include=["number", "bool"]
)

print("\nFeature columns:")

for column in X_df.columns:
    print(" -", column)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = X_df.to_numpy(
    dtype=np.float32
)

print("\nFeature matrix:", X.shape)


# ============================================================
# CLEAN VALUES
# ============================================================

X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

rng = np.random.default_rng(42)

indices = np.arange(len(X))

rng.shuffle(indices)

split = int(len(indices) * 0.80)

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nStandardizing features...")

mean = X_train.mean(axis=0)

std = X_train.std(axis=0)

std[std < 1e-8] = 1.0

X_train = (
    (X_train - mean) / std
)

X_test = (
    (X_test - mean) / std
)


# ============================================================
# ADD BIAS
# ============================================================

X_train = np.column_stack(
    [
        np.ones(len(X_train), dtype=np.float32),
        X_train
    ]
)

X_test = np.column_stack(
    [
        np.ones(len(X_test), dtype=np.float32),
        X_test
    ]
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining NumPy Logistic Regression...")


def sigmoid(z):

    z = np.clip(z, -30, 30)

    return 1.0 / (
        1.0 + np.exp(-z)
    )


weights = np.zeros(
    X_train.shape[1],
    dtype=np.float32
)


# Fraud is rare, so give fraud examples higher weight
positive_count = max(
    np.sum(y_train == 1),
    1
)

negative_count = max(
    np.sum(y_train == 0),
    1
)

positive_weight = (
    negative_count / positive_count
)

print(
    "Fraud class weight:",
    round(float(positive_weight), 2)
)


# ============================================================
# MINI-BATCH TRAINING
# ============================================================

learning_rate = 0.03

epochs = 12

batch_size = 16384

n = len(X_train)

print("\nEpochs:", epochs)
print("Batch size:", batch_size)


for epoch in range(epochs):

    order = rng.permutation(n)

    X_train_epoch = X_train[order]
    y_train_epoch = y_train[order]

    total_loss = 0.0

    batches = 0

    for start in range(
        0,
        n,
        batch_size
    ):

        end = min(
            start + batch_size,
            n
        )

        xb = X_train_epoch[start:end]

        yb = y_train_epoch[start:end]

        predictions = sigmoid(
            xb @ weights
        )

        # Class weights
        sample_weights = np.where(
            yb == 1,
            positive_weight,
            1.0
        )

        error = (
            predictions - yb
        ) * sample_weights

        gradient = (
            xb.T @ error
        ) / len(xb)

        weights -= (
            learning_rate * gradient
        )

        # Loss
        eps = 1e-7

        loss = -(
            yb * np.log(
                predictions + eps
            )
            +
            (1 - yb) * np.log(
                1 - predictions + eps
            )
        )

        total_loss += float(
            np.mean(loss)
        )

        batches += 1

    average_loss = (
        total_loss / batches
    )

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"- loss: {average_loss:.6f}"
    )


print("\nTraining complete!")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nEvaluating model...")

probabilities = sigmoid(
    X_test @ weights
)

# Default probability threshold
threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(np.int32)


# ============================================================
# METRICS
# ============================================================

tp = np.sum(
    (predictions == 1)
    &
    (y_test == 1)
)

tn = np.sum(
    (predictions == 0)
    &
    (y_test == 0)
)

fp = np.sum(
    (predictions == 1)
    &
    (y_test == 0)
)

fn = np.sum(
    (predictions == 0)
    &
    (y_test == 1)
)


accuracy = (
    (tp + tn)
    /
    max(len(y_test), 1)
)

precision = (
    tp
    /
    max(tp + fp, 1)
)

recall = (
    tp
    /
    max(tp + fn, 1)
)

f1 = (
    2 * precision * recall
    /
    max(precision + recall, 1e-9)
)


print("\n" + "=" * 65)
print("MODEL RESULTS")
print("=" * 65)

print("\nConfusion Matrix:")

print(
    f"True Negative : {tn:,}"
)

print(
    f"False Positive: {fp:,}"
)

print(
    f"False Negative: {fn:,}"
)

print(
    f"True Positive : {tp:,}"
)

print("\nMetrics:")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "weights": weights,

    "mean": mean,

    "std": std,

    "features": list(
        X_df.columns
    ),

    "threshold": threshold

}


joblib.dump(
    model_package,
    MODEL_PATH
)


print("\nModel saved successfully!")

print(
    "Location:",
    MODEL_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SCAMGRAPH SENTINEL - MODEL RESULTS\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        f"Dataset shape: {df.shape}\n"
    )

    f.write(
        f"Training rows: {len(X_train):,}\n"
    )

    f.write(
        f"Testing rows: {len(X_test):,}\n\n"
    )

    f.write(
        "Features:\n"
    )

    for feature in X_df.columns:

        f.write(
            f"- {feature}\n"
        )

    f.write("\nMetrics:\n")

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {f1:.4f}\n"
    )

    f.write("\nConfusion Matrix:\n")

    f.write(
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
    )


print(
    "Metrics saved:",
    METRICS_PATH
)


print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETE")
print("=" * 65)