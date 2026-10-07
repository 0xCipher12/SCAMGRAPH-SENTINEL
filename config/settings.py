from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATA_DIR = BASE_DIR / "data"

# ML directories
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVALUATION_DIR = BASE_DIR / "evaluation"

# Dataset configuration
TARGET_COLUMN = "TX_FRAUD"

# Reproducibility
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.20

# Fraud detection
CLASS_WEIGHT = "balanced"