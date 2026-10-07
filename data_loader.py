from pathlib import Path
import pandas as pd

# Existing raw dataset location
RAW_DATA_DIR = Path(
    r"C:\hackathons\simulated-data-raw-main\simulated-data-raw-main\data"
)


def get_files():
    """Return all PKL dataset files."""
    files = sorted(RAW_DATA_DIR.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(
            f"No .pkl files found in: {RAW_DATA_DIR}"
        )

    return files


def load_file(file_path):
    """Load one PKL file."""
    return pd.read_pickle(file_path)


def load_dataset():
    """Load all PKL files into one DataFrame."""
    files = get_files()

    print(f"Found {len(files)} dataset files.")

    frames = []

    for i, file_path in enumerate(files, 1):
        print(f"Loading {i}/{len(files)}: {file_path.name}")
        frames.append(load_file(file_path))

    df = pd.concat(frames, ignore_index=True)

    print(f"\nTotal transactions: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFraud distribution:")
    print(df["TX_FRAUD"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())