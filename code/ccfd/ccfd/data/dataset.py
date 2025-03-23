# ccfd/data/dataset.py
import os
import pandas as pd
import cudf
from sklearn.model_selection import train_test_split
from cuml.model_selection import train_test_split as cuml_train_test_split

def load_dataset(filepath: str, use_gpu: bool = False):
    """
    Loads the dataset from a CSV file using either pandas (CPU) or cuDF (GPU).

    Args:
        filepath (str): Path to the CSV file.
        use_gpu (bool): If True, loads the dataset using cuDF (GPU), otherwise uses pandas (CPU).
    
    Returns:
        pd.DataFrame or cudf.DataFrame: Loaded dataset (Pandas for CPU, cuDF for GPU), or None if loading fails.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        return None

    try:
        if use_gpu:
            print(f"🚀 Loading dataset using cuDF (GPU)...")
            df = cudf.read_csv(filepath)
        else:
            print(f"📄 Loading dataset using pandas (CPU)...")
            df = pd.read_csv(filepath)
        
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def prepare_data(df, target_column: str = "Class", use_gpu: bool = False):
    """
    Splits the dataset into training and test sets. Converts to cuDF if GPU is enabled.

    Args:
        df (pd.DataFrame): Input dataset (always loaded in pandas).
        target_column (str): Name of the target column.
        use_gpu (bool): If True, converts df to cuDF and uses cuML's train-test split.

    Returns:
        Tuple: (df_train, df_test) as pandas or cuDF DataFrames/Series.
    """
    if use_gpu:
        print("🚀 Converting dataset to cuDF for GPU acceleration...")

        if isinstance(df, pd.DataFrame):
            df = cudf.from_pandas(df)

        # Check that X and y are compatible with cuML
        X = df.drop(columns=[target_column]).astype("float32")
        y = df[target_column].astype("int32")

        # Stratify balances the fraud records in train and test data
        return cuml_train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        print("📄 Using pandas for CPU-based train-test split...")

        X = df.drop(columns=[target_column])  # Features
        y = df[target_column]  # Labels

        return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
def show_data_info(df: pd.DataFrame) -> None:
    """
    Prints dataset information and statistics.

    Args:
        df (pd.DataFrame): The dataset.
    """
    print("=== Dataset Info ===")
    print(df.info())
    print("\n=== Statistical Summary ===")
    print(df.describe())
