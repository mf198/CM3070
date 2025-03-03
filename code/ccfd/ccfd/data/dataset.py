# ccfd/data/dataset.py
import os
import pandas as pd
import cudf

def load_dataset_cpu(filepath: str):
    """
    Loads the dataset from a CSV file using pandas (CPU).

    Args:
        filepath (str): Path to the CSV file.        
    
    Returns:
        pd.DataFrame: Loaded dataset or None if loading fails.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        return None

    try:
        print(f"📄 Loading dataset using pandas (CPU)...")
        df = pd.read_csv(filepath)
        
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


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
