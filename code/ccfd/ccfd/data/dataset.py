# ccfd/data/dataset.py
import os
import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None

    try:
        df = pd.read_csv(filepath)        
        return df
    except Exception as e:
        print(f"Error loading file with pandas: {e}")
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
