# ccfd/data/dataset.py
import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filepath)
    return df

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
