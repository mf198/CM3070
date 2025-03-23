# ccfd/data/preprocess.py
import pandas as pd
import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler as cuStandardScaler  # GPU
from sklearn.preprocessing import StandardScaler as skStandardScaler  # CPU

def clean_dataset_old(df, use_gpu=False):
    """
    Cleans the dataset by handling missing values, duplicates, and performing scaling.

    Args:
        df (pd.DataFrame or cudf.DataFrame): Raw dataset.
        use_gpu (bool): If True, process with cuDF (GPU), otherwise use pandas (CPU).

    Returns:
        pd.DataFrame or cudf.DataFrame: Cleaned dataset.
    """
    
    # Drop 'Time' column if present
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    # Scale 'Amount' feature (StandardScaler for CPU, MinMaxScaler for GPU)
    if use_gpu:
        df["Amount"] = GPUStandardScaler().fit_transform(df[["Amount"]])  # cuML MinMaxScaler (GPU)
    else:
        df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])  # StandardScaler (CPU)

    return df

def clean_dataset(df, use_gpu=False):
    """
    Cleans the dataset by handling missing values, duplicates, and performing scaling.

    Args:
        df (pd.DataFrame or cudf.DataFrame): Raw dataset.
        use_gpu (bool): If True, process with cuDF (GPU), otherwise use pandas (CPU).

    Returns:
        pd.DataFrame or cudf.DataFrame: Cleaned dataset.
    """
    
    # Drop 'Time' column if present
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    # Scale 'Amount' feature (GPU vs CPU)
    if "Amount" in df.columns:
        if use_gpu:
            scaler = cuStandardScaler()
            df["Amount"] = scaler.fit_transform(df[["Amount"]].astype("float32"))  # Ensure float32 for cuML
        else:
            scaler = skStandardScaler()
            df["Amount"] = scaler.fit_transform(df[["Amount"]])  # sklearn StandardScaler

    print("✅ Dataset cleaned successfully.")
    return df
