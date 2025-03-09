import numpy as np
import cudf
import cupy as cp
import pandas as pd

def to_numpy_safe(data, select_column=None):
    """
    Converts input data (cuDF DataFrame, cuDF Series, CuPy array, or NumPy array) to a NumPy array.
    
    Args:
        data: Input data, which can be:
            - cuDF.DataFrame
            - cuDF.Series
            - pd.Series
            - CuPy array (cp.ndarray)
            - NumPy array (np.ndarray)
        select_column (int, optional): If `data` is a DataFrame with multiple columns, 
                                       this specifies which column to extract (default: None).

    Returns:
        np.ndarray: The converted NumPy array.
    """
    if isinstance(data, cudf.DataFrame):
        return data.to_pandas().to_numpy()
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()    
    elif isinstance(data, cudf.Series):
        return data.to_pandas().to_numpy()
    elif isinstance(data, pd.Series):
        return data.to_numpy()    
    elif isinstance(data, cp.ndarray):
        return data.get()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
