# ccfd/data/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values, duplicates, and performing other cleaning steps.

    Args:
        df (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    
    # Dropping useless columns    
    df = df.drop(columns=['Time'])

    # Apply scaling to the 'Amount' feature
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

    # Additional cleaning logic can be added here...
    return df