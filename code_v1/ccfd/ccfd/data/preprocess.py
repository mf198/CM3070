# ccfd/data/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

    pca = PCA(n_components=1)
    df["Amount_PCA"] = pca.fit_transform(df[["Amount"]])
    
    return df


