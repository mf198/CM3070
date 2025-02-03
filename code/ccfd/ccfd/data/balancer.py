# ccfd/data/balancer.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE

def apply_smote(df: pd.DataFrame, target_column: str = "Class") -> pd.DataFrame:
    """
    Balances the dataset using SMOTE.

    Args:
        df (pd.DataFrame): The input dataset.
        target_column (str): The column containing class labels.

    Returns:
        pd.DataFrame: A balanced dataset.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    smote = SMOTE(random_state=18)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled

    return df_balanced

def apply_adasyn(df: pd.DataFrame, target_column: str = "Class") -> pd.DataFrame:
    """
    Balances the dataset using ADASYN.

    Args:
        df (pd.DataFrame): The input dataset.
        target_column (str): The column containing class labels.

    Returns:
        pd.DataFrame: A balanced dataset.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    adasyn = ADASYN(random_state=18)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled

    return df_balanced

def apply_svm_smote(df: pd.DataFrame, target_column: str = "Class") -> pd.DataFrame:
    """
    Balances the dataset using SVM-SMOTE.

    Args:
        df (pd.DataFrame): The input dataset.
        target_column (str): The column containing class labels.

    Returns:
        pd.DataFrame: A balanced dataset.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    svm_smote = SVMSMOTE(random_state=18)
    X_resampled, y_resampled = svm_smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled

    return df_balanced
