# ccfd/data/balancer.py
import cudf
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from ccfd.data.gan_oversampler import train_gan, train_wgan, generate_synthetic_samples

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

def apply_gan_oversampling(df, target_column="Class", num_samples=100000):
    """
    Uses a GAN to oversample the minority class in a dataset.
    
    - Automatically detects the minority class.
    - Trains a GAN and generates synthetic samples.
    - Works with cuDF but uses Pandas for GAN processing.
    
    :param df: cuDF or Pandas DataFrame.
    :param target_column: The name of the target (class) column.
    :param num_samples: Number of synthetic samples to generate.
    :return: Augmented cuDF DataFrame with synthetic samples.
    """
    # Ensure df is converted to Pandas for processing
    if isinstance(df, cudf.DataFrame):
        df = df.to_pandas()

    # Detect the minority class
    minority_class = df[target_column].value_counts().idxmin()
    print(f"Detected Minority Class: {minority_class}")

    # Extract minority class samples
    X_minority = df[df[target_column] == minority_class].drop(columns=[target_column]).values.astype(np.float32)

    # Train a GAN on the minority samples
    print("Training GAN for oversampling...")
    generator = train_gan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)

    # Generate synthetic samples
    synthetic_data = generate_synthetic_samples(generator, num_samples=num_samples)

    # Convert synthetic data into a DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.drop(columns=[target_column]).columns)
    synthetic_df[target_column] = minority_class  # Assign class label

    # Append synthetic data to original DataFrame
    df_augmented = pd.concat([df, synthetic_df], axis=0).reset_index(drop=True)

    print(f"Added {num_samples} synthetic samples for class {minority_class}.")
    return cudf.DataFrame(df_augmented)  # Convert back to cuDF for RAPIDS compatibility


def apply_wgan_oversampling(df, target_column="Class", num_samples=100000):
    """
    Uses a WGAN to oversample the minority class in a dataset.
    
    - Detects the minority class automatically
    - Trains a WGAN and generates synthetic samples
    - Works with cuDF for RAPIDS GPU acceleration
    
    :param df: cuDF or Pandas DataFrame.
    :param target_column: The name of the target (class) column.
    :param num_samples: Number of synthetic samples to generate.
    :return: Augmented cuDF DataFrame with synthetic samples.
    """
    if isinstance(df, cudf.DataFrame):
        df = df.to_pandas()  # Convert to Pandas for processing

    # Detect minority class
    minority_class = df[target_column].value_counts().idxmin()
    print(f"Detected Minority Class: {minority_class}")

    # Extract minority class samples
    X_minority = df[df[target_column] == minority_class].drop(columns=[target_column]).values.astype(np.float32)

    # Train WGAN on minority samples
    print("Training WGAN for oversampling...")
    generator = train_wgan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)

    # Generate synthetic samples
    synthetic_data = generate_synthetic_samples(generator, num_samples=num_samples)

    # Convert synthetic data into DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.drop(columns=[target_column]).columns)
    synthetic_df[target_column] = minority_class  # Assign class label

    # Append synthetic data to original DataFrame
    df_augmented = pd.concat([df, synthetic_df], axis=0).reset_index(drop=True)

    print(f"Added {num_samples} synthetic samples for class {minority_class}.")
    return cudf.DataFrame(df_augmented)  # Convert back to cuDF for RAPIDS compatibility

