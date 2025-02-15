import cudf
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from ccfd.data.gan_oversampler import generate_synthetic_samples, train_gan, train_wgan
import cupy as cp

def apply_smote(df: pd.DataFrame, target_column: str = "Class", use_gpu=False) -> pd.DataFrame:
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

    smote = SMOTE(sampling_strategy="minority", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled

    # Convert back to cuDF if GPU is enabled
    return cudf.DataFrame(df_balanced) if use_gpu else df_balanced
###


def apply_adasyn(df: pd.DataFrame, target_column: str = "Class", use_gpu=False) -> pd.DataFrame:
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

    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled

    # Convert back to cuDF if GPU is enabled
    return cudf.DataFrame(df_balanced) if use_gpu else df_balanced
###

def apply_svm_smote(df: pd.DataFrame, target_column: str = "Class", use_gpu=False) -> pd.DataFrame:
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
    
    # Convert back to cuDF if GPU is enabled
    return cudf.DataFrame(df_balanced) if use_gpu else df_balanced
###

def apply_gan_oversampling(df: pd.DataFrame, target_column="Class", use_gpu=False):
    """
    Uses a GAN to oversample the minority class in a dataset. Supports both CPU and GPU.

    Args:
        df (pd.DataFrame): Input dataset.
        target_column (str): Target class column name.
        num_samples (int): Number of synthetic samples to generate.
        use_gpu (bool): If True, use GPU (cuDF), otherwise use CPU (pandas).

    Returns:
        cudf.DataFrame or pd.DataFrame: Balanced dataset (depends on `use_gpu` flag).
    """

    print(f"Target: {target_column}")
    print(f"Class count: {df["Class"].value_counts()}")

    # Identify class distribution
    class_counts = df[target_column].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    # Calculate the number of synthetic samples needed
    num_samples = class_counts[majority_class] - class_counts[minority_class]
    print(f"🔄 Generating {num_samples} synthetic samples to balance the dataset.")

    # Extract minority class samples
    X_minority = df[df[target_column] == minority_class].drop(columns=[target_column]).values.astype(np.float32)

    # Train GAN and generate synthetic samples
    print("🚀 Training GAN for oversampling...")
    generator = train_gan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)
    synthetic_data = generate_synthetic_samples(generator, num_samples=num_samples)

    # Convert synthetic data to DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.drop(columns=[target_column]).columns)
    synthetic_df[target_column] = minority_class  

    # Merge synthetic and original data
    df_augmented = pd.concat([df, synthetic_df], axis=0).reset_index(drop=True)

    # Convert back to cuDF if GPU is enabled
    return cudf.DataFrame(df_augmented) if use_gpu else df_augmented


def apply_wgan_oversampling(df: pd.DataFrame, target_column="Class", use_gpu=False):
    """
    Uses a WGAN to oversample the minority class in a dataset. Supports both CPU and GPU.

    Args:
        df (cudf.DataFrame or pd.DataFrame): Input dataset.
        target_column (str): Target class column name.
        num_samples (int): Number of synthetic samples to generate.
        use_gpu (bool): If True, use GPU (cuDF + CuPy), otherwise use CPU (pandas + NumPy).

    Returns:
        torch.Tensor: Balanced dataset compatible with PyTorch.
    """

    # Identify class distribution
    class_counts = df[target_column].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()  

    # Calculate the number of synthetic samples needed
    num_samples = class_counts[majority_class] - class_counts[minority_class]
    print(f"🔄 Generating {num_samples} synthetic samples to balance the dataset.")

    # Extract minority class samples
    X_minority = df[df[target_column] == minority_class].drop(columns=[target_column]).values.astype(np.float32)

    # Train WGAN and generate synthetic samples
    print("🚀 Training WGAN for oversampling...")
    generator = train_wgan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)
    synthetic_data = generate_synthetic_samples(generator, num_samples=num_samples)

    # Convert synthetic data to DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.drop(columns=[target_column]).columns)
    synthetic_df[target_column] = minority_class  

    # Merge synthetic and original data
    df_augmented = pd.concat([df, synthetic_df], axis=0).reset_index(drop=True)

    # Convert back to cuDF if GPU is enabled
    return cudf.DataFrame(df_augmented) if use_gpu else df_augmented
