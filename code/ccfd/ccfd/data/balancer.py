import cudf
import cupy as cp
import numpy as np
import os
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from ccfd.data.gan_oversampler import (
    generate_synthetic_samples,
    train_gan,
    train_wgan,
    Generator,
    Discriminator,
)


def apply_smote(
    df: pd.DataFrame, target_column: str = "Class", use_gpu=False
) -> pd.DataFrame:
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


def apply_adasyn(
    df: pd.DataFrame, target_column: str = "Class", use_gpu=False
) -> pd.DataFrame:
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


def apply_svm_smote(
    df: pd.DataFrame, target_column: str = "Class", use_gpu=False
) -> pd.DataFrame:
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

    svm_smote = SVMSMOTE(random_state=42)
    X_resampled, y_resampled = svm_smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_column] = y_resampled

    # Convert back to cuDF if GPU is enabled
    return cudf.DataFrame(df_balanced) if use_gpu else df_balanced


###


def apply_gan_oversampling_old(
    df: pd.DataFrame,
    target_column="Class",
    use_gpu=False,
    model_path="ccfd/pretrained_models/pt_gan.pth",
):
    """
    Uses a GAN to oversample the minority class in a dataset. Supports both CPU and GPU.

    Args:
        df (pd.DataFrame): Input dataset.
        target_column (str): Target class column name.
        use_gpu (bool): If True, use GPU (cuDF), otherwise use CPU (pandas).
        model_path (str): Path to a pre-trained GAN model. If available, it will be used instead of training a new one.

    Returns:
        cudf.DataFrame or pd.DataFrame: Balanced dataset (depends on `use_gpu` flag).
    """
    print(f"Class count:\n{df[target_column].value_counts()}")

    # Identify class distribution
    class_counts = df[target_column].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Calculate the number of synthetic samples needed
    num_samples = class_counts[majority_class] - class_counts[minority_class]
    print(f"🔄 Generating {num_samples} synthetic samples to balance the dataset.")

    # Extract minority class samples
    X_minority = (
        df[df[target_column] == minority_class]
        .drop(columns=[target_column])
        .values.astype(np.float32)
    )

    # Set device (GPU or CPU)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Check if pre-trained GAN model exists
    if os.path.exists(model_path):
        print(f"📂 Loading pre-trained GAN model from: {model_path}")

        # Load saved model
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        best_params = checkpoint["params"]

        # Initialize and load Generator
        generator = Generator(best_params["latent_dim"], X_minority.shape[1]).to(device)
        generator.load_state_dict(checkpoint["generator"])
        generator.eval()  # Set to evaluation mode

    else:
        print("🚀 Training a new GAN for oversampling (no pre-trained model found)...")
        generator = train_gan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)

    # Generate synthetic samples
    synthetic_data = generate_synthetic_samples(generator, num_samples=num_samples)

    # Convert synthetic data to DataFrame
    synthetic_df = pd.DataFrame(
        synthetic_data, columns=df.drop(columns=[target_column]).columns
    )
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
    X_minority = (
        df[df[target_column] == minority_class]
        .drop(columns=[target_column])
        .values.astype(np.float32)
    )

    # Train WGAN and generate synthetic samples
    print("🚀 Training WGAN for oversampling...")
    generator = train_wgan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)
    synthetic_data = generate_synthetic_samples(generator, num_samples=num_samples)

    # Convert synthetic data to DataFrame
    synthetic_df = pd.DataFrame(
        synthetic_data, columns=df.drop(columns=[target_column]).columns
    )
    synthetic_df[target_column] = minority_class

    # Merge synthetic and original data
    df_augmented = pd.concat([df, synthetic_df], axis=0).reset_index(drop=True)

    # Convert back to cuDF if GPU is enabled
    return cudf.DataFrame(df_augmented) if use_gpu else df_augmented


import os
import torch
import cudf
import numpy as np
import pandas as pd


import os
import torch
import cudf
import cupy as cp
import numpy as np
import pandas as pd

def apply_gan_oversampling(
    X_train, y_train, use_gpu=False, model_path="ccfd/pretrained_models/pt_gan.pth"
):
    """
    Uses a GAN to oversample the minority class in a dataset. Supports both CPU and GPU.

    Args:
        X_train (numpy.ndarray, cudf.DataFrame, or pandas.DataFrame): Input features.
        y_train (numpy.ndarray, cudf.Series, or pandas.Series): Target labels.
        use_gpu (bool): If True, use GPU (cuDF), otherwise use CPU (pandas).
        model_path (str): Path to a pre-trained GAN model. If available, it will be used instead of training a new one.

    Returns:
        tuple: (X_balanced, y_balanced) - The oversampled dataset (compatible with GPU or CPU).
    """

    # Use CuPy for GPU processing, NumPy for CPU
    if use_gpu and isinstance(y_train, cudf.Series):
        y_train_cupy = cp.asarray(y_train.to_cupy())
        unique_classes, class_counts = cp.unique(y_train_cupy, return_counts=True)
        unique_classes, class_counts = unique_classes.get(), class_counts.get()
    else:
        unique_classes, class_counts = np.unique(y_train, return_counts=True)

    class_distribution = dict(zip(unique_classes, class_counts))
    print(f"Class distribution before oversampling: {class_distribution}")

    # Identify the minority and majority class
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Calculate the number of synthetic samples needed
    num_samples = class_counts.max() - class_counts.min()
    print(f"🔄 Generating {num_samples} synthetic samples to balance the dataset.")

    # Extract minority class samples
    minority_mask = y_train == minority_class
    X_minority = X_train[minority_mask].astype(np.float32)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert data to tensors if needed
    if use_gpu:
        X_minority = torch.tensor(X_minority.to_cupy(), dtype=torch.float32, device=device)
    else:
        X_minority = torch.tensor(X_minority.to_numpy(), dtype=torch.float32, device=device)

    # Check if pre-trained GAN model exists
    if os.path.exists(model_path):
        print(f"📂 Loading pre-trained GAN model from: {model_path}")

        # Load saved model
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        best_params = checkpoint["params"]

        # Initialize and load Generator
        generator = Generator(best_params["latent_dim"], X_minority.shape[1]).to(device)
        generator.load_state_dict(checkpoint["generator"])
        generator.eval()  # Set to evaluation mode

    else:
        print("🚀 Training a new GAN for oversampling (no pre-trained model found)...")
        generator = train_gan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)

    # Generate synthetic samples using optimized function
    synthetic_data = generate_synthetic_samples(generator, num_samples=num_samples, use_gpu=use_gpu)

    # Create labels for synthetic samples
    if use_gpu:
        y_synthetic = cudf.Series(cp.full(num_samples, minority_class))
    else:
        y_synthetic = pd.Series(np.full(num_samples, minority_class))

    # Merge synthetic and original data efficiently
    if use_gpu:
        X_balanced = cudf.DataFrame(cp.vstack((X_train.to_cupy(), synthetic_data)))
        y_balanced = cudf.concat([y_train, y_synthetic])
    else:
        X_balanced = pd.DataFrame(np.vstack((X_train, synthetic_data)), columns=X_train.columns)
        y_balanced = pd.concat([y_train, y_synthetic], ignore_index=True)   

    return X_balanced, y_balanced
