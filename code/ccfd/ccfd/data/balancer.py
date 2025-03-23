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


def apply_smote(X, y, use_gpu=False) -> tuple:
    """
    Balances the dataset using SMOTE on the given training fold.

    Args:
        X (pd.DataFrame or cudf.DataFrame): The training features for the current fold.
        y (pd.Series or cudf.Series): The training labels for the current fold.
        use_gpu (bool): Whether to use GPU (converts back to cuDF if True).

    Returns:
        tuple: (X_resampled, y_resampled) - The oversampled training data (format matches input).
    """
    smote = SMOTE(sampling_strategy="minority", random_state=42)

    # Convert cuDF to pandas for SMOTE (if needed)
    is_cudf = isinstance(X, cudf.DataFrame)
    if is_cudf:
        X, y = X.to_pandas(), y.to_pandas()

    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert back to cuDF if GPU is enabled
    if use_gpu and is_cudf:
        return cudf.DataFrame(X_resampled, columns=X.columns), cudf.Series(y_resampled)
    else:
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


###


def apply_adasyn(X, y, use_gpu=False) -> tuple:
    """
    Balances the dataset using ADASYN on the given training fold.

    Args:
        X (pd.DataFrame or cudf.DataFrame): The training features for the current fold.
        y (pd.Series or cudf.Series): The training labels for the current fold.
        use_gpu (bool): Whether to use GPU (converts back to cuDF if True).

    Returns:
        tuple: (X_resampled, y_resampled) - The oversampled training data (format matches input).
    """
    adasyn = ADASYN(random_state=42)

    # Convert cuDF to pandas for ADASYN (if needed)
    is_cudf = isinstance(X, cudf.DataFrame)
    if is_cudf:
        X, y = X.to_pandas(), y.to_pandas()

    # Apply ADASYN
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # Convert back to cuDF if GPU is enabled
    if use_gpu and is_cudf:
        return cudf.DataFrame(X_resampled, columns=X.columns), cudf.Series(y_resampled)
    else:
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
###


def apply_svm_smote(X, y, use_gpu=False) -> tuple:
    """
    Balances the dataset using SVM-SMOTE on the given training fold.

    Args:
        X (pd.DataFrame or cudf.DataFrame): The training features for the current fold.
        y (pd.Series or cudf.Series): The training labels for the current fold.
        use_gpu (bool): Whether to use GPU (converts back to cuDF if True).

    Returns:
        tuple: (X_resampled, y_resampled) - The oversampled training data (format matches input).
    """
    svm_smote = SVMSMOTE(random_state=42)

    # Convert cuDF to pandas for SVM-SMOTE (if needed)
    is_cudf = isinstance(X, cudf.DataFrame)
    if is_cudf:
        X, y = X.to_pandas(), y.to_pandas()

    # Apply SVM-SMOTE
    X_resampled, y_resampled = svm_smote.fit_resample(X, y)

    # Convert back to cuDF if GPU is enabled
    if use_gpu and is_cudf:
        return cudf.DataFrame(X_resampled, columns=X.columns), cudf.Series(y_resampled)
    else:
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
###


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

    # Identify the minority and majority class
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Calculate the number of synthetic samples needed
    num_samples = class_counts.max() - class_counts.min()    

    # Extract minority class samples
    minority_mask = y_train == minority_class
    X_minority = X_train[minority_mask].astype(np.float32)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert data to tensors if needed
    if use_gpu:
        X_minority = torch.tensor(
            X_minority.to_cupy(), dtype=torch.float32, device=device
        )
    else:
        X_minority = torch.tensor(
            X_minority.to_numpy(), dtype=torch.float32, device=device
        )

    # Use a pre-trained GAN model if it exists
    if os.path.exists(model_path):        

        # Load saved model
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        best_params = checkpoint["params"]

        # Initialize and load Generator
        generator = Generator(best_params["latent_dim"], X_minority.shape[1]).to(device)
        generator.load_state_dict(checkpoint["generator"])
        generator.eval()  # Set to evaluation mode

    else:
        print(" Training a new GAN for oversampling (no pre-trained model found)...")
        generator = train_gan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)

    # Generate synthetic samples using optimized function
    synthetic_data = generate_synthetic_samples(
        generator, num_samples=num_samples, use_gpu=use_gpu
    )

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
        X_balanced = pd.DataFrame(
            np.vstack((X_train, synthetic_data)), columns=X_train.columns
        )
        y_balanced = pd.concat([y_train, y_synthetic], ignore_index=True)

    return X_balanced, y_balanced
###


def apply_wgan_oversampling(
    X_train, y_train, use_gpu=False, model_path="ccfd/pretrained_models/pt_wgan.pth"
):
    """
    Uses a WGAN to oversample the minority class in a dataset. Supports both CPU and GPU.

    Args:
        X_train (numpy.ndarray, cudf.DataFrame, or pandas.DataFrame): Input features.
        y_train (numpy.ndarray, cudf.Series, or pandas.Series): Target labels.
        use_gpu (bool): If True, use GPU (cuDF), otherwise use CPU (pandas).
        model_path (str): Path to a pre-trained WGAN model. If available, it will be used instead of training a new one.

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

    # Identify the minority and majority class
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Calculate the number of synthetic samples needed
    num_samples = class_counts.max() - class_counts.min()

    # Extract minority class samples
    minority_mask = y_train == minority_class
    X_minority = X_train[minority_mask].astype(np.float32)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert data to tensors if needed
    if use_gpu:
        X_minority = torch.tensor(
            X_minority.to_cupy(), dtype=torch.float32, device=device
        )
    else:
        X_minority = torch.tensor(
            X_minority.to_numpy(), dtype=torch.float32, device=device
        )

    # Check if pre-trained WGAN model exists
    if os.path.exists(model_path):
        # Load saved model
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        best_params = checkpoint["params"]

        # Initialize and load Generator
        generator = Generator(best_params["latent_dim"], X_minority.shape[1]).to(device)
        generator.load_state_dict(checkpoint["generator"])
        generator.eval()  # Set to evaluation mode

    else:
        print("Training a new WGAN for oversampling (no pre-trained model found)...")
        generator = train_wgan(X_minority, num_epochs=500, latent_dim=10, batch_size=32)

    # Generate synthetic samples using optimized function
    synthetic_data = generate_synthetic_samples(
        generator, num_samples=num_samples, use_gpu=use_gpu
    )

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
        X_balanced = pd.DataFrame(
            np.vstack((X_train, synthetic_data)), columns=X_train.columns
        )
        y_balanced = pd.concat([y_train, y_synthetic], ignore_index=True)

    return X_balanced, y_balanced
###
