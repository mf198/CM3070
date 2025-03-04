import cudf
import optuna
import torch
import pandas as pd
import cupy as cp
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ccfd.data.gan_oversampler import Generator, Discriminator


def objective_gan(trial, X_train, y_train, use_gpu=False):
    """
    Optuna objective function to optimize GAN hyperparameters.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (numpy.ndarray or cupy.ndarray): Training dataset.
        use_gpu (bool): Whether to use GPU.

    Returns:
        float: The best loss value (minimized).
    """
    # Hyperparameters to optimize
    num_epochs = trial.suggest_int("num_epochs", 500, 5000, step=500)
    latent_dim = trial.suggest_int("latent_dim", 10, 100)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr_g = trial.suggest_float("lr_g", 1e-5, 1e-2, log=True)
    lr_d = trial.suggest_float("lr_d", 1e-5, 1e-2, log=True)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]

    # Initialize a unique step counter before training
    global_step = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in skf.split(X_train.to_numpy(), y_train.to_numpy()):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]

        # Convert to GPU or CPU tensors
        if use_gpu:
            X_train_fold = torch.tensor(
                X_train_fold.to_cupy(), dtype=torch.float32, device=device
            )
            X_val_fold = torch.tensor(
                X_val_fold.to_cupy(), dtype=torch.float32, device=device
            )
        else:
            X_train_fold = torch.tensor(
                X_train_fold.to_numpy(), dtype=torch.float32, device=device
            )
            X_val_fold = torch.tensor(
                X_val_fold.to_numpy(), dtype=torch.float32, device=device
            )

        # Initialize models
        generator = Generator(latent_dim, input_dim).to(device)
        discriminator = Discriminator(input_dim).to(device)

        # Optimizers and Loss
        optimizer_G = optim.Adam(generator.parameters(), lr=lr_g)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d)
        loss_function = nn.BCELoss()

        # Training Loop
        for epoch in range(num_epochs):  # Shortened for optimization
            optimizer_D.zero_grad()
            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            idx = torch.randint(0, X_train_fold.shape[0], (batch_size,), device=device)
            real_data = X_train_fold[idx]
            real_output = discriminator(real_data)
            real_loss = loss_function(real_output, real_labels)

            z = torch.randn((batch_size, latent_dim), device=device)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            fake_loss = loss_function(fake_output, fake_labels)

            loss_D = real_loss + fake_loss
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_output = discriminator(fake_data)
            loss_G = loss_function(fake_output, real_labels)
            loss_G.backward()
            optimizer_G.step()

            # Validation: Evaluate Generator on `X_val_fold`
            with torch.no_grad():
                z_val = torch.randn((batch_size, latent_dim), device=device)
                fake_val_data = generator(z_val)
                fake_val_output = discriminator(fake_val_data)
                G_val_loss = loss_function(fake_val_output, real_labels)

            # Append validation loss instead of training loss
            val_losses.append(G_val_loss.item())

            # Use a global step counter instead of `epoch`
            trial.report(G_val_loss.item(), global_step)

            # Only prune if the step is new
            if trial.should_prune():
                raise optuna.TrialPruned()

            global_step += 1 # Increment step counter to ensure uniqueness across folds            

    # return the mean of the losses
    return np.mean(val_losses)


def optimize_gan(X_train, y_train, use_gpu=False, n_trials=20, n_jobs=-1):
    """
    Runs Optuna optimization for GAN training with pruning and parallel execution.

    Args:
        X_train (training dataset without the labels
        y_train (data labels)
        use_gpu (bool): Whether to use GPU.
        n_trials (int): Number of optimization trials.
        n_jobs (int): Number of parallel jobs (-1 uses all available cores).

    Returns:
        optuna.Study: The completed study.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert the dataset type
    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.astype("float32")
    elif isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)

    if isinstance(y_train, cudf.Series):
        y_train = y_train.astype("int32")
    elif isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.int32, device=device)

    # Optimize using multiple parallel jobs
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    # study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective_gan(trial, X_train, y_train, use_gpu),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print("Best Parameters for GAN:", study.best_params)
    return study.best_params
