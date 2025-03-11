import cudf
import optuna
import torch
import pandas as pd
import cupy as cp
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from ccfd.data.gan_oversampler import Generator, Critic


def objective_wgan_old(trial, X_train, y_train, use_gpu=False):
    """
    Optuna objective function to optimize WGAN hyperparameters.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (numpy.ndarray or cupy.ndarray): Training dataset.
        use_gpu (bool): Whether to use GPU.

    Returns:
        float: The best loss value (minimized).
    """
    # Hyperparameters to optimize
    num_epochs = trial.suggest_int("num_epochs", 500, 10000, step=500)
    latent_dim = trial.suggest_int("latent_dim", 10, 100)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr_g = trial.suggest_float("lr_g", 1e-5, 1e-2, log=True)
    lr_c = trial.suggest_float("lr_c", 1e-5, 1e-2, log=True)
    weight_clip = trial.suggest_float("weight_clip", 0.001, 0.05, log=True)
    critic_iterations = trial.suggest_int("critic_iterations", 1, 10)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]

    # Initialize a unique step counter before training
    global_step = 0

    # Initialize models
    generator = Generator(latent_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)

    # Optimizers
    optimizer_G = optim.RMSprop(generator.parameters(), lr=lr_g)
    optimizer_C = optim.RMSprop(critic.parameters(), lr=lr_c)

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

        # Training Loop
        for epoch in range(num_epochs):
            for _ in range(critic_iterations):
                optimizer_C.zero_grad()

                real_idx = torch.randint(
                    0, X_train_fold.shape[0], (batch_size,), device=device
                )
                real_data = X_train_fold[real_idx]

                real_output = critic(real_data)
                loss_real = -torch.mean(real_output)

                z = torch.randn((batch_size, latent_dim), device=device)
                fake_data = generator(z)
                fake_output = critic(fake_data.detach())
                loss_fake = torch.mean(fake_output)

                loss_C = loss_real + loss_fake
                loss_C.backward()
                optimizer_C.step()

                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

            optimizer_G.zero_grad()
            fake_data = generator(z)
            fake_output = critic(fake_data)
            loss_G = -torch.mean(fake_output)
            loss_G.backward()
            optimizer_G.step()

            # Validation Loss
            with torch.no_grad():
                z_val = torch.randn((batch_size, latent_dim), device=device)
                fake_val_data = generator(z_val)
                fake_val_output = critic(fake_val_data)
                G_val_loss = -torch.mean(fake_val_output)  # Minimize Generator Score

            # Append Validation Loss
            val_losses.append(G_val_loss.item())

            # Use global step for reporting
            trial.report(G_val_loss.item(), global_step)

            # Prune Trials
            if trial.should_prune():
                raise optuna.TrialPruned()

            global_step += 1  # Increment step counter

    return np.mean(val_losses)  # Return Average Validation Loss

def objective_wgan(trial, X_train, y_train, use_gpu=False):
    """
    Optuna objective function to optimize WGAN hyperparameters.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        y_train (cuDF.Series or pandas.Series): Training labels.
        use_gpu (bool): Whether to use GPU.

    Returns:
        float: The best loss value (minimized).
    """
    # Hyperparameters to optimize
    num_epochs = trial.suggest_int("num_epochs", 500, 10000, step=500)
    latent_dim = trial.suggest_int("latent_dim", 10, 100)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr_g = trial.suggest_float("lr_g", 1e-5, 1e-2, log=True)
    lr_c = trial.suggest_float("lr_c", 1e-5, 1e-2, log=True)
    weight_clip = trial.suggest_float("weight_clip", 0.001, 0.05, log=True)
    critic_iterations = trial.suggest_int("critic_iterations", 1, 10)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]

    # Initialize models
    generator = Generator(latent_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)

    # Optimizers
    optimizer_G = optim.RMSprop(generator.parameters(), lr=lr_g)
    optimizer_C = optim.RMSprop(critic.parameters(), lr=lr_c)

    # Convert dataset **before** training loop
    if use_gpu:
        X_train_tensor = torch.tensor(X_train.to_cupy(), dtype=torch.float32, device=device)
    else:
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32, device=device)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in skf.split(X_train.to_numpy(), y_train.to_numpy()):
        X_train_fold = X_train_tensor[train_idx]
        X_val_fold = X_train_tensor[val_idx]

        # Training Loop
        for epoch in range(num_epochs):
            for _ in range(critic_iterations):
                optimizer_C.zero_grad()

                # **Avoid slow indexing**
                real_idx = torch.randint(0, X_train_fold.shape[0], (batch_size,), device=device)
                real_data = X_train_fold.index_select(0, real_idx).clone().detach()

                # Use **autocast for faster computation**
                with torch.amp.autocast("cuda" if use_gpu else "cpu"):
                    real_output = critic(real_data)
                    loss_real = -torch.mean(real_output)

                    z = torch.randn((batch_size, latent_dim), device=device)
                    fake_data = generator(z)
                    fake_output = critic(fake_data.detach())
                    loss_fake = torch.mean(fake_output)

                    loss_C = loss_real + loss_fake

                loss_C.backward()
                optimizer_C.step()

                # **Use tensor-wide operation
                with torch.no_grad():
                    for p in critic.parameters():
                        p.data.clamp_(-weight_clip, weight_clip)

            optimizer_G.zero_grad()

            with torch.amp.autocast("cuda" if use_gpu else "cpu"):
                fake_data = generator(z)
                fake_output = critic(fake_data)
                loss_G = -torch.mean(fake_output)

            loss_G.backward()
            optimizer_G.step()

            # Validation Loss
            with torch.no_grad(), torch.amp.autocast("cuda" if use_gpu else "cpu"):
                z_val = torch.randn((batch_size, latent_dim), device=device)
                fake_val_data = generator(z_val)
                fake_val_output = critic(fake_val_data)
                G_val_loss = -torch.mean(fake_val_output)  # Minimize Generator Score

            val_losses.append(G_val_loss.item())

            trial.report(G_val_loss.item(), epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(val_losses)  # Return Average Validation Loss


def optimize_wgan(X_train, y_train, train_params):
    """
    Runs Optuna optimization for WGAN training.

    Args:
        X_train (DataFrame, NumPy array, or cuDF DataFrame): Training dataset.
        y_train (Series, NumPy array, or cuDF Series): Training labels.
        train_params (dict): Dictionary containing:
            - "device" (str): Device to use (GPU or CPU)
            - "trials" (int): Number of optimization trials.
            - "jobs" (int): Number of parallel jobs.
            - "output_folder" (str): Directory where the model will be saved.

    Returns:
        dict: Best hyperparameters from Optuna.
    """

    # Extract parameters
    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define model save path dynamically
    save_path = os.path.join(output_folder, "pt_wgan.pth")

    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert dataset to the correct format
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
    study.optimize(
        lambda trial: objective_wgan(trial, X_train, y_train, use_gpu),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print("✅ Best Parameters for WGAN:", study.best_params)

    # Re-train best WGAN with found parameters
    best_params = study.best_params
    best_generator = Generator(best_params["latent_dim"], X_train.shape[1]).to(device)
    best_critic = Critic(X_train.shape[1]).to(device)

    # Save best model
    torch.save({
        "generator": best_generator.state_dict(),
        "critic": best_critic.state_dict(),
        "params": best_params
    }, save_path)

    print(f"🎯 Best WGAN model saved at: {save_path}")

    return best_params
