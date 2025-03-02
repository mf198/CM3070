import cudf
import optuna
import torch
import pandas as pd
import cupy as cp
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ccfd.data.gan_oversampler import Generator, Discriminator

def objective_gan(trial, X_real, use_gpu=False):
    """
    Optuna objective function to optimize GAN hyperparameters.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_real (numpy.ndarray or cupy.ndarray): Training dataset.
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

    input_dim = X_real.shape[1]

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

        idx = torch.randint(0, X_real.size(0), (batch_size,), device=device)
        real_data = X_real[idx]
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

        # Report intermediate loss for pruning
        trial.report(loss_G.item(), epoch)

        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()        

    return loss_G.item()  # Minimize Generator Loss

def optimize_gan(X_real, use_gpu=False, n_trials=20, n_jobs=-1):
    """
    Runs Optuna optimization for GAN training with pruning and parallel execution.

    Args:
        X_real (DataFrame, np.ndarray, or cudf.DataFrame): Training dataset.
        use_gpu (bool): Whether to use GPU.
        n_trials (int): Number of optimization trials.
        n_jobs (int): Number of parallel jobs (-1 uses all available cores).

    Returns:
        optuna.Study: The completed study.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert the dataframe to a torch tensor
    if isinstance(X_real, cudf.DataFrame):
        X_real = torch.tensor(X_real.to_pandas().to_numpy(), dtype=torch.float32, device=device)
    elif isinstance(X_real, pd.DataFrame):
        X_real = torch.tensor(X_real.to_numpy(), dtype=torch.float32, device=device)
    elif isinstance(X_real, cp.ndarray):  # If it's a CuPy array
        X_real = torch.tensor(cp.asnumpy(X_real), dtype=torch.float32, device=device)
    else:  # Assume it's already a NumPy array or compatible format
        X_real = torch.tensor(X_real, dtype=torch.float32, device=device)

    # Optimize using multiple parallel jobs
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())    
    study.optimize(lambda trial: objective_gan(trial, X_real, use_gpu), n_trials=n_trials, n_jobs=n_jobs)

    print("Best Parameters for GAN:", study.best_params)
    return study.best_params


