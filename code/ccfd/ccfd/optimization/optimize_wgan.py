import cudf
import optuna
import torch
import cupy as cp
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ccfd.data.gan_oversampler import Generator, Critic

def objective_wgan(trial, X_real, use_gpu=False):
    """
    Optuna objective function to optimize WGAN hyperparameters.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_real (numpy.ndarray or cupy.ndarray): Training dataset.
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

    input_dim = X_real.shape[1]

    # Initialize models
    generator = Generator(latent_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)

    # Optimizers
    optimizer_G = optim.RMSprop(generator.parameters(), lr=lr_g)
    optimizer_C = optim.RMSprop(critic.parameters(), lr=lr_c)

    # Training Loop
    for epoch in range(num_epochs):
        for _ in range(critic_iterations):
            optimizer_C.zero_grad()
            real_idx = torch.randint(0, X_real.size(0), (batch_size,), device=device)
            real_data = X_real[real_idx]

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

        # Report intermediate loss for pruning
        trial.report(loss_G.item(), epoch)

        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()      

    return loss_G.item()

def optimize_wgan(X_real, use_gpu=False, n_trials=20, n_jobs=-1):
    """
    Runs Optuna optimization for WGAN training.

    Args:
        X_real (numpy.ndarray or cupy.ndarray): Training dataset.
        use_gpu (bool): Whether to use GPU.
        n_trials (int): Number of optimization trials.

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
    #study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_wgan(trial, X_real, use_gpu), n_trials=n_trials, n_jobs=n_jobs)    

    print("Best Parameters for WGAN:", study.best_params)
    return study.best_params
