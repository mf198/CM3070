import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import joblib
import os
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import cudf
from ccfd.models.vae import FraudVariationalAutoencoder
from ccfd.utils.timer import Timer
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div


from sklearn.metrics import average_precision_score
from ccfd.utils.type_converter import to_numpy_safe


def objective_vae(trial, X_train, train_params):
    use_gpu = train_params["device"] == "gpu"
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # TensorBoard loggers for this trial
    trial_id = trial.number
    model_logger = ModelTensorBoardLogger(log_dir=f"runs/vae_trials/trial_{trial_id}")
    gpu_logger = (
        GPUTensorBoardLogger(log_dir=f"runs/vae_trials/gpu_trial_{trial_id}")
        if use_gpu
        else None
    )

    # Suggest hyperparameters
    latent_dim = trial.suggest_int("latent_dim", 4, 16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    epochs = trial.suggest_int("epochs", 10, 50)

    # Convert & normalize data
    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas().values
    else:
        X_train = X_train.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]

        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True
        )

        model = FraudVariationalAutoencoder(X_train.shape[1], latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                batch = batch[0].to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                loss = vae_loss_function(recon, batch, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model_logger.log_scalar("Loss/train", total_loss, step=epoch)
            if gpu_logger:
                gpu_logger.log_gpu_stats(step=epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(X_val_tensor)
            val_loss = vae_loss_function(recon, X_val_tensor, mu, logvar).item()
            fold_losses.append(val_loss)

    model_logger.close()
    if gpu_logger:
        gpu_logger.close()

    return np.mean(fold_losses)


def optimize_vae(X_train, train_params):
    timer = Timer()
    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    os.makedirs(output_folder, exist_ok=True)

    model_filename = "pt_vae.pth"
    scaler_filename = "pt_vae_scaler.pkl"
    model_path = os.path.join(output_folder, model_filename)
    scaler_path = os.path.join(output_folder, scaler_filename)

    timer.start()

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_vae(trial, X_train, train_params),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print(f"🔥 Best VAE Parameters: {study.best_params}")
    print(f"🔥 Best Validation Loss: {study.best_value}")

    best_params = study.best_params
    latent_dim = best_params["latent_dim"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    epochs = best_params["epochs"]

    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas().values
    else:
        X_train = X_train.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    train_loader = DataLoader(
        TensorDataset(X_tensor), batch_size=batch_size, shuffle=True
    )

    model = FraudVariationalAutoencoder(X_train.shape[1], latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_logger = ModelTensorBoardLogger(log_dir="runs/vae_optimization")
    gpu_logger = GPUTensorBoardLogger(log_dir="runs/vae_gpu") if use_gpu else None

    print("\n🚀 Retraining the best VAE with optimal parameters...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log metrics
        model_logger.log_scalar("Loss/train", total_loss, step=epoch)
        if gpu_logger:
            gpu_logger.log_gpu_stats(step=epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}")

    print("\n✅ Best VAE retrained successfully!")

    # Save model with structure
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "latent_dim": latent_dim,
        },
        model_path,
    )

    joblib.dump(scaler, scaler_path)
    print(f"✅ VAE model saved at: {model_path}")
    print(f"✅ Scaler saved at: {scaler_path}")

    # Close loggers
    model_logger.close()
    if gpu_logger:
        gpu_logger.close()

    elapsed = round(timer.elapsed_final(), 2)
    save_time_performance(train_params, study.best_value, elapsed)

    return best_params
