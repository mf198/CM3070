import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import cudf
from ccfd.models.autoencoder import FraudAutoencoder
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger


def objective_autoencoder(trial, X_train, train_params):
    use_gpu = train_params["device"] == "gpu"
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    latent_dim = trial.suggest_int("latent_dim", 4, 16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    epochs = trial.suggest_int("epochs", 10, 50)

    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas().values
    else:
        X_train = X_train.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_errors = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

        input_dim = X_train.shape[1]
        model = FraudAutoencoder(input_dim, latent_dim=latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Setup TensorBoard loggers for this fold
        log_dir = f"runs/optuna_trials/trial_{trial.number}_fold_{fold_idx}"
        model_logger = ModelTensorBoardLogger(log_dir=log_dir)
        gpu_logger = GPUTensorBoardLogger(log_dir=log_dir) if use_gpu else None

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                batch = batch[0].to(device)
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model_logger.log_scalar("Loss/train", epoch_loss, step=epoch)
            if gpu_logger:
                gpu_logger.log_gpu_stats(step=epoch)

        model_logger.close()
        if gpu_logger:
            gpu_logger.close()

        model.eval()
        with torch.no_grad():
            reconstructed = model(X_val_tensor)
            reconstruction_error = torch.mean((X_val_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        fold_errors.append(np.mean(reconstruction_error))

    return np.mean(fold_errors)

###

def optimize_autoencoder(X_train, train_params):
    timer = Timer()
    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    os.makedirs(output_folder, exist_ok=True)

    model_filename = "pt_autoencoder.pth"
    scaler_filename = "pt_autoencoder_scaler.pkl"
    model_path = os.path.join(output_folder, model_filename)
    scaler_path = os.path.join(output_folder, scaler_filename)

    timer.start()

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_autoencoder(trial, X_train, train_params), n_trials=n_trials, n_jobs=n_jobs)

    print(f"🔥 Best Autoencoder Parameters: {study.best_params}")
    print(f"🔥 Best Reconstruction Error: {study.best_value}")

    best_params = study.best_params
    latent_dim = best_params["latent_dim"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    epochs = best_params["epochs"]

    # Convert and scale
    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas().values
    else:
        X_train = X_train.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    best_model = FraudAutoencoder(X_train.shape[1], latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=lr)

    # Initialize TensorBoard loggers
    model_logger = ModelTensorBoardLogger(log_dir="runs/autoencoder_optimization")

    gpu_logger = None
    if use_gpu:
        from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger
        gpu_logger = GPUTensorBoardLogger(log_dir="runs/autoencoder_gpu")    

    print("\n🚀 Retraining the best Autoencoder with optimal parameters...")
    best_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = best_model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log training loss and GPU usage
        model_logger.log_scalar("Loss/train", epoch_loss, step=epoch)
        if gpu_logger:
            gpu_logger.log_gpu_stats(step=epoch)        

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6f}")

    print("\n✅ Best Autoencoder retrained successfully!")

    # Save model with full config
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "input_dim": X_train.shape[1],
        "latent_dim": latent_dim
    }, model_path)

    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Scaler saved at: {scaler_path}")

    # Close TensorBoard loggers
    model_logger.close()
    if gpu_logger:
        gpu_logger.close()

    # Log training time and best result
    elapsed_time = round(timer.elapsed_final(), 2)
    save_time_performance(train_params, study.best_value, elapsed_time)

    return best_params



def optimize_autoencoder_old(X_train, train_params):
    """
    Runs Optuna optimization for the Autoencoder.

    Args:
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        train_params (dict): Dictionary containing training parameters.

    Returns:
        dict: The best hyperparameters found for the Autoencoder.
    """
    timer = Timer()
    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define model save path
    model_filename = "pt_autoencoder.pth"
    scaler_filename = "pt_autoencoder_scaler.pkl"
    model_path = os.path.join(train_params["output_folder"], model_filename)
    scaler_path = os.path.join(train_params["output_folder"], scaler_filename)

    # Start the timer
    timer.start()

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_autoencoder(trial, X_train, train_params), n_trials=n_trials, n_jobs=n_jobs)

    print(f"🔥 Best Autoencoder Parameters: {study.best_params}")
    print(f"🔥 Best Reconstruction Error: {study.best_value}")

    # Extract best hyperparameters
    best_params = study.best_params
    latent_dim = best_params["latent_dim"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    epochs = best_params["epochs"]

    # Convert DataFrame to NumPy and normalize
    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas().values
    else:
        X_train = X_train.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Convert to PyTorch tensors
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    # Retrain the best Autoencoder using the full dataset
    best_model = FraudAutoencoder(X_train.shape[1], latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=lr)

    print("\n🚀 Retraining the best Autoencoder with optimal parameters...")
    best_model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = best_model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

    print("\n✅ Best Autoencoder retrained successfully!")

    # Save model with input and latent dim
    torch.save(best_model.state_dict(), model_path)
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "input_dim": X_train.shape[1],
        "latent_dim": latent_dim
    }, model_path)

    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Scaler saved at: {scaler_path}")

    # Save training performance details
    elapsed_time = round(timer.elapsed_final(), 2)
    save_time_performance(train_params, study.best_value, elapsed_time)

    return study.best_params
