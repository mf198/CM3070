import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import cudf
from ccfd.models.autoencoder import FraudAutoencoder
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer

def objective_autoencoder(trial, X_train, train_params):
    """
    Optuna objective function to optimize the Autoencoder.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        train_params (dict): Dictionary containing training parameters.

    Returns:
        float: Mean reconstruction error across folds.
    """
    use_gpu = train_params["device"] == "gpu"
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Suggest hyperparameters
    latent_dim = trial.suggest_int("latent_dim", 4, 16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    epochs = trial.suggest_int("epochs", 10, 50)

    # Convert DataFrame to NumPy and normalize
    X_train = X_train.to_pandas() if isinstance(X_train, cudf.DataFrame) else X_train
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_errors = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)

        train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

        # Initialize Autoencoder Model
        input_dim = X_train.shape[1]
        model = FraudAutoencoder(input_dim, latent_dim=latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train Autoencoder
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                batch = batch[0].to(device)
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_val_tensor)
            reconstruction_error = torch.mean((X_val_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        fold_errors.append(np.mean(reconstruction_error))

    return np.mean(fold_errors)

def optimize_autoencoder(X_train, train_params):
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
    model_filename = "pt_autoencoder.pkl"
    scaler_filename = "scaler.pkl"
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
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_pandas().values if isinstance(X_train, cudf.DataFrame) else X_train.values)

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
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = best_model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}")

    print("\n✅ Best Autoencoder retrained successfully!")

    # Save the best model
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Best Autoencoder model saved at: {model_path}")
    print(f"✅ Scaler saved at: {scaler_path}")

    # Save training performance details
    elapsed_time = round(timer.elapsed_final(), 2)
    save_time_performance(train_params, study.best_value, elapsed_time)

    return study.best_params