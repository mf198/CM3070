import cudf
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from ccfd.data.gan_oversampler import Generator, Discriminator
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger


def objective_gan(trial, X_train, y_train, use_gpu=False):
    """
    Optuna objective function to optimize GAN hyperparameters.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (numpy.ndarray or cupy.ndarray): Training dataset.
        y_train (numpy.ndarray or cupy.ndarray): Training labels.
        use_gpu (bool): Whether to use GPU.

    Returns:
        float: The best validation loss value (minimized).
    """
    # Hyperparameter optimization
    num_epochs = trial.suggest_int("num_epochs", 500, 5000, step=500)
    latent_dim = trial.suggest_int("latent_dim", 10, 100)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr_g = trial.suggest_float("lr_g", 1e-5, 1e-2, log=True)
    lr_d = trial.suggest_float("lr_d", 1e-5, 1e-2, log=True)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # TensorBoard loggers for this trial
    trial_id = trial.number
    model_logger = ModelTensorBoardLogger(log_dir=f"runs/gan_trials/trial_{trial_id}")
    gpu_logger = (
        GPUTensorBoardLogger(log_dir=f"runs/gan_trials/gpu_trial_{trial_id}")
        if use_gpu
        else None
    )

    input_dim = X_train.shape[1]

    # Initialize a unique step counter before training
    global_step = 0

    # Convert dataset **before** training loop
    if use_gpu:
        X_train_tensor = torch.tensor(
            X_train.to_cupy(), dtype=torch.float32, device=device
        )
    else:
        X_train_tensor = torch.tensor(
            X_train.to_numpy(), dtype=torch.float32, device=device
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in skf.split(X_train.to_numpy(), y_train.to_numpy()):
        X_train_fold = X_train_tensor[train_idx]
        X_val_fold = X_train_tensor[val_idx]

        # Initialize models
        generator = Generator(latent_dim, input_dim).to(device)
        discriminator = Discriminator(input_dim).to(device)

        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(
            discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
        )

        # Use BCEWithLogitsLoss instead of BCELoss
        loss_function = nn.BCEWithLogitsLoss()

        # GradScaler initialization for mixed precision training
        scaler = torch.amp.GradScaler(enabled=use_gpu)

        # Training Loop
        for epoch in range(num_epochs):
            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            idx = torch.randint(0, X_train_fold.shape[0], (batch_size,), device=device)
            real_data = X_train_fold[idx]

            # Train Discriminator
            optimizer_D.zero_grad()
            with torch.amp.autocast(device_type="cuda" if use_gpu else "cpu"):
                real_output = discriminator(real_data)
                real_loss = loss_function(real_output, real_labels)

                z = torch.randn((batch_size, latent_dim), device=device)
                fake_data = generator(z)
                fake_output = discriminator(fake_data.detach())
                fake_loss = loss_function(fake_output, fake_labels)

                loss_D = real_loss + fake_loss

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()

            # Train Generator
            optimizer_G.zero_grad()
            with torch.amp.autocast(device_type="cuda" if use_gpu else "cpu"):
                fake_output = discriminator(fake_data)
                loss_G = loss_function(fake_output, real_labels)

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # Log losses to TensorBoard
            model_logger.log_scalars(
                {
                    "Loss/Generator": loss_G.item(),
                    "Loss/Discriminator": loss_D.item(),
                },
                step=epoch,
            )

            # Log gradients for monitoring GAN stability
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    model_logger.log_histogram(f"Gradients/Generator/{name}", param.grad, step=epoch)
            for name, param in discriminator.named_parameters():
                if param.grad is not None:
                    model_logger.log_histogram(f"Gradients/Discriminator/{name}", param.grad, step=epoch)

            # Validation: Evaluate Generator on X_val_fold
            with torch.no_grad():
                z_val = torch.randn((batch_size, latent_dim), device=device)
                fake_val_data = generator(z_val)
                fake_val_output = discriminator(fake_val_data)
                G_val_loss = loss_function(fake_val_output, real_labels)

            # Append validation loss for monitoring
            val_losses.append(G_val_loss.item())

            # Log validation loss
            model_logger.log_scalar("Loss/Validation", G_val_loss.item(), step=epoch)

            # Log GPU stats if applicable
            if gpu_logger:
                gpu_logger.log_gpu_stats(step=epoch)

            # Report to Optuna & prune if needed
            trial.report(G_val_loss.item(), global_step)
            if trial.should_prune():
                raise optuna.TrialPruned()

            global_step += 1  # Ensure unique step count

        # Close TensorBoard loggers after training
        model_logger.close()
        if gpu_logger:
            gpu_logger.close()

    # Return the mean of validation losses across all folds
    return np.mean(val_losses)


def optimize_gan(X_train, y_train, train_params):
    """
    Runs Optuna optimization for GAN training with pruning and parallel execution.

    Args:
        X_train (DataFrame, NumPy array, or cuDF DataFrame): Training dataset without labels.
        y_train (Series, NumPy array, or cuDF Series): Training labels.
        train_params (dict): Dictionary containing:
            - "device" (str): Device to use (GPU or CPU)
            - "trials" (int): Number of optimization trials.
            - "jobs" (int): Number of parallel jobs.
            - "output_folder" (str): Directory where the model will be saved.

    Returns:
        dict: Best hyperparameters from Optuna.
    """
    timer = Timer()

    # Extract parameters
    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define model save path dynamically
    save_path = os.path.join(output_folder, "pt_gan.pth")

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

    # Start the timer to calculate training time
    timer.start()

    # Optimize using multiple parallel jobs
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_gan(trial, X_train, y_train, use_gpu),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print("Best Parameters for GAN:", study.best_params)
    print("Best Value for GAN:", study.best_value)

    # Re-train best GAN with found parameters
    best_params = study.best_params
    best_generator = Generator(best_params["latent_dim"], X_train.shape[1]).to(device)
    best_discriminator = Discriminator(X_train.shape[1]).to(device)

    # Total execution time
    elapsed_time = round(timer.elapsed_final(), 2)
    print(f"Total training time: {elapsed_time}")

    # Save best model
    torch.save(
        {
            "generator": best_generator.state_dict(),
            "discriminator": best_discriminator.state_dict(),
            "params": best_params,
        },
        save_path,
    )

    print(f"Best GAN model saved at: {save_path}")

    # Save training performance details to CSV
    save_time_performance(train_params, study.best_value, elapsed_time)

    return best_params
