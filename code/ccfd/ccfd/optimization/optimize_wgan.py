import cudf
import optuna
import torch
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from ccfd.data.gan_oversampler import Generator, Critic
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger


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
    # Hyperparameter optimization
    num_epochs = trial.suggest_int("num_epochs", 500, 5000, step=500)
    latent_dim = trial.suggest_int("latent_dim", 10, 100)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    lr_g = trial.suggest_float("lr_g", 1e-5, 1e-2, log=True)
    lr_c = trial.suggest_float("lr_c", 1e-5, 1e-2, log=True)
    weight_clip = trial.suggest_float("weight_clip", 0.001, 0.05, log=True)
    critic_iterations = trial.suggest_int("critic_iterations", 1, 5)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # TensorBoard loggers
    trial_id = trial.number
    model_logger = ModelTensorBoardLogger(log_dir=f"runs/wgan_trials/trial_{trial_id}")
    gpu_logger = (
        GPUTensorBoardLogger(log_dir=f"runs/wgan_trials/gpu_trial_{trial_id}")
        if use_gpu
        else None
    )

    input_dim = X_train.shape[1]

    global_step = 0
    patience = 20  # Stop training if no improvement for 20 consecutive epochs
    best_val_loss = float("inf")    

    # Convert dataset before training loop
    X_train_tensor = torch.tensor(X_train.to_cupy() if use_gpu else X_train.to_numpy(), 
                                  dtype=torch.float32, device=device)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train.to_numpy(), y_train.to_numpy())):
        X_train_fold = X_train_tensor[train_idx]
        X_val_fold = X_train_tensor[val_idx]

        # Initialize models
        generator = Generator(latent_dim, input_dim).to(device)
        critic = Critic(input_dim).to(device)

        # Optimizers
        optimizer_G = optim.RMSprop(generator.parameters(), lr=lr_g)
        optimizer_C = optim.RMSprop(critic.parameters(), lr=lr_c)

        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            for _ in range(critic_iterations):
                optimizer_C.zero_grad()
                real_idx = torch.randint(0, X_train_fold.shape[0], (batch_size,), device=device)
                real_data = X_train_fold.index_select(0, real_idx).clone().detach()

                with torch.amp.autocast(device_type="cuda" if use_gpu else "cpu"):
                    real_output = critic(real_data)
                    loss_real = -torch.mean(real_output)

                    z = torch.randn((batch_size, latent_dim), device=device)
                    fake_data = generator(z)
                    fake_output = critic(fake_data.detach())
                    loss_fake = torch.mean(fake_output)

                    loss_C = loss_real + loss_fake

                loss_C.backward()
                optimizer_C.step()

                with torch.no_grad():
                    for p in critic.parameters():
                        p.data.clamp_(-weight_clip, weight_clip)

            optimizer_G.zero_grad()

            with torch.amp.autocast(device_type="cuda" if use_gpu else "cpu"):
                fake_data = generator(z)
                fake_output = critic(fake_data)
                loss_G = -torch.mean(fake_output)

            loss_G.backward()
            optimizer_G.step()

            # Log losses to TensorBoard
            model_logger.log_scalars(
                {
                    "Loss/Generator": loss_G.item(),
                    "Loss/Critic": loss_C.item(),
                },
                step=epoch,
            )

            # Log gradients for monitoring GAN stability
            for name, param in generator.named_parameters():
                if param.grad is not None and param.grad.numel() > 0 and torch.isfinite(param.grad).all(): # prevents an error if the histogram is empty
                    model_logger.log_histogram(f"Gradients/Generator/{name}", param.grad, step=epoch)
            for name, param in critic.named_parameters():
                if param.grad is not None and param.grad.numel() > 0 and torch.isfinite(param.grad).all(): # prevents an error if the histogram is empty
                    model_logger.log_histogram(f"Gradients/Critic/{name}", param.grad, step=epoch)            

            # Validation Loss
            with torch.no_grad(), torch.amp.autocast(device_type="cuda" if use_gpu else "cpu"):
                z_val = torch.randn((batch_size, latent_dim), device=device)
                fake_val_data = generator(z_val)
                fake_val_output = critic(fake_val_data)
                G_val_loss = -torch.mean(fake_val_output)

            val_losses.append(G_val_loss.item())

            # Log validation loss
            model_logger.log_scalar("Loss/Validation", G_val_loss.item(), step=epoch)

            # Log GPU stats if applicable
            if gpu_logger:
                gpu_logger.log_gpu_stats(step=epoch)

            # Early Stopping Logic
            if G_val_loss.item() < best_val_loss:
                best_val_loss = G_val_loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs).")                
                break  # Stop training

            # Report to Optuna & prune if needed
            trial.report(G_val_loss.item(), global_step)
            if trial.should_prune():
                raise optuna.TrialPruned()

            global_step += 1  # Ensure unique step count

    return np.mean(val_losses)  # Return the mean validation loss



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
    timer = Timer()

    # Extract parameters
    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

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

    # Start the timer to calculate training time
    timer.start()

    # Optimize using multiple parallel jobs
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_wgan(trial, X_train, y_train, use_gpu),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print("Best Parameters for WGAN:", study.best_params)
    print("Best Value for WGAN:", study.best_value)

    # Re-train best WGAN with found parameters
    best_params = study.best_params
    best_generator = Generator(best_params["latent_dim"], X_train.shape[1]).to(device)
    best_critic = Critic(X_train.shape[1]).to(device)

    # Total execution time
    elapsed_time = round(timer.elapsed_final(), 2)
    print(f"Total training time: {elapsed_time}")

    # Save best model
    torch.save(
        {
            "generator": best_generator.state_dict(),
            "critic": best_critic.state_dict(),
            "params": best_params,
        },
        save_path,
    )

    print(f"Best WGAN model saved at: {save_path}")

    # Save training performance details to CSV
    save_time_performance(train_params, study.best_value, elapsed_time)

    return best_params
