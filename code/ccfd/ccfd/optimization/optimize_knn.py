import optuna
import cupy as cp
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from cuml.neighbors import KNeighborsClassifier as cuKNN
from sklearn.neighbors import KNeighborsClassifier as skKNN
from ccfd.evaluation.evaluate_models import evaluate_model_metric
from ccfd.utils.type_converter import to_numpy_safe
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger


def objective_knn(trial, X_train, y_train, train_params):
    """
    Optuna objective function to optimize K-Nearest Neighbors (KNN).

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        y_train (cuDF.Series or pandas.Series): Training labels.
        train_params (dict): Dictionary containing training parameters, including:
            - "device" (str): Device for training. Options: ["gpu", "cpu"].
            - "metric" (str): Evaluation metric to optimize. Options: ["pr_auc", "f1", "precision", "cost"].

    Returns:
        float: The computed evaluation metric score.
    """

    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
    }

    use_gpu = train_params["device"] == "gpu"
    ovs_function = train_params["oversampling_function"]

    model = cuKNN(**params) if use_gpu else skKNN(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    evaluation_scores = []

    # Convert cuDF to CuPy for GPU, NumPy for CPU
    if use_gpu:
        X_train_np = X_train.to_cupy()
        y_train_np = y_train.to_cupy().get()
    else:
        X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        if use_gpu:
            X_train_fold, X_val_fold = (
                X_train.iloc[cp.array(train_idx)],
                X_train.iloc[cp.array(val_idx)],
            )
            y_train_fold, y_val_fold = (
                y_train.iloc[cp.array(train_idx)],
                y_train.iloc[cp.array(val_idx)],
            )
        else:
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        log_dir = f"runs/knn_trial/trial_{trial.number}_fold_{fold_idx}"
        model_logger = ModelTensorBoardLogger(log_dir=log_dir)
        gpu_logger = GPUTensorBoardLogger(log_dir=log_dir) if use_gpu else None

        # Apply an oversampling method if selected
        if ovs_function:
            X_train_fold_oversampled, y_train_fold_oversampled = ovs_function(
                X_train_fold, y_train_fold, use_gpu
            )
        else:
            X_train_fold_oversampled = X_train_fold
            y_train_fold_oversampled = y_train_fold

        # Train model on the oversampled fold
        model.fit(X_train_fold_oversampled, y_train_fold_oversampled)

        # Predict probabilities on the validation fold
        y_proba = model.predict_proba(X_val_fold)

        # Convert to numpy and extract result
        y_proba = to_numpy_safe(y_proba)[:, 1]

        # Ensure y_val_fold is also a NumPy array before evaluation
        y_val_fold = to_numpy_safe(y_val_fold)

        # Evaluate the model using the specified metric
        evaluation_score = evaluate_model_metric(y_val_fold, y_proba, train_params)

        evaluation_scores.append(evaluation_score)

        # Log performance metrics
        log_metrics = {
            "Metric/Eval_Score": evaluation_score,
            "kNN/N_Neighbors": params["n_neighbors"],
        }
        model_logger.log_scalars(log_metrics, step=fold_idx)

        # Log GPU usage (if applicable)
        if gpu_logger:
            gpu_logger.log_gpu_stats(step=fold_idx)

        # Close loggers for this fold
        model_logger.close()
        if gpu_logger:
            gpu_logger.close()     

    return np.mean(evaluation_scores)


def optimize_knn(X_train, y_train, train_params):
    """
    Runs Optuna optimization for the K-Nearest Neighbors (KNN) classifier.

    Args:
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        y_train (cuDF.Series or pandas.Series): Training labels.
        train_params (dict): Dictionary containing training parameters, including:
            - "device" (str): Device for training. Options: ["gpu", "cpu"].
            - "trials" (int): Number of optimization trials.
            - "metric" (str): Evaluation metric to optimize. Options: ["pr_auc", "f1", "precision", "cost"].
            - "jobs" (int): Number of parallel jobs (-1 to use all available cores).

    Returns:
        dict: The best hyperparameters found for KNN.
    """
    timer = Timer()

    use_gpu = train_params["device"] == "gpu"
    metric = train_params["metric"]
    model_name = train_params["model"]
    n_trials = train_params["trials"]
    n_jobs = train_params["jobs"]
    ovs_name = train_params["ovs"] if train_params["ovs"] else "no_ovs"
    output_folder = train_params["output_folder"]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define model save path dynamically
    model_filename = f"pt_{model_name}_{ovs_name}_{metric}.pkl"
    model_path = os.path.join(train_params["output_folder"], model_filename)

    # Start the timer to calculate training time
    timer.start()

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_knn(trial, X_train, y_train, train_params),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print(f"Best KNN Parameters ({metric}):", study.best_params)
    print(f"Best KNN Value ({metric}):", study.best_value)

    # Retrain the best model using the full dataset
    best_model = cuKNN(**study.best_params) if use_gpu else skKNN(**study.best_params)
    best_model.fit(X_train, y_train)

    # Total execution time
    elapsed_time = round(timer.elapsed_final(), 2)
    print(f"Total training time: {elapsed_time}")

    # Save the best model
    joblib.dump(best_model, model_path)
    print(f"Best KNN model saved at: {model_path}")

    # Save training performance details to CSV
    save_time_performance(train_params, study.best_value, elapsed_time)

    return study.best_params
