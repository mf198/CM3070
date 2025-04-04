import optuna
import cupy as cp
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from ccfd.evaluation.evaluate_models import evaluate_model_metric
from ccfd.utils.type_converter import to_numpy_safe
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer
from cuml.preprocessing import StandardScaler as cuStandardScaler
from sklearn.preprocessing import StandardScaler as skStandardScaler
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger


def objective_logistic_regression(trial, X_train, y_train, train_params):
    """
    Optuna objective function to optimize Logistic Regression.

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

    use_gpu = train_params["device"] == "gpu"
    ovs_function = train_params["oversampling_function"]

    if use_gpu:
        params = {
            "penalty": "l2",
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            #"C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "solver": "qn",  # cuML only supports "qn"            
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
        }
        model = cuLogisticRegression(**params)

        # Scale the dataset
        X_train = cuStandardScaler().fit_transform(X_train)

    else:
        solver = trial.suggest_categorical(
            "solver", ["lbfgs", "liblinear", "saga", "sag", "newton-cg", "newton-cholesky"]
        )

        if solver in ["liblinear", "saga"]:
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        else:
            penalty = "l2"

        params = {
            "penalty": penalty,
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "solver": solver,
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
            "class_weight": "balanced",
            "warm_start": True,
        }
        model = skLogisticRegression(**params)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    evaluation_scores = []

    # Ensure correct format for CPU/GPU mode
    # Convert cuDF to CuPy for GPU, NumPy for CPU
    if use_gpu:
        X_train_np = X_train.to_cupy()
        y_train_np = y_train.to_cupy().get()
    else:
        X_train_np = to_numpy_safe(X_train)
        y_train_np = to_numpy_safe(y_train)

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

        # Setup TensorBoard loggers
        log_dir = f"runs/lr_trial/trial_{trial.number}_fold_{fold_idx}"
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

        # Predict probabilities
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
            "LR/C": params["C"]
        }
        model_logger.log_scalars(log_metrics, step=fold_idx)

        # Log GPU usage (if applicable)
        if gpu_logger:
            gpu_logger.log_gpu_stats(step=fold_idx)

        # Close loggers for this fold
        model_logger.close()
        if gpu_logger:
            gpu_logger.close()        

    return np.mean(evaluation_scores)  # Optuna optimizes based on the selected metric


def optimize_logistic_regression(X_train, y_train, train_params):
    """
    Runs Optuna optimization for Logistic Regression.

    Args:
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        y_train (cuDF.Series or pandas.Series): Training labels.
        train_params (dict): Dictionary containing training parameters, including:
            - "device" (str): Device for training. Options: ["gpu", "cpu"].
            - "trials" (int): Number of optimization trials.
            - "metric" (str): Evaluation metric to optimize. Options: ["pr_auc", "f1", "precision", "cost"].
            - "jobs" (int): Number of parallel jobs (-1 to use all available cores).
        save_path (str, optional): Path to save the best model. Default: "ccfd/pretrained_models/pt_logistic_regression.pkl".

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

    save_filename = f"pt_{model_name}_{ovs_name}_{metric}.pkl"
    save_path = os.path.join(train_params["output_folder"], save_filename)

    # Start the timer to calculate training time
    timer.start()

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_logistic_regression(
            trial, X_train, y_train, train_params
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print(f"Best Logistic Regression Parameters ({metric}):", study.best_params)
    print(f"Best Logistic Regression Value ({metric}):", study.best_value)

    # Retrain the best model
    best_model = (
        cuLogisticRegression(**study.best_params)
        if use_gpu
        else skLogisticRegression(**study.best_params)
    )
    best_model.fit(X_train, y_train)  # Keep cuDF format

    # Total execution time
    elapsed_time = round(timer.elapsed_final(), 2)
    print(f"Total training time: {elapsed_time}")

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"Best Logistic Regression model saved at: {save_path}")

    # Save training performance details to CSV
    save_time_performance(train_params, study.best_value, elapsed_time)

    return study.best_params
