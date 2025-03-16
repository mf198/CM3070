import optuna
import cudf
import cupy as cp
import joblib
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from cuml.linear_model import MBSGDClassifier
from sklearn.model_selection import StratifiedKFold
from ccfd.evaluation.evaluate_models import evaluate_model_metric
from ccfd.utils.type_converter import to_numpy_safe
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer


def objective_sgd(trial, X_train, y_train, train_params):
    """
    Optuna objective function to optimize Stochastic Gradient Descent (SGD) classifier.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        y_train (cuDF.Series or pandas.Series): Training labels.
        train_params (dict): Dictionary containing training parameters, including:
            - "device" (str): Device for training. Options: ["gpu", "cpu"].
            - "metric" (str): Evaluation metric to optimize. Options: ["pr_auc", "f1", "precision", "cost"].
            - "cost_fp" (float, optional): Cost of a false positive (used if metric="cost").
            - "cost_fn" (float, optional): Cost of a false negative (used if metric="cost").

    Returns:
        float: The computed evaluation metric score.
    """

    use_gpu = train_params["device"] == "gpu"
    ovs_function = train_params["oversampling_function"]

    if use_gpu:
        params = {
            "loss": "adam",  # Logistic Regression
            "eta0": trial.suggest_float("eta0", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            "epochs": trial.suggest_int(
                "epochs", 500, 5000, step=500
            ),  # cuML uses "epochs"
        }
        model = MBSGDClassifier(**params)
    else:
        params = {
            "eta0": trial.suggest_float("eta0", 1e-6, 1e-2, log=True),
            "learning_rate": "constant",
            "max_iter": trial.suggest_int(
                "max_iter", 500, 5000, step=5000
            ),  # CPU uses "max_iter"
        }
        model = SGDClassifier(
            loss="log_loss", **params
        )  # No "epochs" in scikit-learngt

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    evaluation_scores = []

    # Convert cuDF to CuPy for GPU, NumPy for CPU
    if use_gpu:
        X_train_np = X_train.to_cupy()
        y_train_np = y_train.to_cupy().get()
    else:
        X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()

    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
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

        # Apply an oversampling method if selected
        if ovs_function:
            X_train_fold_oversampled, y_train_fold_oversampled = ovs_function(X_train_fold, y_train_fold, use_gpu)
        else:
            X_train_fold_oversampled = X_train_fold
            y_train_fold_oversampled = y_train_fold

        # Train model on the oversampled fold
        model.fit(X_train_fold_oversampled, y_train_fold_oversampled)

        # Predict probabilities
        try:
            if use_gpu:
                y_proba = model.predict(X_val_fold).astype(np.float32)
                y_proba = 1 / (1 + np.exp(-y_proba))  # Apply Sigmoid to approximate probability                
            else:
                y_proba = model.predict_proba(X_val_fold)[:, 1]  # Use `predict_proba()` in CPU mode                
        except AttributeError:
            print("⚠️ Warning: cuML MBSGDClassifier does not support `predict_proba()`, using sigmoid approximation.")

        # Convert to numpy and extract result
        y_proba = to_numpy_safe(y_proba)

        # Ensure y_val_fold is also a NumPy array before evaluation
        y_val_fold = to_numpy_safe(y_val_fold)

        # Evaluate the model using the specified metric
        evaluation_score = evaluate_model_metric(y_val_fold, y_proba, train_params)

        evaluation_scores.append(evaluation_score)

    return np.mean(evaluation_scores)


def optimize_sgd(X_train, y_train, train_params):
    """
    Runs Optuna optimization for the K-Nearest Neighbors (KNN) classifier.

    Args:
        X_train (cuDF.DataFrame or pandas.DataFrame): Training dataset.
        y_train (cuDF.Series or pandas.Series): Training labels.
        train_params (dict): Dictionary containing training parameters, including:
            - "device" (str): Device for training. Options: ["gpu", "cpu"].
            - "trials" (int): Number of optimization trials.
            - "metric" (str): Evaluation metric to optimize. Options: ["pr_auc", "f1", "precision", "cost"].
            - "cost_fp" (float, optional): Cost of a false positive (used if metric="cost").
            - "cost_fn" (float, optional): Cost of a false negative (used if metric="cost").
            - "jobs" (int): Number of parallel jobs (-1 to use all available cores).

    Returns:
        dict: The best hyperparameters found for SGD.
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
    save_filename = f"pt_{model_name}_{ovs_name}_{metric}.pkl"
    save_path = os.path.join(train_params["output_folder"], save_filename)

    # Start the timer to calculate training time
    timer.start()

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_sgd(trial, X_train, y_train, train_params), n_trials=n_trials, n_jobs=n_jobs
    )

    print(f"🔥 Best SGD Parameters ({metric}):", study.best_params)
    print(f"🔥 Best SGD Value ({metric}):", study.best_value)

    # Retrain the best model using the full dataset
    if use_gpu:
        best_model = MBSGDClassifier(**study.best_params)
    else:
        best_params = {
            k: v for k, v in study.best_params.items() if k != "batch_size"
        }  # Remove batch_size for CPU
        best_model = SGDClassifier(loss="log_loss", **best_params)

    best_model.fit(X_train, y_train)

    # Total execution time
    elapsed_time = round(timer.elapsed_final(), 2)
    print(f"📊 Total training time: {elapsed_time}")

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best SGD model saved at: {save_path}")

   # Save training performance details to CSV
    save_time_performance(train_params, study.best_value, elapsed_time)

    return study.best_params
