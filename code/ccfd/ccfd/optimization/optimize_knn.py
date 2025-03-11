import optuna
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from cuml.neighbors import KNeighborsClassifier as cuKNN
from sklearn.neighbors import KNeighborsClassifier as skKNN
from ccfd.evaluation.evaluate_models import evaluate_model
from ccfd.utils.type_converter import to_numpy_safe


def objective_knn_old(trial, X_train, y_train, train_params):
    """
    Optuna objective function to optimize K-Nearest Neighbors (KNN).

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

    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
    }

    use_gpu = train_params["device"] == "gpu"

    model = cuKNN(**params) if use_gpu else skKNN(**params)

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

        model.fit(X_train_fold, y_train_fold)

        # Predict probabilities
        y_proba = model.predict_proba(X_val_fold)

        # Convert to numpy and extract result
        y_proba = to_numpy_safe(y_proba)[:, 1]

        # Ensure y_val_fold is also a NumPy array before evaluation
        y_val_fold = to_numpy_safe(y_val_fold)

        # Evaluate the model using the specified metric
        evaluation_score = evaluate_model(y_val_fold, y_proba, train_params)

        evaluation_scores.append(evaluation_score)

    return np.mean(evaluation_scores)

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
            - "cost_fp" (float, optional): Cost of a false positive (used if metric="cost").
            - "cost_fn" (float, optional): Cost of a false negative (used if metric="cost").

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

        # Predict probabilities on the validation fold
        y_proba = model.predict_proba(X_val_fold)

        # Convert to numpy and extract result
        y_proba = to_numpy_safe(y_proba)[:, 1]

        # Ensure y_val_fold is also a NumPy array before evaluation
        y_val_fold = to_numpy_safe(y_val_fold)

        # Evaluate the model using the specified metric
        evaluation_score = evaluate_model(y_val_fold, y_proba, train_params)

        evaluation_scores.append(evaluation_score)

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
            - "cost_fp" (float, optional): Cost of a false positive (used if metric="cost").
            - "cost_fn" (float, optional): Cost of a false negative (used if metric="cost").
            - "jobs" (int): Number of parallel jobs (-1 to use all available cores).        

    Returns:
        dict: The best hyperparameters found for KNN.
    """

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

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_knn(trial, X_train, y_train, train_params),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print(f"🔥 Best KNN Parameters ({metric}):", study.best_params)

    # Retrain the best model using the full dataset
    best_model = cuKNN(**study.best_params) if use_gpu else skKNN(**study.best_params)

    # Data fit
    best_model.fit(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best KNN model saved at: {save_path}")

    return study.best_params
