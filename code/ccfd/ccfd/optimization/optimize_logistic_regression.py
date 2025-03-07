import optuna
import cudf
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from cuml.metrics import roc_auc_score as cu_roc_auc_score
from sklearn.metrics import roc_auc_score as sk_roc_auc_score


def sigmoid(x):
    """Computes sigmoid activation to map logits to probabilities."""
    return 1 / (1 + np.exp(-x))


def objective_logistic_regression(trial, X_train, y_train, use_gpu=True):
    """
    Optuna objective function to optimize Logistic Regression.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train: Training dataset (cudf.DataFrame or pandas.DataFrame).
        y_train: Training labels (cudf.Series or pandas.Series).
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).

    Returns:
        float: Average AUC score across K-Folds.
    """
    if use_gpu:
        solver = "qn"  # cuML only supports 'qn'
    else:
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga", "sag", "newton-cg", "newton-cholesky"])

    params = {
        "penalty": "l2",
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "solver": solver,
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=100),
    }

    model = cuLogisticRegression(**params) if use_gpu else skLogisticRegression(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    # Ensure correct format for CPU/GPU mode
    if use_gpu:
        X_train_np, y_train_np = X_train.to_pandas().to_numpy(), y_train.to_pandas().to_numpy()
    else:
        X_train_np, y_train_np = X_train.values, y_train.values  # Direct conversion for pandas

    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
        if use_gpu:
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        else:
            X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
            y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

        model.fit(X_train_fold, y_train_fold)

        # Handle cuML predict_proba() issues
        try:
            y_proba = model.predict_proba(X_val_fold)
            
            # Check if y_proba is empty or has incorrect shape
            if y_proba.shape[0] == 0 or len(y_proba.shape) == 1:
                raise ValueError("predict_proba() returned an empty or incorrectly shaped array")

            # Ensure it has two columns (probabilities for both classes)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]  # Extract positive class probability
            else:
                raise ValueError("predict_proba() did not return two columns")

        except (AttributeError, ValueError):
            # Fallback: Use predict() and apply sigmoid activation
            y_proba = model.predict(X_val_fold).astype(float)
            y_proba = sigmoid(y_proba)  # Convert logits to probabilities

        # Compute AUC score using the correct metric
        auc = cu_roc_auc_score(y_val_fold, y_proba) if use_gpu else sk_roc_auc_score(y_val_fold, y_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)


def optimize_logistic_regression(X_train, y_train, n_trials=50, use_gpu=True, save_path="ccfd/optimized_models/best_logistic_regression.pkl"):
    """
    Runs Optuna optimization for Logistic Regression.

    Args:
        X_train: Training dataset (cudf.DataFrame or pandas.DataFrame).
        y_train: Training labels (cudf.Series or pandas.Series).
        n_trials (int): Number of optimization trials.
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).
        save_path (str): Path to save the best model.

    Returns:
        dict: Best Logistic Regression hyperparameters.
    """
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_logistic_regression(trial, X_train, y_train, use_gpu), n_trials=n_trials)

    print("🔥 Best Logistic Regression Parameters:", study.best_params)

    # Retrain the best model using the full dataset
    best_model = cuLogisticRegression(**study.best_params) if use_gpu else skLogisticRegression(**study.best_params)

    # Ensure correct data format before fitting
    if use_gpu:
        best_model.fit(X_train, y_train)  # Keep cuDF format
    else:
        best_model.fit(X_train.values, y_train.values)  # Convert pandas to NumPy

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best Logistic Regression model saved at: {save_path}")

    return study.best_params
