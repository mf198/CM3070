import optuna
import cudf
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from cuml.metrics import roc_auc_score as cu_roc_auc_score
from sklearn.metrics import roc_auc_score as sk_roc_auc_score


def objective_random_forest(trial, X_train, y_train, use_gpu=True):
    """
    Optuna objective function to optimize Random Forest.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train: Training dataset (cudf.DataFrame or pandas.DataFrame).
        y_train: Training labels (cudf.Series or pandas.Series).
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).

    Returns:
        float: Average AUC score across K-Folds.
    """
    # Define parameters separately for GPU and CPU
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
    }

    if use_gpu:
        params["n_bins"] = trial.suggest_int("n_bins", 32, 256, step=32)  # Only for cuML

    model = cuRandomForestClassifier(**params) if use_gpu else skRandomForestClassifier(**params)

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

            # Check if y_proba is empty or incorrectly shaped
            if y_proba.shape[0] == 0 or len(y_proba.shape) == 1:
                raise ValueError("predict_proba() returned an empty or incorrectly shaped array")

            # Ensure two columns (probabilities for both classes)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]  # Extract positive class probability
            else:
                raise ValueError("predict_proba() did not return two columns")

        except (AttributeError, ValueError):
            # Fallback: Use predict() and map outputs to probabilities
            y_proba = model.predict(X_val_fold).astype(float)

        # Compute AUC score using the correct metric
        auc = cu_roc_auc_score(y_val_fold, y_proba) if use_gpu else sk_roc_auc_score(y_val_fold, y_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)


def optimize_random_forest(X_train, y_train, n_trials=50, use_gpu=True, save_path="ccfd/optimized_models/best_random_forest.pkl"):
    """
    Runs Optuna optimization for Random Forest.

    Args:
        X_train: Training dataset (cudf.DataFrame or pandas.DataFrame).
        y_train: Training labels (cudf.Series or pandas.Series).
        n_trials (int): Number of optimization trials.
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).
        save_path (str): Path to save the best model.

    Returns:
        dict: Best Random Forest hyperparameters.
    """
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_random_forest(trial, X_train, y_train, use_gpu), n_trials=n_trials)

    print("🔥 Best Random Forest Parameters:", study.best_params)

    # Retrain the best model using the full dataset
    if use_gpu:
        best_model = cuRandomForestClassifier(**study.best_params)
    else:
        best_params = {k: v for k, v in study.best_params.items() if k != "n_bins"}  # Remove 'n_bins' for CPU
        best_model = skRandomForestClassifier(**best_params)

    best_model.fit(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best Random Forest model saved at: {save_path}")

    return study.best_params
