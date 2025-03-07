import optuna
import cudf
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from cuml.linear_model import MBSGDClassifier
from sklearn.model_selection import StratifiedKFold
from cuml.metrics import roc_auc_score as cu_roc_auc_score
from sklearn.metrics import roc_auc_score as sk_roc_auc_score


def objective_sgd(trial, X_train, y_train, use_gpu=True):
    """
    Optuna objective function to optimize Stochastic Gradient Descent (SGD) classifier.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train: Training dataset (cudf.DataFrame or pandas.DataFrame).
        y_train: Training labels (cudf.Series or pandas.Series).
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).

    Returns:
        float: Average AUC score across K-Folds.
    """
    if use_gpu:
        params = {
            "loss": "log",  # Logistic Regression
            "eta0": trial.suggest_float("eta0", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "epochs": trial.suggest_int("epochs", 100, 1900, step=200),  # cuML uses "epochs"
        }
        model = MBSGDClassifier(**params)
    else:
        params = {
            "eta0": trial.suggest_float("eta0", 1e-5, 1e-1, log=True),
            "learning_rate": "constant",
            "max_iter": trial.suggest_int("max_iter", 100, 1900, step=200),  # CPU uses "max_iter"
        }
        model = SGDClassifier(loss="log_loss", **params)  # No "epochs" in scikit-learn

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

        # Use predict_proba if available, otherwise fallback to predict
        try:
            y_proba = model.predict_proba(X_val_fold)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]  # Extract positive class probability
            else:
                raise ValueError("predict_proba() did not return two columns")
        except (AttributeError, ValueError):
            y_proba = model.predict(X_val_fold).astype(float)  # Fallback

        # Compute AUC score
        auc = cu_roc_auc_score(y_val_fold, y_proba) if use_gpu else sk_roc_auc_score(y_val_fold, y_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)


def optimize_sgd(X_train, y_train, n_trials=50, use_gpu=True, save_path="ccfd/optimized_models/best_sgd.pkl"):
    """
    Runs Optuna optimization for SGD.

    Args:
        X_train: Training dataset (cudf.DataFrame or pandas.DataFrame).
        y_train: Training labels (cudf.Series or pandas.Series).
        n_trials (int): Number of optimization trials.
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).
        save_path (str): Path to save the best model.

    Returns:
        dict: Best SGD hyperparameters.
    """
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_sgd(trial, X_train, y_train, use_gpu), n_trials=n_trials)

    print("🔥 Best SGD Parameters:", study.best_params)

    # Retrain the best model using the full dataset
    if use_gpu:
        best_model = MBSGDClassifier(**study.best_params)
    else:
        best_params = {k: v for k, v in study.best_params.items() if k != "batch_size"}  # Remove batch_size for CPU
        best_model = SGDClassifier(loss="log_loss", **best_params)

    best_model.fit(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best SGD model saved at: {save_path}")

    return study.best_params
