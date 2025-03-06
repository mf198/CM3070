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
    params = {
        "loss": "log",  # Logistic Regression
        "eta0": trial.suggest_float("eta0", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": trial.suggest_int("epochs", 100, 2000, step=200),
    }

    model = MBSGDClassifier(**params) if use_gpu else SGDClassifier(loss="log_loss", learning_rate="constant", **params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X_train.to_pandas(), y_train.to_pandas()):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Convert to NumPy if using CPU
        if not use_gpu:
            X_train_fold, X_val_fold = X_train_fold.to_numpy(), X_val_fold.to_numpy()
            y_train_fold, y_val_fold = y_train_fold.to_numpy(), y_train_fold.to_numpy()

        model.fit(X_train_fold, y_train_fold)
        
        # Use predict_proba if available, otherwise fallback to predict
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val_fold)[:, 1]
        else:
            y_proba = model.predict(X_val_fold)

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
    best_model = MBSGDClassifier(**study.best_params) if use_gpu else SGDClassifier(loss="log_loss", learning_rate="constant", **study.best_params)
    best_model.fit(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best SGD model saved at: {save_path}")

    return study.best_params
