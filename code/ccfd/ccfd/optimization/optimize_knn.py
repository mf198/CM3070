import optuna
import cudf
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from cuml.neighbors import KNeighborsClassifier as cuKNN
from sklearn.neighbors import KNeighborsClassifier as skKNN
from cuml.metrics import roc_auc_score as cu_roc_auc_score
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

def objective_knn(trial, X_train, y_train, use_gpu=True):
    """
    Optuna objective function to optimize KNN.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train: Training dataset (cuDF or pandas).
        y_train: Training labels (cuDF Series or pandas Series).
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).

    Returns:
        float: Average AUC score across K-Folds.
    """
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
    }

    model = cuKNN(**params) if use_gpu else skKNN(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    # Convert cuDF to NumPy only for CPU mode
    if use_gpu:
        X_train_np, y_train_np = X_train.to_pandas().to_numpy(), y_train.to_pandas().to_numpy()
    else:
        X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()  # Directly use pandas

    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
        if use_gpu:
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        else:
            X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
            y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

        model.fit(X_train_fold, y_train_fold)

        # Handle cuML's missing predict_proba
        try:
            y_proba = model.predict_proba(X_val_fold)[:, 1]
            if y_proba.shape[0] == 0:  # If empty, fallback to kneighbors()
                raise ValueError("predict_proba() returned empty array")
        except (AttributeError, ValueError):
            # Approximate probability using distances
            distances, indices = model.kneighbors(X_val_fold)
            y_proba = 1 / (1 + distances.mean(axis=1))  # Simple inverse-distance weighting

        auc = cu_roc_auc_score(y_val_fold, y_proba) if use_gpu else sk_roc_auc_score(y_val_fold, y_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)


def optimize_knn(X_train, y_train, n_trials=50, use_gpu=True, save_path="ccfd/optimized_models/best_knn.pkl"):
    """
    Runs Optuna optimization for KNN.

    Args:
        X_train: Training dataset (cuDF or pandas).
        y_train: Training labels (cuDF Series or pandas Series).
        n_trials (int): Number of optimization trials.
        use_gpu (bool): Whether to use GPU (cuML) or CPU (scikit-learn).
        save_path (str): Path to save the best model.

    Returns:
        dict: Best KNN hyperparameters.
    """
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_knn(trial, X_train, y_train, use_gpu), n_trials=n_trials)

    print("🔥 Best KNN Parameters:", study.best_params)

    # Retrain the best model using the full dataset
    best_model = cuKNN(**study.best_params) if use_gpu else skKNN(**study.best_params)

    # Ensure correct data format before fitting
    if use_gpu:
        best_model.fit(X_train, y_train)  # Keep cuDF format
    else:
        best_model.fit(X_train.to_numpy(), y_train.to_numpy())  # Convert pandas to NumPy

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best KNN model saved at: {save_path}")

    return study.best_params
