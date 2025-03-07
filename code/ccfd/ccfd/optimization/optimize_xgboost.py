import optuna
import cudf
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def objective_xgboost(trial, X_train, y_train, use_gpu=True):
    """
    Optuna objective function to optimize XGBoost hyperparameters.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train: Training dataset (cuDF.DataFrame or pandas.DataFrame).
        y_train: Training labels (cuDF.Series or pandas.Series).
        use_gpu (bool): Whether to use GPU (CUDA) or CPU.

    Returns:
        float: Average AUC score across K-Folds.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
    }

    # Ensure data consistency (convert to NumPy for XGBoost)
    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas().to_numpy()
    if isinstance(y_train, cudf.Series):
        y_train = y_train.to_pandas().to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        if use_gpu == True:
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        else:
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]        

        # Convert to DMatrix for GPU training
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dval, "eval")],
            early_stopping_rounds=50,  # Enables `best_iteration`
            verbose_eval=False,
        )

        # Use best_iteration only if early stopping is enabled
        best_iteration = model.best_iteration if model.best_iteration else 5000

        # Predict using `best_iteration` (if available)
        y_proba = model.predict(dval, iteration_range=(0, best_iteration))

        # Compute AUC
        auc = roc_auc_score(y_val_fold, y_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)


def optimize_xgboost(
    X_train,
    y_train,
    n_trials=50,
    use_gpu=True,
    save_path="ccfd/optimized_models/best_xgboost.pkl",
):
    """
    Runs Optuna optimization for XGBoost.

    Args:
        X_train: Training dataset (cuDF.DataFrame or pandas.DataFrame).
        y_train: Training labels (cuDF.Series or pandas.Series).
        n_trials (int): Number of optimization trials.
        use_gpu (bool): Whether to use GPU (CUDA) or CPU.
        save_path (str): Path to save the best model.

    Returns:
        dict: Best XGBoost hyperparameters.
    """
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, use_gpu),
        n_trials=n_trials,
    )

    print("🔥 Best XGBoost Parameters:", study.best_params)

    # Convert full dataset to NumPy (required for XGBoost)
    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas().to_numpy()
    if isinstance(y_train, cudf.Series):
        y_train = y_train.to_pandas().to_numpy()

    # ✅ Retrain the best model using the full dataset
    best_model = xgb.XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    # ✅ Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best XGBoost model saved at: {save_path}")

    return study.best_params
