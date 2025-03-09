import optuna
import cudf
import cupy as cp
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from ccfd.evaluation.evaluate_models import evaluate_model
from ccfd.utils.type_converter import to_numpy_safe


def objective_xgboost(trial, X_train, y_train, train_params):
    """
    Optuna objective function to optimize XGBoost.

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

    # XGBoost-specific hyperparameters
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

        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(to_numpy_safe(X_train_fold), label=to_numpy_safe(y_train_fold))
        dval = xgb.DMatrix(to_numpy_safe(X_val_fold), label=to_numpy_safe(y_val_fold))

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

        # Convert labels to NumPy
        y_val_fold = to_numpy_safe(y_val_fold)

        # Evaluate the model using the specified metric
        evaluation_score = evaluate_model(y_val_fold, y_proba, train_params)

        evaluation_scores.append(evaluation_score)

    return np.mean(evaluation_scores)


def optimize_xgboost(
    X_train, y_train, train_params, save_path="ccfd/pretrained_models/pt_xgboost.pkl"
):
    """
    Runs Optuna optimization for XGBoost classifier.

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
        save_path (str, optional): Path to save the best model. Default: "ccfd/pretrained_models/pt_xgboost.pkl".

    Returns:
        dict: The best hyperparameters found for XGBoost.
    """

    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    metric = train_params["metric"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define model save path dynamically
    save_path = os.path.join(output_folder, "pt_xbgoost.pkl")    

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, train_params),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print(f"🔥 Best XGBoost Parameters ({metric}):", study.best_params)

    # Convert full dataset to NumPy (required for XGBoost)
    X_train = to_numpy_safe(X_train)
    y_train = to_numpy_safe(y_train)

    # Retrain the best model using the full dataset
    best_model = xgb.XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best XGBoost model saved at: {save_path}")

    return study.best_params
