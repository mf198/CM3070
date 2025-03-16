import optuna
import cudf
import cupy as cp
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from ccfd.evaluation.evaluate_models import evaluate_model_metric
from ccfd.utils.type_converter import to_numpy_safe
from ccfd.utils.time_performance import save_time_performance
from ccfd.utils.timer import Timer


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
    ovs_function = train_params["oversampling_function"]

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

        # Apply an oversampling method if selected
        if ovs_function:
            X_train_fold_oversampled, y_train_fold_oversampled = ovs_function(X_train_fold, y_train_fold, use_gpu)
        else:
            X_train_fold_oversampled = X_train_fold
            y_train_fold_oversampled = y_train_fold

        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(to_numpy_safe(X_train_fold_oversampled), label=to_numpy_safe(y_train_fold_oversampled))
        dval = xgb.DMatrix(to_numpy_safe(X_val_fold), label=to_numpy_safe(y_val_fold))

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dval, "eval")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Get the best iteration (if early stopping triggered)
        best_iteration = model.best_iteration if model.best_iteration else 5000

        # Use the best iteration for prediction
        y_proba = model.predict(dval, iteration_range=(0, best_iteration))   

        # Convert labels to NumPy
        y_val_fold = to_numpy_safe(y_val_fold)

        # Evaluate the model using the specified metric
        evaluation_score = evaluate_model_metric(y_val_fold, y_proba, train_params)

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
    timer = Timer()

    use_gpu = train_params["device"] == "gpu"
    n_trials = train_params["trials"]
    metric = train_params["metric"]
    n_jobs = train_params["jobs"]
    output_folder = train_params["output_folder"]

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define model save path dynamically
    save_path = os.path.join(output_folder, "pt_xbgoost.pkl")    

    # Start the timer to calculate training time
    timer.start()

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, train_params),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print(f"🔥 Best XGBoost Parameters ({metric}):", study.best_params)
    print(f"🔥 Best XGBoost Value ({metric}):", study.best_value)

    # Convert full dataset to NumPy (required for XGBoost)
    X_train = to_numpy_safe(X_train)
    y_train = to_numpy_safe(y_train)

    # Retrain the best model using the full dataset
    best_model = xgb.XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    # Total execution time
    elapsed_time = round(timer.elapsed_final(), 2)
    print(f"📊 Total training time: {elapsed_time}")

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best XGBoost model saved at: {save_path}")

   # Save training performance details to CSV
    save_time_performance(train_params, study.best_value, elapsed_time)

    return study.best_params
