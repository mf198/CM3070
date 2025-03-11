import optuna
import cudf
import cupy as cp
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from ccfd.evaluation.evaluate_models import evaluate_model
from ccfd.utils.type_converter import to_numpy_safe

def sigmoid(x):
    """Computes sigmoid activation to map logits to probabilities."""
    return 1 / (1 + np.exp(-x))

def objective_logistic_regression(trial, X_train, y_train, train_params):
    """
    Optuna objective function to optimize Logistic Regression.

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
    evaluation_scores = []

    # Ensure correct format for CPU/GPU mode
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
        y_proba = model.predict_proba(X_val_fold)
        
        # Convert to numpy and extract result
        y_proba = to_numpy_safe(y_proba)[:, 1]
        
        # Ensure y_val_fold is also a NumPy array before evaluation
        y_val_fold = to_numpy_safe(y_val_fold)

        # Evaluate the model using the specified metric
        evaluation_score = evaluate_model(y_val_fold, y_proba, train_params)

        evaluation_scores.append(evaluation_score)

    return np.mean(evaluation_scores)  # Optuna optimizes based on the selected metric


def optimize_logistic_regression(
    X_train, y_train, train_params):
    """
    Runs Optuna optimization for Logistic Regression.

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
        save_path (str, optional): Path to save the best model. Default: "ccfd/pretrained_models/pt_logistic_regression.pkl".

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

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_logistic_regression(trial, X_train, y_train, train_params),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print(f"🔥 Best Logistic Regression Parameters ({metric}):", study.best_params)

    # Retrain the best model using the full dataset
    best_model = cuLogisticRegression(**study.best_params) if use_gpu else skLogisticRegression(**study.best_params)

    # Model fit
    best_model.fit(X_train, y_train)  # Keep cuDF format
    #best_model.fit(X_train.values, y_train.values)  # Convert pandas to NumPy

    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best Logistic Regression model saved at: {save_path}")

    return study.best_params
