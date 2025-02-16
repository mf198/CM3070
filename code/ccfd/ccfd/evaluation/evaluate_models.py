import cudf
import pandas as pd
import numpy as np
from cuml.metrics import roc_auc_score as roc_auc_score_gpu
from cuml.metrics import accuracy_score as accuracy_score_gpu
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)
from typing import Dict, Optional
from ccfd.evaluation.curve_based_methods import find_best_threshold_cost, find_best_threshold_pr


def compute_metrics(y_test, y_pred, y_proba, use_gpu=False) -> Dict[str, float]:
    """
    Computes and returns evaluation metrics for classification models.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        use_gpu (bool, optional): If True, uses cuML (GPU) metrics. Otherwise, uses scikit-learn (CPU).

    Returns:
        Dict[str, float]: Dictionary containing accuracy, ROC AUC, F1-score, precision, and recall.
    """

    # Compute metrics using cuML (GPU) or scikit-learn (CPU)
    if use_gpu:
        metrics = {
            "accuracy": accuracy_score_gpu(y_test, y_pred),
            "roc_auc": roc_auc_score_gpu(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
    else:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

    return metrics


###


def evaluate_model_cpu(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_method: Optional[str] = "default",
    cost_fp: float = 1,
    cost_fn: float = 10,
) -> Dict[str, float]:
    """
    Evaluates the model on the test set using GPU (cuDF/cuML) or CPU (pandas/scikit-learn).

    Args:
        model: Trained model.
        X_test (cudf.DataFrame or pd.DataFrame): Test features.
        y_test (cudf.Series, np.ndarray, or pd.Series): True labels.
        threshold_method (str, optional): Curve-based threshold selection method.
                                          Options: "default" (0.5), "pr_curve", "cost_based".
        cost_fp (float, optional): Cost of a false positive (for cost-based optimization).
        cost_fn (float, optional): Cost of a false negative (for cost-based optimization).

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    # Predict labels
    y_pred = model.predict(X_test)

    print(f"Pred: {y_pred}")

    # Convert y_pred based on type
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)  # Convert NumPy to Pandas

    # Convert y_test to Pandas if needed
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)  # Convert NumPy to Pandas

    # Handle `predict_proba()` for probability-based metrics
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            if isinstance(y_proba, np.ndarray):
                y_proba = pd.Series(y_proba[:, 1])  # Convert NumPy to Pandas
            else:
                y_proba = y_pred  # Default to predictions if empty
        except Exception:
            print(
                f"Warning: {model.__class__.__name__} does not support predict_proba(). Using raw predictions."
            )
            y_proba = y_pred  # Default to predictions

    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
            if isinstance(y_proba, np.ndarray):
                y_proba = pd.Series(y_proba)  # Convert NumPy to Pandas
            else:
                y_proba = y_proba.to_pandas()
        except Exception:
            print(
                f"Warning: {model.__class__.__name__} does not support decision_function(). Using raw predictions."
            )
            y_proba = y_pred  # Default to predictions

    else:
        print(
            f"Warning: {model.__class__.__name__} does not support probability outputs. Using raw predictions."
        )
        y_proba = y_pred  # Default to predictions

    # Select threshold based on method
    if threshold_method == "pr_curve":
        best_threshold = find_best_threshold_pr(y_test, y_proba)
    elif threshold_method == "cost_based":
        best_threshold = find_best_threshold_cost(y_test, y_proba, cost_fp, cost_fn)
    else:
        best_threshold = 0.5  # Default threshold

    print(f"🔍 Using threshold: {best_threshold:.4f}")

    # Compute evaluation metrics (scikit-learn for CPU)
    return compute_metrics(y_test, y_pred, y_proba, False)


def evaluate_model_gpu(
    model,
    X_test: cudf.DataFrame,
    y_test: cudf.Series,
    threshold_method: Optional[str] = "default",
    cost_fp: float = 1,
    cost_fn: float = 10,
) -> Dict[str, float]:
    """
    Evaluates the GPU-based model on the test set.

    Args:
        model: Trained model.
        X_test (cudf.DataFrame): Test features.
        y_test: True labels (cuDF Series or NumPy array).
        threshold_method (str, optional): Curve-based threshold selection method.
                                          Options: "default" (0.5), "pr_curve", "cost_based".
        cost_fp (float, optional): Cost of a false positive (for cost-based optimization).
        cost_fn (float, optional): Cost of a false negative (for cost-based optimization).

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """

    y_pred = model.predict(X_test)  # XGBoost returns a NumPy array
    if isinstance(y_pred, np.ndarray):
        y_pred = cudf.Series(y_pred)  # Convert NumPy to cuDF

    # Convert X_test to a numpy array which is compatible with all the models
    if isinstance(y_pred, np.ndarray) is False:
        X_test = X_test.to_numpy()

    # Handle `predict_proba()`
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)

            if isinstance(y_proba, np.ndarray) is True:
                y_proba = y_proba[:, 1]
        except Exception:
            print(
                f"Warning: {model.__class__.__name__} does not support predict_proba(). Using raw predictions."
            )
            y_proba = y_pred  # Default to predictions

    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            print(
                f"Warning: {model.__class__.__name__} does not support decision_function(). Using raw predictions."
            )
            y_proba = y_pred  # Default to predictions

    else:
        print(
            f"Warning: {model.__class__.__name__} does not support probability outputs. Using raw predictions."
        )
        y_proba = y_pred  # Default to predictions

    print(f"proba: {y_proba.min()}, {y_proba.max()}")

    # Convert y_proba to a numpy array to use roc_curve
    if isinstance(y_proba, np.ndarray) is False:
        y_proba = y_proba.to_numpy()

    y_pred = y_pred.to_pandas()
    y_test = y_test.to_pandas()

    # Select threshold based on method
    if threshold_method == "pr_curve":
        best_threshold = find_best_threshold_pr(y_test, y_proba)
    elif threshold_method == "cost_based":
        best_threshold = find_best_threshold_cost(y_test, y_proba, cost_fp, cost_fn)
    else:
        best_threshold = 0.5  # Default threshold

    print(f"🔍 Using threshold: {best_threshold:.4f}")

    # Convert probabilities to binary predictions using best threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    # Compute evaluation metrics (GPU)
    return compute_metrics(y_test, y_pred, y_proba, True)


###
