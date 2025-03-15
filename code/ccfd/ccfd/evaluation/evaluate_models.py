import cudf
import pandas as pd
import numpy as np
from cuml.metrics import roc_auc_score as roc_auc_score_gpu
from cuml.metrics import accuracy_score as accuracy_score_gpu
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)
from typing import Dict, Optional
from ccfd.evaluation.threshold_analysis import compute_curve_values, find_best_threshold
from ccfd.evaluation.save_curves import save_selected_curve

def compute_metrics(
    y_test, y_pred, y_proba, best_threshold, use_gpu=False
) -> Dict[str, float]:
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
            "selected_threshold": best_threshold,
        }
    else:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "selected_threshold": best_threshold,
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
    return compute_metrics(y_test, y_pred, y_proba, best_threshold, False)


def evaluate_model_gpu(
    model,
    X_test: cudf.DataFrame,
    y_test: cudf.Series,
    threshold_method: Optional[str] = "default",
    cost_fp: float = 1,
    cost_fn: float = 10,
    save_curve: bool = False,
    output_file: str = "ccfd/results/curves.csv",
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
    if isinstance(X_test, np.ndarray) is False:
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

    # Convert y_proba to NumPy format
    if isinstance(y_proba, np.ndarray) is False:
        y_proba = y_proba.to_numpy()

    y_pred = y_pred.to_pandas()
    y_test = y_test.to_pandas()

    # Select the best threshold based on method
    if threshold_method in ["pr_curve", "roc_curve", "cost_based"]:
        metric1, metric2, thresholds = compute_curve_values(
            y_test,
            y_proba,
            curve_type=threshold_method,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )
        best_threshold = find_best_threshold(
            metric1, metric2, thresholds, curve_type=threshold_method
        )
    else:
        best_threshold = 0.5  # Default threshold

    print(f"🔍 Using threshold: {best_threshold:.4f}")

    # Convert probabilities to binary predictions using best threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    # Save selected curve if requested
    if save_curve and threshold_method in ["pr_curve", "roc_curve", "cost_based"]:
        save_selected_curve(
            y_test, y_proba, output_file, threshold_method, cost_fp, cost_fn
        )

    # Compute evaluation metrics (GPU)
    return compute_metrics(y_test, y_pred, y_proba, best_threshold, True)


###

def evaluate_model(y_true, y_pred, train_params):
    """
    Evaluates model performance based on the chosen metric.

    Args:
        y_true (np.ndarray): True labels (0 = legitimate transaction, 1 = fraud).
        y_pred (np.ndarray): Model predicted scores or probabilities.
        train_params (dict): Dictionary containing training parameters, including:
            - "metric" (str): Evaluation metric to use. Options: ["pr_auc", "f1", "precision", "cost"].
            - "cost_fp" (float, optional): Cost of a false positive (legitimate transaction flagged as fraud).
            - "cost_fn" (float, optional): Cost of a false negative (fraudulent transaction not detected).
              Only required if "metric" is set to "cost".

    Returns:
        float: The computed metric value based on the selected evaluation method.
    """

    # Ensure y_true and y_pred are NumPy arrays (in case they are lists)
    y_true = np.asarray(y_true, dtype=np.int32)  # Ensure labels are integer
    y_pred = np.asarray(y_pred, dtype=np.float32)  # Ensure probabilities are float

    # Convert probabilities to binary predictions (threshold = 0.5)
    y_binary = (y_pred >= 0.5).astype(int)

    metric = train_params["metric"]    

    if metric == "prauc":
        return average_precision_score(y_true, y_pred)  # PR AUC needs probabilities

    elif metric == "f1":
        return f1_score(y_true, y_binary)  # F1-score uses binary labels

    elif metric == "precision":
        return precision_score(y_true, y_binary)  # Precision uses binary labels
    
    elif metric == "recall":
        return recall_score(y_true, y_binary)  # Recall uses binary labels

    elif metric == "cost":
        cost_fp = train_params["cost_fp"]
        cost_fn = train_params["cost_fn"]

        # Compute cost function based on false positives & false negatives
        false_positives = np.sum((y_binary == 1) & (y_true == 0))  # False positives
        false_negatives = np.sum((y_binary == 0) & (y_true == 1))  # False negatives
        return (false_positives * cost_fp) + (false_negatives * cost_fn)

    else:
        raise ValueError("Invalid metric. Choose from ['prauc', 'f1', 'precision', 'cost'].")