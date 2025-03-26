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
            "pr_auc": average_precision_score(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "selected_threshold": best_threshold,
        }
    else:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "selected_threshold": best_threshold,
        }

    return metrics


###


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, test_params: Dict
) -> Dict[str, float]:
    """
    Evaluates model performance based on the chosen metric.

    Args:
        y_true (np.ndarray): True labels as a NumPy array.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        test_params (dict): Dictionary containing evaluation parameters, including:
            - "threshold" (float, optional): Fixed threshold for classification.
            - "threshold_method" (str): Threshold selection method ("default", "pr_curve", "cost_based").
            - "cost_fp" (float): Cost of a false positive (for cost-based optimization).
            - "cost_fn" (float): Cost of a false negative (for cost-based optimization).
            - "use_gpu" (bool): If True, uses GPU (cuML/cudf), otherwise uses CPU (sklearn/pandas).

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    use_gpu = test_params.get("device", False) == "gpu"
    eval_method = test_params.get("eval_method", "default")
    cost_fp = test_params.get("cost_fp", 1)
    cost_fn = test_params.get("cost_fn", 3)

    # Check if a fixed threshold is provided
    if "threshold" in test_params and test_params["threshold"] is not None:
        best_threshold = test_params["threshold"]
        threshold_source = "manual override"

    elif eval_method == "pr_curve":
        # Compute Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        best_threshold = find_best_threshold(
            precision, recall, thresholds, curve_type="pr_curve"
        )
        threshold_source = "PR curve optimization"

    elif eval_method == "cost_based":
        # Compute Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        # Compute Cost-Based Metric
        cost_metric = (cost_fp * (1 - precision)) + (cost_fn * (1 - recall))

        best_threshold = find_best_threshold(
            cost_metric, [], thresholds, curve_type="cost_based"
        )
        threshold_source = "cost-based optimization"

    elif eval_method == "percentile":
        percentile = test_params.get("percentile", 99)
        best_threshold = np.percentile(y_proba, percentile)
        threshold_source = f"percentile-based ({percentile}th percentile)"

    else:
        best_threshold = 0.5  # Default threshold
        threshold_source = "default (0.5)"

    print(f"Using threshold: {best_threshold:.4f} ({threshold_source})")

    # Apply the Threshold to Convert Probabilities into Binary Labels
    y_pred_adjusted = (y_proba >= best_threshold).astype(int)

    # Return evaluation metrics
    return compute_metrics(y_true, y_pred_adjusted, y_proba, best_threshold, use_gpu)


def evaluate_model_metric(y_true, y_pred, train_params):
    """
    Evaluates model performance based on the chosen metric.

    Args:
        y_true (np.ndarray): True labels (0 = legitimate transaction, 1 = fraud).
        y_pred (np.ndarray): Model predicted scores or probabilities.
        train_params (dict): Dictionary containing training parameters, including:
            - "metric" (str): Evaluation metric to use. Options: ["prauc", "f1", "precision", "cost"].
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
        raise ValueError(
            "Invalid metric. Choose from ['prauc', 'f1', 'precision', 'cost']."
        )
