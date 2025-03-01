import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

def compute_curve_values(y_test, y_proba, curve_type="pr_curve", cost_fp=1, cost_fn=10):
    """
    Computes Precision-Recall, ROC, or Cost-Based curve values.

    Args:
        y_test (array-like): True class labels.
        y_proba (array-like): Predicted probabilities for the positive class (fraud).
        curve_type (str): Type of curve to compute. Options: "pr_curve", "roc_curve", "cost_based".
        cost_fp (float, optional): Cost of a false positive (for cost-based curve).
        cost_fn (float, optional): Cost of a false negative (for cost-based curve).

    Returns:
        tuple: (metric1, metric2, thresholds) based on the selected curve type.
    """

    if curve_type == "pr_curve":
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        return precision[:-1], recall[:-1], thresholds  # Remove last threshold

    elif curve_type == "roc_curve":
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        return tpr, fpr, thresholds  # ROC: True Positive Rate (TPR) vs False Positive Rate (FPR)

    elif curve_type == "cost_based":
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        cost_metric = cost_fp * (1 - precision) + cost_fn * (1 - recall)

        # Choose the threshold with the lowest cost
        best_idx = np.argmin(cost_metric)
        best_threshold = thresholds[best_idx]

        return cost_metric[:-1], thresholds, thresholds

    else:
        raise ValueError("Invalid curve type. Choose 'pr_curve', 'roc_curve', or 'cost_based'.")


def find_best_threshold(metric1, metric2, thresholds, curve_type="pr_curve"):
    """
    Finds the best threshold from computed curve values.

    Args:
        metric1 (array-like): Precision (for PR curve), TPR (for ROC curve), or Cost Metric (for cost-based).
        metric2 (array-like): Recall (for PR curve), FPR (for ROC curve), or empty for cost-based.
        thresholds (array-like): Corresponding thresholds.
        curve_type (str): Type of curve used. Options: "pr_curve", "roc_curve", "cost_based".

    Returns:
        float: Optimal threshold.
    """
    if curve_type == "pr_curve":
        best_idx = np.argmin(np.abs(metric1 - metric2))  # Find where Precision ≈ Recall
    elif curve_type == "roc_curve":
        best_idx = np.argmax(metric1 - metric2)  # Maximize TPR - FPR
    elif curve_type == "cost_based":
        best_idx = np.argmin(metric1)  # Find the minimum cost point
    else:
        raise ValueError("Invalid curve type. Choose 'pr_curve', 'roc_curve', or 'cost_based'.")

    best_threshold = thresholds[best_idx]
    print(f"🏆 Best {curve_type.upper()} Threshold: {best_threshold:.4f} ")

    return best_threshold
