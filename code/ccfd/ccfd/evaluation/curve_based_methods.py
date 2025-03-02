import numpy as np
from sklearn.metrics import precision_recall_curve

def find_best_threshold_pr(y_test, y_proba):
    """
    Finds the best decision threshold using the Precision-Recall (PR) Curve.

    Args:
        y_test (array-like): True class labels.
        y_proba (array-like): Predicted probabilities for the positive class (fraud).

    Returns:
        float: Optimal threshold for classification.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)   

    # Find the threshold where Precision ≈ Recall
    best_idx = np.argmin(np.abs(precision - recall))
    best_threshold = thresholds[best_idx]

    print(f"🏆 Best PR Threshold: {best_threshold:.4f} (Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f})")
    return best_threshold



def find_best_threshold_cost(y_test, y_proba, cost_fp=1, cost_fn=10):
    """
    Finds the best threshold using a cost-sensitive approach.

    Args:
        y_test (array-like): True class labels.
        y_proba (array-like): Predicted probabilities for the positive class (fraud).
        cost_fp (float): Cost of a false positive (legitimate transaction flagged as fraud).
        cost_fn (float): Cost of a false negative (missed fraud).

    Returns:
        float: Optimal threshold based on fraud costs.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Compute cost-sensitive metric
    cost_metric = cost_fp * (1 - precision) + cost_fn * (1 - recall)

    # Choose the threshold with the lowest cost
    best_idx = np.argmin(cost_metric)
    best_threshold = thresholds[best_idx]

    print(f"🏆 Best Cost-Based Threshold: {best_threshold:.4f} (Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f})")
    return best_threshold
