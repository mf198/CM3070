import cudf
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from ccfd.utils.type_converter import to_numpy_safe


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

    if isinstance(y_test, cudf.Series):
        y_test = y_test.astype(
            "int32"
        ).to_numpy()  # Ensure cuDF → NumPy with integer format
    else:
        y_test = np.array(y_test, dtype=int)

    if curve_type == "pr_curve":
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        return precision[:-1], recall[:-1], thresholds  # Remove last threshold

    elif curve_type == "roc_curve":
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        return (
            tpr,
            fpr,
            thresholds,
        )  # ROC: True Positive Rate (TPR) vs False Positive Rate (FPR)

    elif curve_type == "cost_based":
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        cost_metric = cost_fp * (1 - precision) + cost_fn * (1 - recall)
        return cost_metric[:-1], thresholds, thresholds  # Return cost metric values

    elif curve_type == "percentile":
        return (
            y_proba,
            None,
            y_proba,
        )  # Directly return probabilities to apply percentile thresholding

    else:
        raise ValueError(
            "Invalid curve type. Choose 'pr_curve', 'roc_curve', 'cost_based', or 'percentile'."
        )


def find_best_threshold(
    metric1, metric2, thresholds, curve_type="pr_curve", percentile=99
):
    """
    Finds the best threshold from computed curve values.

    Args:
        metric1 (array-like): Precision (for PR curve), TPR (for ROC curve), or Cost Metric (for cost-based).
        metric2 (array-like): Recall (for PR curve), FPR (for ROC curve), or empty for cost-based.
        thresholds (array-like): Corresponding thresholds.
        curve_type (str): Type of curve used. Options: "pr_curve", "roc_curve", "cost_based", "percentile".
        percentile (int, optional): Percentile threshold for "percentile" method. Default is 99.

    Returns:
        float: Optimal threshold.
    """
    metric1 = np.array(metric1)
    metric2 = np.array(metric2) if metric2 is not None else None
    thresholds = np.array(thresholds)

    if curve_type == "pr_curve":
        f1_scores = np.divide(
            2 * metric1 * metric2,
            metric1 + metric2,
            where=(metric1 + metric2) > 0,  # Avoid division where sum is zero
            out=np.zeros_like(
                metric1
            ),  # Default F1-score = 0 where division is not valid
        )

        best_idx = np.argmax(f1_scores)  # Find the index of max F1-score

    elif curve_type == "roc_curve":
        best_idx = np.argmax(metric1 - metric2)  # Maximize TPR - FPR

    elif curve_type == "cost_based":
        best_idx = np.argmin(metric1)  # Find the minimum cost point

    elif curve_type == "percentile":
        best_threshold = np.percentile(
            metric1, percentile
        )  # Select threshold at the chosen percentile
        print(
            f"Best Percentile-Based Threshold ({percentile}th percentile): {best_threshold:.4f}"
        )
        return best_threshold

    else:
        raise ValueError(
            "Invalid curve type. Choose 'pr_curve', 'roc_curve', 'cost_based', or 'percentile'."
        )

    best_threshold = thresholds[best_idx]
    print(f"Best {curve_type.upper()} Threshold: {best_threshold:.4f}")
    return best_threshold
