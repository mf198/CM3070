import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def save_selected_curve(
    y_test, y_proba, output_file, threshold_method="default", cost_fp=1, cost_fn=10
):
    """
    Computes and saves only the selected curve type (PR, ROC, or Cost-Based).

    Args:
        y_test (array-like): True labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        output_file (str): Path to save the CSV file.
        threshold_method (str): Selected threshold method. Options: "pr_curve", "roc_curve", "cost_based".
        cost_fp (float, optional): Cost of a false positive (for cost-based curve).
        cost_fn (float, optional): Cost of a false negative (for cost-based curve).

    Returns:
        None
    """

    if threshold_method == "pr_curve":
        # Compute Precision-Recall Curve
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
        df_pr = pd.DataFrame(
            {
                "threshold": np.append(thresholds_pr, 1.0),
                "precision": precision,
                "recall": recall,
            }
        )
        df_pr.to_csv(output_file.replace(".csv", "_pr.csv"), index=False)
        print(f"PR Curve saved to {output_file.replace('.csv', '_pr.csv')}")

    elif threshold_method == "roc_curve":
        # Compute ROC Curve
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
        df_roc = pd.DataFrame({"threshold": thresholds_roc, "fpr": fpr, "tpr": tpr})
        df_roc.to_csv(output_file.replace(".csv", "_roc.csv"), index=False)
        print(f"ROC Curve saved to {output_file.replace('.csv', '_roc.csv')}")

    elif threshold_method == "cost_based":
        # Compute Precision-Recall for Cost-Based Optimization
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)

        # Ensure `thresholds_pr` and `cost_metric` have the same length
        thresholds_pr = np.append(thresholds_pr, 1.0)  # Extend threshold list
        cost_metric = cost_fp * (1 - precision) + cost_fn * (1 - recall)

        # Save Cost-Based Curve
        df_cost = pd.DataFrame({"threshold": thresholds_pr, "cost_metric": cost_metric})
        df_cost.to_csv(output_file.replace(".csv", "_cost.csv"), index=False)
        print(f"Cost-Based Curve saved to {output_file.replace('.csv', '_cost.csv')}")
    else:
        print("No curve selected. No file saved.")


###
