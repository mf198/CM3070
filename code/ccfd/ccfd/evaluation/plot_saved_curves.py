import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_saved_curves(pr_file, roc_file, cost_file):
    """
    Reads saved curve data from files and plots them.

    Args:
        pr_file (str): Path to Precision-Recall curve CSV.
        roc_file (str): Path to ROC curve CSV.
        cost_file (str): Path to Cost-Based curve CSV.

    Returns:
        None
    """
    # Load saved curve data
    df_pr = pd.read_csv(pr_file)
    df_roc = pd.read_csv(roc_file)
    df_cost = pd.read_csv(cost_file)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Precision-Recall Curve
    axes[0].plot(df_pr["recall"], df_pr["precision"], label="PR Curve", color="blue")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()
    axes[0].grid()

    # ROC Curve
    axes[1].plot(df_roc["fpr"], df_roc["tpr"], label="ROC Curve", color="green")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid()

    # Cost-Based Curve
    axes[2].plot(df_cost["threshold"], df_cost["cost_metric"], label="Cost-Based Curve", color="red")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Cost Metric")
    axes[2].set_title("Cost-Based Optimization Curve")
    axes[2].legend()
    axes[2].grid()

    plt.show()


def plot_curve_from_file(csv_file):
    """
    Reads a saved curve file and plots it.

    Args:
        csv_file (str): Path to the saved CSV file.

    Returns:
        None
    """
    # Extract Model and Oversampling Method from Filename
    file_name = os.path.basename(csv_file)
    parts = file_name.replace("curves_", "").replace(".csv", "").split("_")
    
    if len(parts) < 2:
        print(f"Invalid file name format: {file_name}")
        return
    
    model_name = parts[0]  # Example: "LogisticRegression"
    oversampling_method = parts[1]  # Example: "ADASYN"

    # Detect Curve Type
    if "pr" in parts:
        curve_type = "pr_curve"
    elif "roc" in parts:
        curve_type = "roc_curve"
    elif "cost" in parts:
        curve_type = "cost_based"
    else:
        print(f"Could not determine curve type from filename: {file_name}")
        return

    # Read CSV File
    df = pd.read_csv(csv_file)

    # Plot the Appropriate Curve
    plt.figure(figsize=(7, 5))

    if curve_type == "pr_curve":
        plt.plot(df["recall"], df["precision"], label="PR Curve", color="blue")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({model_name} + {oversampling_method})")

    elif curve_type == "roc_curve":
        plt.plot(df["fpr"], df["tpr"], label="ROC Curve", color="green")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({model_name} + {oversampling_method})")

    elif curve_type == "cost_based":
        plt.plot(df["threshold"], df["cost_metric"], label="Cost-Based Curve", color="red")
        plt.xlabel("Threshold")
        plt.ylabel("Cost Metric")
        plt.title(f"Cost-Based Curve ({model_name} + {oversampling_method})")

    else:
        print(f"Unrecognized curve type in file: {file_name}")
        return

    plt.legend()
    plt.grid()
    plt.show()
