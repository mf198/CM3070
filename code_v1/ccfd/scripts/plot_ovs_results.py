# scripts/plot_ovs_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ccfd.data.dataset import load_dataset



def plot_by_oversampling(df):
    """
    Plots multiple metrics grouped by Oversampling Method.
    Displays 3 charts per row.
    """    

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    # Determine the number of rows needed (ceil division)
    num_metrics = len(metrics)
    num_cols = 3  # 3 charts per row
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Compute number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 5), sharey=True)
    axes = axes.flatten()  # Convert to 1D array for easy iteration

    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="Oversampling", y=metric, hue="Model", ax=axes[i], palette="viridis")
        axes[i].set_title(f"{metric.capitalize()} by Oversampling Method")
        axes[i].set_xlabel("Oversampling Method")
        axes[i].tick_params(axis='x', rotation=45)        
        axes[i].legend(loc="lower right", bbox_to_anchor=(1, 0), title="ML Model")

    # Hide empty subplots if metrics < total grid slots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove unused subplot spaces

    plt.tight_layout()
    plt.show()

def plot_by_model(df):
    """
    Plots multiple metrics grouped by ML model.
    """
    # Define metrics and number of subplots
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    num_metrics = len(metrics)
    num_cols = 3  # 3 charts per row
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Compute number of rows

    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 5), sharey=True)
    axes = axes.flatten()  # Convert to 1D array for easy iteration

    # Loop through each metric and create a subplot
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="Model", y=metric, hue="Oversampling", ax=axes[i], palette="coolwarm")
        axes[i].set_title(f"{metric.capitalize()} by ML Model")
        axes[i].set_xlabel("ML Model")
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
        axes[i].legend(loc="lower right", bbox_to_anchor=(1, 0), title="OVS Method")
                        
    # Hide empty subplots if metrics < total grid slots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove unused subplot spaces

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Example usage:
    df = load_dataset("cuml_oversampling_results.csv")  # Load dataset
    if df is not None:
        plot_by_oversampling(df)  # Generate plots
        plot_by_model(df)
        

