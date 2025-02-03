# scripts/plot_oversampling_results.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(filepath: str):
    """Plots model performance across different oversampling techniques."""
    df = pd.read_csv(filepath)
    
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        for model in df["Model"].unique():
            subset = df[df["Model"] == model]
            plt.plot(subset["Oversampling"], subset[metric], marker="o", label=model)
        
        plt.xlabel("Oversampling Method")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Comparison Across Models")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    plot_results("oversampling_results.csv")
