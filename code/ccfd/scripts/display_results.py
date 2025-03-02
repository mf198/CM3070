import argparse
import cudf
import pandas as pd

def display_results_by_model_and_oversampling(filepath="oversampling_results_gpu.csv", use_gpu=True):
    """
    Reads the results from the CSV file and displays a table grouping by ML model and oversampling methods using cuDF.

    Args:
        filepath (str): Path to the CSV file.
    """
    try:
        if use_gpu:
            # Load the CSV file using cuDF for GPU acceleration
            df = cudf.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)

        # Sort the results by Model and Oversampling Method
        sorted_df = df.sort_values(by=["Oversampling", "Model"])

        # Reset index to remove any extra index column
        sorted_df = sorted_df.reset_index(drop=True)        

        # Display the results as a cuDF DataFrame
        print("\n📊 Results Grouped by Model and Oversampling Method:")
        print(sorted_df)

    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Please check the file path.")


def display_results_by_model_ordered(filepath="oversampling_results_gpu.csv", use_gpu=True):
    """
    Reads the ssults from the CSV file and displays a table grouping by ML model,
    ordering results by recall, precision, f1-score, and roc-auc using cuDF.

    Args:
        filepath (str): Path to the CSV file.
    """
    try:
        if use_gpu:
            # Load the CSV file using cuDF for GPU acceleration
            df = cudf.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)

        # Ensure the required metrics exist in the dataset
        required_metrics = ["recall", "precision", "f1_score", "roc_auc"]
        available_metrics = [metric for metric in required_metrics if metric in df.columns]

        if not available_metrics:
            print("❌ Error: None of the required metrics (recall, precision, f1_score, roc_auc) are found in the dataset.")
            return

        # Sort results by recall, precision, f1-score, and roc-auc (in descending order)
        sorted_df = df.sort_values(by=available_metrics, ascending=False)

        # Reset index to remove any extra index column
        sorted_df = sorted_df.reset_index(drop=True)

        # Display the results as a cuDF DataFrame
        print("\n📊 Results Grouped by Model (Ordered by Recall, Precision, F1-score, ROC-AUC):")
        print(sorted_df)

    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Please check the file path.")


if __name__ == "__main__":

    # Allow user to choose between GPU(default) or CPU training    
    parser = argparse.ArgumentParser(description="Run model training on GPU or CPU")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="Choose device for model training: gpu (default) or cpu")
    args = parser.parse_args()
    
    use_gpu = args.device == "gpu"

    if (use_gpu):
        display_results_by_model_and_oversampling("ovs_models_results_gpu.csv", use_gpu=True)    
        display_results_by_model_ordered("ovs_models_results_gpu.csv")
    else:
        display_results_by_model_and_oversampling("ovs_models_results_cpu.csv", use_gpu=False)
        display_results_by_model_ordered("ovs_models_results_cpu.csv")
###