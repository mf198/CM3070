import cudf

def display_results_by_model_and_oversampling(filepath="cuml_oversampling_results.csv"):
    """
    Reads the results from the CSV file and displays a table grouping by ML model and oversampling methods using cuDF.

    Args:
        filepath (str): Path to the CSV file.
    """
    try:
        # Load the CSV file using cuDF for GPU acceleration
        df = cudf.read_csv(filepath)

        # ✅ Sort the results by Model and Oversampling Method
        sorted_df = df.sort_values(by=["Oversampling", "Model"])

        # Display the results as a cuDF DataFrame
        print("\n📊 Results Grouped by Model and Oversampling Method (cuDF):")
        print(sorted_df)

    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Please check the file path.")


def display_results_by_model_ordered(filepath="cuml_oversampling_results.csv"):
    """
    Reads the results from the CSV file and displays a table grouping by ML model,
    ordering results by recall, precision, f1-score, and roc-auc using cuDF.

    Args:
        filepath (str): Path to the CSV file.
    """
    try:
        # Load the CSV file using cuDF for GPU acceleration
        df = cudf.read_csv(filepath)

        # Ensure the required metrics exist in the dataset
        required_metrics = ["recall", "precision", "f1_score", "roc_auc"]
        available_metrics = [metric for metric in required_metrics if metric in df.columns]

        if not available_metrics:
            print("❌ Error: None of the required metrics (recall, precision, f1_score, roc_auc) are found in the dataset.")
            return

        # ✅ Sort results by recall, precision, f1-score, and roc-auc (in descending order)
        sorted_df = df.sort_values(by=available_metrics, ascending=False)

        # Display the results as a cuDF DataFrame
        print("\n📊 Results Grouped by Model (Ordered by Recall, Precision, F1-score, ROC-AUC):")
        print(sorted_df.to_pandas())  # Convert to pandas for better printing in console

    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Please check the file path.")


if __name__ == "__main__":
    #display_results_by_oversampling_cudf()
    #display_results_by_algorithm_cudf()
    display_results_by_model_and_oversampling()    
    display_results_by_model_ordered()


