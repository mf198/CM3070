import argparse
import cudf
import pandas as pd

def display_results(filepath, use_gpu=True, sort_column=None, ascending=False):
    """
    Reads the results from the CSV file and displays a table.
    Optionally sorts the results by a specified column.

    Args:
        filepath (str): Path to the CSV file.
        use_gpu (bool): Whether to use GPU acceleration (cuDF) or CPU (pandas).
        sort_column (str, optional): Column to sort by.
        ascending (bool): Whether to sort in ascending order. Defaults to False.
    """
    try:
        if use_gpu:
            df = cudf.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)

        if sort_column:
            if sort_column in df.columns:
                df = df.sort_values(by=sort_column, ascending=ascending)
            else:
                print(f"Warning: Column '{sort_column}' not found. Displaying unsorted results.")

        df = df.reset_index(drop=True)

        # Display results
        print("\nResults:")
        print(df)

    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please check the file path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display model evaluation results from CSV.")

    # Command-line arguments
    parser.add_argument("--file", required=True, help="Path to the results CSV file.")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="Choose device for loading: gpu (default) or cpu.")
    parser.add_argument("--sort", choices=["accuracy", "roc_auc", "pr_auc", "f1_score", "precision", "recall", "Model", "Oversampling", "Metric"], help="Column to sort by.")
    parser.add_argument("--ascending", action="store_true", help="Sort in ascending order (default: descending).")

    args = parser.parse_args()

    use_gpu = args.device == "gpu"

    display_results(filepath=args.file, use_gpu=use_gpu, sort_column=args.sort, ascending=args.ascending)
