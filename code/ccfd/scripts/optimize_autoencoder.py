import argparse
import pandas as pd

# Import Autoencoder optimization function
from ccfd.optimization.optimize_ae import optimize_autoencoder

# Import dataset and preprocessing functions
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset

def optimize_autoencoder_pipeline(train_params: dict):
    """
    Runs Optuna optimization for the Autoencoder.

    Args:
        train_params (dict): Dictionary containing all experiment parameters, including:
            - dataset (str): Path to the dataset.
            - device (str): "gpu" or "cpu".
            - trials (int): Number of optimization trials.
            - jobs (int): Number of parallel jobs.
            - output_folder (str): Folder where results will be saved.
    """

    # Extract parameters
    dataset_path = train_params["dataset"]
    use_gpu = train_params["device"] == "gpu"
    output_folder = train_params["output_folder"]

    # Load dataset
    df = load_dataset(dataset_path, use_gpu)

    # Clean the dataset
    df = clean_dataset(df, use_gpu)

    # Remove target column (Unsupervised learning)
    X_train = df.drop(columns=["Class"])

    print("\n🚀 Running Autoencoder optimization...")
    best_autoencoder_params = optimize_autoencoder(X_train, train_params)

    print(f"🎯 Best Autoencoder Parameters: {best_autoencoder_params}")

    # **Save results to CSV**
    results_df = pd.DataFrame([best_autoencoder_params])
    results_filepath = f"{output_folder}/pt_autoencoder_params.csv"
    results_df.to_csv(results_filepath, index=False)

    print(f"\n✅ Best Autoencoder hyperparameters saved to '{results_filepath}'.")


###

if __name__ == "__main__":
    """
    Runs Optuna optimization for the Autoencoder and saves the results.
    """

    dataset_path = "ccfd/data/creditcard.csv"

    parser = argparse.ArgumentParser(
        description="Optimize Autoencoder hyperparameters using Optuna."
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Choose device for training: 'gpu' (default) or 'cpu'",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of optimization trials (default: 30)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Select the number of parallel jobs (-1 = CPU count)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="ccfd/pretrained_models",
        help="Folder where optimization results will be saved.",
    )

    args = parser.parse_args()

    # Store experiment parameters in a dictionary
    params = {
        "dataset": dataset_path,
        "device": args.device,
        "trials": args.trials,
        "jobs": args.jobs,
        "output_folder": args.output_folder,
    }

    # Print experiment setup
    print(f"🔍 Optimizing Autoencoder with parameters: {params}")

    # Run optimization
    optimize_autoencoder_pipeline(params)
