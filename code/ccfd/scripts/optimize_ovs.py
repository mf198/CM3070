import argparse
import cudf
import pandas as pd
from sklearn.model_selection import train_test_split
from cuml.model_selection import train_test_split as cuml_train_test_split

# Import GAN/WGAN optimizers
from ccfd.optimization.optimize_gan import optimize_gan
from ccfd.optimization.optimize_wgan import optimize_wgan

# Import dataset and preprocessing functions
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset
from ccfd.data.dataset import prepare_data


def optimize_ovs(train_params: dict):
    """
    Runs Optuna optimization for GAN and WGAN.

    Args:
        train_params (dict): Dictionary containing all experiment parameters, including:
            - dataset (str): Path to the dataset.
            - device (str): "gpu" or "cpu".
            - ovs (str): Oversampling technique to optimize ("gan", "wgan", "knn", etc.).
            - trials (int): Number of optimization trials.
            - jobs (int): Number of parallel jobs.
            - oversampling (str, optional): Oversampling method ("smote", "adasyn", etc.).
            - output_folder (str): Folder where results will be saved.
    """

    # Extract parameters from dictionary
    dataset_path = train_params["dataset"]
    use_gpu = train_params["device"] == "gpu"
    ovs = train_params["ovs"]
    output_folder = train_params["output_folder"]

    # Load dataset
    df = load_dataset(dataset_path, use_gpu)

    # Clean the dataset
    df = clean_dataset(df, use_gpu)

    # Split data before optimization
    X_train, X_test, y_train, y_test = prepare_data(df, use_gpu=use_gpu)

    results = {}

    ### **GAN / WGAN Optimization**
    if ovs in ["gan", "all"]:
        print("\nRunning GAN optimization...")
        best_gan_params = optimize_gan(X_train, y_train, train_params)
        results["GAN"] = best_gan_params
        print(f"Best GAN Parameters: {best_gan_params}")

    if ovs in ["wgan", "all"]:
        print("\nRunning WGAN optimization...")
        best_wgan_params = optimize_wgan(X_train, y_train, train_params)
        results["WGAN"] = best_wgan_params
        print(f"Best WGAN Parameters: {best_wgan_params}")

    # **Save results to CSV**
    results_df = pd.DataFrame(results)
    results_filepath = f"{output_folder}/pt_ovs_params.csv"
    results_df.to_csv(results_filepath, index=False)

    print(f"\nBest hyperparameters saved to '{results_filepath}'.")


###

if __name__ == "__main__":
    """
    Runs Optuna optimization for GAN, WGAN, and ML models, then saves the results.
    """

    dataset_path = "ccfd/data/creditcard.csv"

    parser = argparse.ArgumentParser(
        description="Optimize GAN, WGAN, and ML model hyperparameters using Optuna."
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
        "--ovs",
        choices=[
            "gan",
            "wgan",
            "all",
        ],
        default="all",
        help="Select which oversampling method to optimize.",
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
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Folder where training results will be saved.",
    )

    args = parser.parse_args()

    # Store experiment parameters in a dictionary
    params = {
        "dataset": dataset_path,
        "device": args.device,
        "ovs": args.ovs,
        "trials": args.trials,
        "jobs": args.jobs,
        "output_folder": args.output_folder,
        "results_folder": args.results_folder,
        "model": None
    }

    # Print experiment setup
    print(f"Training ovs method with parameters: {params}")

    # Run optimization
    optimize_ovs(params)
