import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import argparse
import pandas as pd

# Import optimization functions
from ccfd.optimization.optimize_ae import optimize_autoencoder
from ccfd.optimization.optimize_vae import optimize_vae

# Import dataset and preprocessing functions
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset


from ccfd.data.dataset import load_dataset, prepare_data
from ccfd.data.preprocess import clean_dataset
from ccfd.optimization.optimize_ae import optimize_autoencoder
from ccfd.optimization.optimize_vae import optimize_vae


def optimize_model_pipeline(train_params: dict):
    """
    Runs Optuna optimization for Autoencoder or VAE based on `model`.

    Args:
        train_params (dict): Dictionary containing all experiment parameters.
    """
    dataset_path = train_params["dataset"]
    use_gpu = train_params["device"] == "gpu"
    output_folder = train_params["output_folder"]
    model = train_params.get("model", "autoencoder").lower()

    # Load and clean dataset
    df = load_dataset(dataset_path, use_gpu=use_gpu)
    df = clean_dataset(df, use_gpu=use_gpu)

    # Split into train and test (stratified)    
    X_train, X_test, y_train, y_test = prepare_data(
        df, target_column="Class", use_gpu=use_gpu
    )

    # Filter out frauds for training (only normal data)    
    normal_train_mask = (y_train == 0).values if not use_gpu else (y_train == 0)
    X_train = X_train[normal_train_mask]

    # Run optimization
    if model == "vae":
        print("\n Running Variational Autoencoder (VAE) optimization...")
        best_params = optimize_vae(X_train, train_params)
        filename_prefix = "pt_vae"
    else:
        print("\n Running Autoencoder optimization...")
        best_params = optimize_autoencoder(X_train, train_params)
        filename_prefix = "pt_autoencoder"

    print(f" Best {model.upper()} Parameters: {best_params}")

    # Save best hyperparameters
    results_df = pd.DataFrame([best_params])
    results_filepath = f"{output_folder}/{filename_prefix}_params.csv"
    results_df.to_csv(results_filepath, index=False)

    print(f"\n Best {model.upper()} hyperparameters saved to '{results_filepath}'.")
###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize Autoencoder or VAE hyperparameters using Optuna."
    )

    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--output_folder", type=str, default="ccfd/pretrained_models")
    parser.add_argument("--results_folder", type=str, default="results")
    parser.add_argument("--model", choices=["ae", "vae"], default="ae")
    parser.add_argument("--dataset", type=str, default="ccfd/data/creditcard.csv")

    args = parser.parse_args()

    params = {
        "dataset": args.dataset,
        "device": args.device,
        "trials": args.trials,
        "jobs": args.jobs,
        "output_folder": args.output_folder,
        "results_folder": args.results_folder,
        "model": args.model,
    }

    print(f"🔍 Optimizing {params['model']} with parameters: {params}")
    optimize_model_pipeline(params)
