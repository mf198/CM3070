import argparse
import cudf
import pandas as pd
from ccfd.optimization.optimize_gan import optimize_gan
from ccfd.optimization.optimize_wgan import optimize_wgan
import numpy as np
from sklearn.model_selection import train_test_split
from cuml.model_selection import train_test_split as cuml_train_test_split
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset


def prepare_data(df, target_column: str = "Class", use_gpu: bool = False):
    """
    Splits the dataset into training and test sets. Converts to cuDF if GPU is enabled.

    Args:
        df (pd.DataFrame): Input dataset (always loaded in pandas).
        target_column (str): Name of the target column.
        use_gpu (bool): If True, converts df to cuDF and uses cuML's train-test split.

    Returns:
        Tuple: (df_train, df_test) as pandas or cuDF DataFrames/Series.
    """

    if use_gpu:
        print("🚀 Converting dataset to cuDF for GPU acceleration...")
        
        if isinstance(df, pd.DataFrame):
            df = cudf.from_pandas(df)

        # Check that X and y are compatible with cuml
        X = df.drop(columns=[target_column]).astype("float32")
        y = df[target_column].astype("int32")
        
        # Stratify balances the fraud records in train and test data
        return cuml_train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:        
        print("📄 Using pandas for CPU-based train-test split...")

        X = df.drop(columns=[target_column])  # Features
        y = df[target_column]  # Labels

        return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


def optimize_model(filepath: str, use_gpu: bool, model: str, trials: int, jobs: int = -1):

    print(f"\n📌 Loading dataset...")
    df = load_dataset(filepath, use_gpu)

    # Clean the dataset
    df = clean_dataset(df, use_gpu)

    # Split data before optimization
    print("\n✂️ Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = prepare_data(df, use_gpu=use_gpu)

    results = {}

    if args.model in ["gan", "both"]:
        print("\n🚀 Running GAN optimization...")
        best_gan_params = optimize_gan(X_train, y_train, use_gpu=use_gpu, n_trials=args.trials, n_jobs=jobs)
        results["GAN"] = best_gan_params
        print(f"🎯 Best GAN Parameters: {best_gan_params}")

    if args.model in ["wgan", "both"]:
        print("\n🚀 Running WGAN optimization...")
        best_wgan_params = optimize_wgan(X_train, use_gpu=use_gpu, n_trials=args.trials, n_jobs=jobs)
        results["WGAN"] = best_wgan_params
        print(f"🎯 Best WGAN Parameters: {best_wgan_params}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("ccfd/results/gan_wgan_best_params.csv", index=False)
    print("\n✅ Best hyperparameters saved to 'ccfd/results/gan_wgan_best_params.csv'.")


if __name__ == "__main__":
    """
    Runs Optuna optimization for GAN and WGAN hyperparameters and saves the results.
    """

    dataset_path = "ccfd/data/creditcard.csv"

    parser = argparse.ArgumentParser(
        description="Optimize GAN and WGAN hyperparameters using Optuna."
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
        "--model",
        choices=["gan", "wgan", "both"],
        default="both",
        help="Select which model to optimize: 'gan', 'wgan', or 'both'",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Select the number of parallel jobs (-1 = CPU count)",
    )    

    args = parser.parse_args()

    # Convert selections
    use_gpu = args.device == "gpu"
    model = args.model  # "gan", "wgan"
    trials = args.trials
    jobs = args.jobs

    optimize_model(dataset_path, True, model, trials, jobs)
