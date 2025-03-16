import argparse
import cudf
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from cuml.model_selection import train_test_split as cuml_train_test_split

# Import GAN/WGAN optimizers
from ccfd.optimization.optimize_gan import optimize_gan
from ccfd.optimization.optimize_wgan import optimize_wgan

# Import dataset and preprocessing functions
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset
from ccfd.data.dataset import prepare_data


def xxxprepare_data(df, target_column: str = "Class", use_gpu: bool = False):
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

        # Check that X and y are compatible with cuML
        X = df.drop(columns=[target_column]).astype("float32")
        y = df[target_column].astype("int32")

        # Stratify balances the fraud records in train and test data
        return cuml_train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        print("📄 Using pandas for CPU-based train-test split...")

        X = df.drop(columns=[target_column])  # Features
        y = df[target_column]  # Labels

        return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


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
    print("\n✂️ Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = prepare_data(df, use_gpu=use_gpu)

    results = {}

    ### **GAN / WGAN Optimization**
    if ovs in ["gan", "all"]:
        print("\n🚀 Running GAN optimization...")
        best_gan_params = optimize_gan(X_train, y_train, train_params)
        results["GAN"] = best_gan_params
        print(f"🎯 Best GAN Parameters: {best_gan_params}")

    if ovs in ["wgan", "all"]:
        print("\n🚀 Running WGAN optimization...")
        best_wgan_params = optimize_wgan(X_train, y_train, train_params)
        results["WGAN"] = best_wgan_params
        print(f"🎯 Best WGAN Parameters: {best_wgan_params}")

    # **Save results to CSV**
    results_df = pd.DataFrame(results)
    results_filepath = f"{output_folder}/pt_ovs_params.csv"
    results_df.to_csv(results_filepath, index=False)

    print(f"\n✅ Best hyperparameters saved to '{results_filepath}'.")


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

    args = parser.parse_args()

    # Convert selections
    ovs = args.ovs  # "gan", "wgan", "all"
    trials = args.trials
    jobs = args.jobs
    output_folder = args.output_folder

    # Store experiment parameters in a dictionary
    params = {
        "dataset": dataset_path,
        "device": args.device,
        "ovs": ovs,
        "trials": trials,
        "jobs": jobs,
        "output_folder": output_folder,
    }

    # Print experiment setup
    print(f"🔍 Training ovs method with parameters: {params}")

    # Run optimization
    optimize_ovs(params)
