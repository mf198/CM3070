import argparse
import cudf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from cuml.model_selection import train_test_split as cuml_train_test_split

# Import GAN/WGAN optimizers
from ccfd.optimization.optimize_gan import optimize_gan
from ccfd.optimization.optimize_wgan import optimize_wgan

# Import ML model optimizers
from ccfd.optimization.optimize_knn import optimize_knn
from ccfd.optimization.optimize_logistic_regression import optimize_logistic_regression
from ccfd.optimization.optimize_random_forest import optimize_random_forest
from ccfd.optimization.optimize_sgd import optimize_sgd
from ccfd.optimization.optimize_xgboost import optimize_xgboost

# Import dataset and preprocessing functions
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


def optimize_model(
    filepath: str, use_gpu: bool, model: str, trials: int, jobs: int = -1
):
    """
    Runs Optuna optimization for GAN, WGAN, and ML models.

    Args:
        filepath (str): Path to the dataset.
        use_gpu (bool): Whether to use GPU.
        model (str): Model to optimize ("gan", "wgan", "knn", etc.).
        trials (int): Number of optimization trials.
        jobs (int): Number of parallel jobs.
    """

    print(f"\n📌 Loading dataset...")
    df = load_dataset(filepath, use_gpu)

    # Clean the dataset
    df = clean_dataset(df, use_gpu)

    # Split data before optimization
    print("\n✂️ Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = prepare_data(df, use_gpu=use_gpu)

    results = {}

    ### ✅ **GAN / WGAN Optimization**
    if model in ["gan", "both"]:
        print("\n🚀 Running GAN optimization...")
        best_gan_params = optimize_gan(
            X_train, y_train, use_gpu=use_gpu, n_trials=trials, n_jobs=jobs
        )
        results["GAN"] = best_gan_params
        print(f"🎯 Best GAN Parameters: {best_gan_params}")

    if model in ["wgan", "both"]:
        print("\n🚀 Running WGAN optimization...")
        best_wgan_params = optimize_wgan(
            X_train, y_train, use_gpu=use_gpu, n_trials=trials, n_jobs=jobs
        )
        results["WGAN"] = best_wgan_params
        print(f"🎯 Best WGAN Parameters: {best_wgan_params}")

    ### ✅ **Machine Learning Models Optimization**
    if model in ["knn", "all"]:
        print("\n🚀 Running KNN optimization...")
        best_knn_params = optimize_knn(
            X_train, y_train, n_trials=trials, use_gpu=use_gpu
        )
        results["KNN"] = best_knn_params
        print(f"🎯 Best KNN Parameters: {best_knn_params}")

    if model in ["logistic_regression", "all"]:
        print("\n🚀 Running Logistic Regression optimization...")
        best_lr_params = optimize_logistic_regression(
            X_train, y_train, n_trials=trials, use_gpu=use_gpu
        )
        results["LogisticRegression"] = best_lr_params
        print(f"🎯 Best Logistic Regression Parameters: {best_lr_params}")

    if model in ["random_forest", "all"]:
        print("\n🚀 Running Random Forest optimization...")
        best_rf_params = optimize_random_forest(
            X_train, y_train, n_trials=trials, use_gpu=use_gpu
        )
        results["RandomForest"] = best_rf_params
        print(f"🎯 Best Random Forest Parameters: {best_rf_params}")

    if model in ["sgd", "all"]:
        print("\n🚀 Running SGD optimization...")
        best_sgd_params = optimize_sgd(
            X_train, y_train, n_trials=trials, use_gpu=use_gpu
        )
        results["SGD"] = best_sgd_params
        print(f"🎯 Best SGD Parameters: {best_sgd_params}")

    if model in ["xgboost", "all"]:
        print("\n🚀 Running XGBoost optimization...")
        best_xgb_params = optimize_xgboost(
            X_train, y_train, n_trials=trials, use_gpu=use_gpu
        )
        results["XGBoost"] = best_xgb_params
        print(f"🎯 Best XGBoost Parameters: {best_xgb_params}")

    # ✅ **Save results to CSV**
    results_df = pd.DataFrame(results)
    results_df.to_csv("ccfd/results/best_model_params.csv", index=False)
    print("\n✅ Best hyperparameters saved to 'ccfd/results/best_model_params.csv'.")


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
        "--model",
        choices=[
            "gan",
            "wgan",
            "knn",
            "logistic_regression",
            "random_forest",
            "sgd",
            "xgboost",
            "both",
            "all",
        ],
        default="all",
        help="Select which model to optimize.",
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
    model = args.model  # "gan", "wgan", "knn", "logistic_regression", etc.
    trials = args.trials
    jobs = args.jobs

    optimize_model(dataset_path, use_gpu, model, trials, jobs)
