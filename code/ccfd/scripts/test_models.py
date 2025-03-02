import argparse
import cudf
import pandas as pd
import cupy as cp
import numpy as np
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.svm import SVC as cuSVC
from cuml.linear_model import LogisticRegression as cuLogReg
from cuml.neighbors import KNeighborsClassifier as cuKNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from ccfd.data.balancer import (
    apply_smote,
    apply_adasyn,
    apply_svm_smote,
    apply_gan_oversampling,
    apply_wgan_oversampling,
)
from ccfd.models.classifiers_gpu import (
    train_random_forest_gpu,
    train_knn_gpu,
    train_logistic_regression_gpu,
    train_mbgd_gpu,
    train_xgboost_gpu,
)
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset
from ccfd.utils.timer import Timer
from ccfd.utils.gpu_monitor import track_gpu_during_training
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger
from cuml.model_selection import train_test_split as cuml_train_test_split
from ccfd.data.balancer import apply_smote, apply_adasyn, apply_svm_smote
from ccfd.models.classifiers_cpu import (
    train_random_forest,
    train_knn,
    train_logistic_regression,
    train_sgd,
    train_xgboost,
)
from ccfd.evaluation.evaluate_models import evaluate_model_cpu, evaluate_model_gpu


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
        df = cudf.DataFrame(df)

        # Use cuML's GPU-based train-test split
        return cuml_train_test_split(df, test_size=0.3, random_state=42)
    else:
        print("📄 Using pandas for CPU-based train-test split...")
        return train_test_split(df, test_size=0.2, random_state=42)


###


def test_models_with_oversampling(filepath: str, use_gpu: bool, threshold_method: str, cost_fp: int, cost_fn: int):
    """
    Tests all models (GPU-accelerated cuML or CPU-based scikit-learn) with all oversampling methods.

    Args:
        filepath (str): Path to the dataset.
        use_gpu (bool): If True, runs models using cuML (GPU), otherwise uses scikit-learn (CPU).
    """
    timer = Timer()

    # The dataset is always loaded with pandas because
    # cuML does not have many of the oversampling methods used
    print(f"\n📌 Loading dataset...")
    df = load_dataset(filepath)

    oversampling_methods = {
        "GAN": apply_gan_oversampling,
        "WGAN": apply_wgan_oversampling,
        "SMOTE": apply_smote,
        # "SVM-SMOTE": apply_svm_smote,
        "ADASYN": apply_adasyn,
    }

    # Choose models based on GPU/CPU selection
    if use_gpu:
        models = {
            "LogisticRegression": train_logistic_regression_gpu,
            "RandomForest": train_random_forest_gpu,
            "kNN": train_knn_gpu,
            # "SGD": train_mbgd_gpu,
            "XGBoost": train_xgboost_gpu,
        }
        evaluate_model = evaluate_model_gpu
    else:
        models = {
            "LogisticRegression": train_logistic_regression,
            # "RandomForest": train_random_forest,
            # "kNN": train_knn,
            # "SGD": train_sgd,
            # "XGBoost": train_xgboost
        }
        evaluate_model = evaluate_model_cpu

    results = []

    if use_gpu:
        # Initialize TensorBoard gpu logger
        gpu_monitor = GPUTensorBoardLogger(log_dir="runs/gpu_monitor")

    # Initialize TensorBoard model logger
    model_monitor = ModelTensorBoardLogger(log_dir="runs/model_monitor")

    # Clean the dataset
    df = clean_dataset(df)

    # Split data before applying oversampling (to prevent data leakage)
    print("\n✂️ Splitting dataset into train and test sets BEFORE oversampling...")
    df_train, df_test = prepare_data(df, use_gpu=False)
    X_test = df_test.drop(columns=["Class"])
    y_test = df_test["Class"]

    # Convert to cuDF if using GPU
    if use_gpu == True:
        X_test = cudf.DataFrame(X_test)
        y_test = cudf.Series(y_test)

    # Loop through oversampling methods
    for oversampling_name, oversampling_function in oversampling_methods.items():
        # Start the timer to calculate the execution time
        timer.start()

        print(f"\n===============================================")
        print(f"🔄 Applying {oversampling_name} oversampling...")
        print(f"===============================================")
        df_train_balanced = oversampling__function(df_train, use_gpu=use_gpu)

        # Oversampling method execution time
        ovs_time = round(timer.elapsed_final(), 2)
        print(f"Oversampling time: {ovs_time}")

        # Extract balanced features and labels
        X_train_balanced = df_train_balanced.drop(columns=["Class"])
        y_train_balanced = df_train_balanced["Class"]

        step = 0
        for model_name, model_function in models.items():
            print(
                f"\n🚀 Training {model_name} with {oversampling_name} on {'GPU' if use_gpu else 'CPU'}..."
            )

            timer.start()

            # Train model
            model = model_function(X_train_balanced, y_train_balanced)

            # Create the output filename based on the model used
            filename = f"ccfd/results/curves_{model_name}_{oversampling_name}.csv"

            # Evaluate model
            metrics = evaluate_model(
                model, X_test, y_test, threshold_method=threshold_method, cost_fp=cost_fp, cost_fn=cost_fn, save_curve=True, output_file=filename
            )
            metrics["Model"] = model_name
            metrics["Oversampling"] = oversampling_name
            metrics["Oversampling time"] = ovs_time
            metrics["Training Time (s)"] = round(timer.elapsed_final(), 2)

            if use_gpu:
                # Log GPU usage
                gpu_monitor.log_gpu_stats(step)

            # Log model metrics
            model_monitor.log_scalar("Accuracy", metrics["accuracy"], step)
            model_monitor.log_scalar("Precision", metrics["precision"], step)
            model_monitor.log_scalar("Recall", metrics["recall"], step)
            model_monitor.log_scalar("F1-score", metrics["f1_score"], step)
            model_monitor.log_scalar("AUC", metrics["roc_auc"], step)

            step += 1
            results.append(metrics)

    # Save results
    results_df = pd.DataFrame(results)
    if use_gpu:
        results_df.to_csv("ovs_models_results_gpu.csv", index=False)
    else:
        results_df.to_csv("ovs_models_results_cpu.csv", index=False)

    # Close loggers
    if use_gpu:
        gpu_monitor.close()
    model_monitor.close()


if __name__ == "__main__":

    dataset_path = "ccfd/data/creditcard.csv"

    # Allow user to choose between GPU(default) or CPU training
    parser = argparse.ArgumentParser(description="Run model training on GPU or CPU")
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Choose device for model training: gpu (default) or cpu",
    )

    parser.add_argument(
        "--threshold",
        choices=["default", "pr_curve", "cost_based"],
        default="default",
        help="Choose threshold selection method: 'default' (0.5), 'pr_curve' (Precision-Recall Curve), or 'cost_based'",
    )

    # Cost Parameters (for cost-based thresholding)
    parser.add_argument(
        "--cost_fp",
        type=int,
        choices=range(1, 11),
        default=1,
        help="Cost of a false positive (1-10, default: 1)",
    )

    parser.add_argument(
        "--cost_fn",
        type=int,
        choices=range(1, 11),
        default=10,
        help="Cost of a false negative (1-10, default: 10)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert selections
    use_gpu = args.device == "gpu"
    threshold_method = args.threshold  # "default", "pr_curve", or "cost_based"
    cost_fp = args.cost_fp
    cost_fn = args.cost_fn

    test_models_with_oversampling(dataset_path, use_gpu, threshold_method, cost_fp, cost_fn)
