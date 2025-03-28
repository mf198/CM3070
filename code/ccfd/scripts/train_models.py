import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from cuml.model_selection import train_test_split as cuml_train_test_split

# Import ML model optimizers
from ccfd.optimization.optimize_knn import optimize_knn
from ccfd.optimization.optimize_logistic_regression import optimize_logistic_regression
from ccfd.optimization.optimize_random_forest import optimize_random_forest
from ccfd.optimization.optimize_sgd import optimize_sgd
from ccfd.optimization.optimize_xgboost import optimize_xgboost
from ccfd.data.balancer import (
    apply_smote,
    apply_adasyn,
    apply_svm_smote,
    apply_gan_oversampling,
    apply_wgan_oversampling,
)

# Import dataset and preprocessing functions
from ccfd.data.dataset import load_dataset, prepare_data
from ccfd.data.preprocess import clean_dataset


def optimize_model(train_params: dict):
    """
    Runs Optuna optimization for GAN, WGAN, and ML models using a parameter dictionary.

    Args:
        train_params (dict): Dictionary containing all experiment parameters, including:
            - dataset (str): Path to the dataset.
            - device (str): "gpu" or "cpu".
            - model (str): Model to optimize (""knn", "lr", "rf" etc.).
            - trials (int): Number of optimization trials.
            - jobs (int): Number of parallel jobs.
            - ovs (str, optional): Oversampling method ("smote", "adasyn", etc.).
            - metric (str, optional): Evaluation metric ("prauc", "cost", etc.).
            - output_folder (str): Folder where results will be saved.model_name = train_params["model"]
    """

    # Define the mapping of oversampling methods to functions
    oversampling_methods = {
        "smote": apply_smote,
        "adasyn": apply_adasyn,
        "svmsmote": apply_svm_smote,
        "gan": apply_gan_oversampling,
        "wgan": apply_wgan_oversampling,
    }

    # Extract parameters from dictionary
    dataset_path = train_params["dataset"]
    use_gpu = train_params["device"] == "gpu"
    model = train_params["model"]
    oversampling_method = train_params.get("ovs", None)  # Might be None
    output_folder = train_params["output_folder"]

    # Store the selected oversampling function in train_params
    train_params["oversampling_function"] = oversampling_methods.get(
        oversampling_method, None
    )

    # Load dataset
    df = load_dataset(dataset_path, use_gpu)

    # Clean the dataset
    df = clean_dataset(df, use_gpu)

    # Split data before optimization    
    X_train, X_test, y_train, y_test = prepare_data(df, use_gpu=use_gpu)

    results = {}

    ### **Machine Learning Models Optimization**
    if model in ["knn", "all"]:
        print("\nRunning KNN optimization...")
        best_knn_params = optimize_knn(X_train, y_train, train_params)
        results["KNN"] = best_knn_params
        print(f"Best KNN Parameters: {best_knn_params}")

    if model in ["lr", "all"]:
        print("\nRunning Logistic Regression optimization...")
        best_lr_params = optimize_logistic_regression(X_train, y_train, train_params)
        results["LogisticRegression"] = best_lr_params
        print(f"Best Logistic Regression Parameters: {best_lr_params}")

    if model in ["rf", "all"]:
        print("\nRunning Random Forest optimization...")
        best_rf_params = optimize_random_forest(X_train, y_train, train_params)
        results["RandomForest"] = best_rf_params
        print(f"Best Random Forest Parameters: {best_rf_params}")

    if model in ["sgd", "all"]:
        print("\nRunning SGD optimization...")
        best_sgd_params = optimize_sgd(X_train, y_train, train_params)
        results["SGD"] = best_sgd_params
        print(f"Best SGD Parameters: {best_sgd_params}")

    if model in ["xgboost", "all"]:
        print("\nRunning XGBoost optimization...")
        best_xgb_params = optimize_xgboost(X_train, y_train, train_params)
        results["XGBoost"] = best_xgb_params
        print(f"Best XGBoost Parameters: {best_xgb_params}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_filepath = f"{output_folder}/pt_model_params.csv"
    results_df.to_csv(results_filepath, index=False)

    print(f"\nBest hyperparameters saved to '{results_filepath}'.")


###

if __name__ == "__main__":
    """
    Runs Optuna optimization for GAN, WGAN, and ML models, then saves the results.
    """

    dataset_path = "ccfd/data/creditcard.csv"

    parser = argparse.ArgumentParser(
        description="Train GAN, WGAN, and ML model with hyperparameters optimization using Optuna."
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Choose device for training: 'gpu' (default) or 'cpu'.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of optimization trials (default: 30).",
    )
    parser.add_argument(
        "--model",
        choices=[
            "knn",
            "lr",
            "rf",
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
        help="Select the number of parallel jobs (-1 = CPU count).",
    )
    parser.add_argument(
        "--ovs",
        choices=["smote", "adasyn", "svmsmote", "gan", "wgan"],
        default=None,
        help="Select the oversampling method (not used for 'gan' or 'wgan' models).",
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
    parser.add_argument(
        "--metric",
        choices=["prauc", "f1", "precision", "recall", "cost"],
        default="prauc",
        help="Evaluation metric to optimize. Options: 'prauc', 'f1', 'precision', 'recall', 'cost'.",
    )
    parser.add_argument(
        "--cost_fp",
        type=float,
        default=1.0,
        help="Cost of a false positive (only used if metric='cost').",
    )
    parser.add_argument(
        "--cost_fn",
        type=float,
        default=5.0,
        help="Cost of a false negative (only used if metric='cost').",
    )

    args = parser.parse_args()

    # Ensure cost parameters are only used when metric is 'cost'
    if args.metric != "cost":
        args.cost_fp = None
        args.cost_fn = None

    # Convert selections
    model = args.model  # "knn", "lr", etc.
    trials = args.trials
    jobs = args.jobs
    ovs = args.ovs if model not in ["gan", "wgan"] else None  # Only for ML models
    output_folder = args.output_folder
    results_folder = args.results_folder
    metric = args.metric
    cost_fp = args.cost_fp
    cost_fn = args.cost_fn

    # Store experiment parameters in a dictionary
    params = {
        "dataset": dataset_path,
        "device": args.device,
        "model": model,
        "trials": trials,
        "jobs": jobs,
        "ovs": ovs,
        "metric": metric,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "output_folder": output_folder,
        "results_folder": results_folder,
    }

    # Print experiment setup
    print(f"Training model(s) with parameters: {params}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save parameters to a JSON file for tracking
    config_path = os.path.join(output_folder, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(params, f, indent=4)

    print(f"Experiment configuration saved at: {config_path}")

    # Run optimization
    optimize_model(params)
