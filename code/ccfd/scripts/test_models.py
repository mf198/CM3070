import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from ccfd.data.dataset import load_dataset, prepare_data
from ccfd.data.preprocess import clean_dataset
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger
from cuml.model_selection import train_test_split as cuml_train_test_split
from ccfd.evaluation.evaluate_models import evaluate_model
from ccfd.utils.type_converter import to_numpy_safe
from ccfd.utils.timer import Timer
from datetime import datetime
import joblib
import gc


def test_models(params):
    """
    Loads and tests pre-trained models (GPU-accelerated cuML or CPU-based scikit-learn) with test data.

    Args:
        params (dict): Dictionary containing all command-line arguments.
    """
    timer = Timer()
    
    use_gpu = params["device"] == "gpu"

    df = load_dataset(params["dataset_path"], use_gpu)

    results = []

    if use_gpu:
        gpu_monitor = GPUTensorBoardLogger(log_dir="runs/gpu_monitor")
    model_monitor = ModelTensorBoardLogger(log_dir="runs/model_monitor")

    df = clean_dataset(df, use_gpu)

    _, X_test, _, y_test = prepare_data(df, use_gpu=use_gpu)

    # Convert to NumPy for compatibility
    y_test = to_numpy_safe(y_test)

    model_list = ["knn", "lr", "rf", "sgd", "xgboost"]
    metric_list = ["cost", "prauc", "f1", "recall", "precision"]
    ovs_list = ["smote", "adasyn", "svmsmote", "gan", "wgan"]

    # Determine selected models
    selected_models = [params["model"]] if params["model"] in model_list else model_list

    # Determine selected metrics
    selected_metrics = (
        [params["metric"]] if params["metric"] in metric_list else metric_list
    )

    # Determine selected oversampling methods
    selected_ovs = [params["ovs"]] if params["ovs"] in ovs_list else ovs_list

    step = 0
    for model_name in selected_models:
        for ovs in selected_ovs:
            for metric in selected_metrics:
                model_filename = (
                    f"{params['model_folder']}/pt_{model_name}_{ovs}_{metric}.pkl"
                )
                print(
                    f"\nLoading pre-trained {model_name} optimized for {metric} using {ovs}..."
                )

                try:
                    model = joblib.load(model_filename)
                    print(f"Loaded model from {model_filename}")
                except FileNotFoundError:
                    print(f"Model file {model_filename} not found. Skipping...")
                    continue

                timer.start()

                # Get discrete predictions
                y_pred = model.predict(X_test)

                # Get probability estimates (if supported)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)

                    # Extracts only Class 1 probabilities
                    if hasattr(y_proba, "iloc"):  # cuDF or pandas DataFrame
                        y_proba = y_proba.iloc[:, 1].to_numpy()
                    else:  # NumPy array
                        y_proba = y_proba[:, 1]
                else:
                    y_proba = y_pred  # Default to predictions if the model does not support probabilities

                y_proba = to_numpy_safe(y_proba)

                # Evaluate model with selected metric
                metrics = evaluate_model(y_test, y_pred, y_proba, params)

                elapsed_time = round(timer.elapsed_final(), 2)
                print(f"Total testing time: {elapsed_time}")

                metrics["Model"] = model_name
                metrics["Oversampling"] = ovs
                metrics["Metric"] = metric

                if use_gpu:
                    gpu_monitor.log_gpu_stats(step)

                # Log metrics dynamically
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):  # Log only numeric values
                        model_monitor.log_scalar(metric_name, value, step)

                step += 1
                results.append(metrics)

                # Explicitly delete the model to avoid memory issues
                del model
                gc.collect()

    # Construct the filename with a timestamp
    timestamp_str = datetime.now().strftime("%Y_%m_%d")
    model_name = params["model"]
    ovs = params["ovs"]
    metric = params["metric"]
    output_file = f"{params["results_folder"]}/models_test_{model_name}_{ovs}_{metric}_{timestamp_str}.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    if use_gpu:
        gpu_monitor.close()
    model_monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training on GPU or CPU")
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Choose device for training: 'gpu' (default) or 'cpu'.",
    )
    parser.add_argument(
        "--model",
        choices=["knn", "lr", "rf", "sgd", "xgboost", "all"],
        default="all",
        help="Select which model to optimize.",
    )
    parser.add_argument(
        "--ovs",
        choices=["smote", "adasyn", "svmsmote", "gan", "wgan", "all"],
        default=None,
        help="Select the oversampling method (not used for 'gan' or 'wgan' models).",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        default="ccfd/pretrained_models",
        help="Folder containing pre-trained models.",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Folder where training results will be saved.",
    )
    parser.add_argument(
        "--metric",
        choices=["prauc", "f1", "precision", "recall", "cost", "all"],
        default="prauc",
        help="Evaluation metric to optimize.",
    )
    parser.add_argument(
        "--eval_method",
        choices=["pr_curve", "cost_based", "default"],
        default="default",
        help="Evaluation method to use.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold value for evaluation.",
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

    params = {
        "dataset_path": "ccfd/data/creditcard.csv",
        "device": args.device,
        "model": args.model,
        "ovs": args.ovs,
        "model_folder": args.model_folder,
        "results_folder": args.results_folder,
        "metric": args.metric,
        "eval_method": args.eval_method,
        "threshold": args.threshold,
        "cost_fp": args.cost_fp,
        "cost_fn": args.cost_fn,
    }

    test_models(params)
