import argparse
import cudf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from ccfd.data.dataset import load_dataset, prepare_data
from ccfd.data.preprocess import clean_dataset
from ccfd.utils.timer import Timer
from ccfd.utils.gpu_monitor import track_gpu_during_training
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger
from cuml.model_selection import train_test_split as cuml_train_test_split
from ccfd.evaluation.evaluate_models import evaluate_model
from ccfd.utils.type_converter import to_numpy_safe
from ccfd.utils.timer import Timer
from datetime import datetime
import joblib
import gc


def aaaprepare_data(df, target_column: str = "Class", use_gpu: bool = False):
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
        df = cudf.DataFrame(df)

        # Use cuML's GPU-based train-test split
        return cuml_train_test_split(df, test_size=0.3, random_state=42)
    else:
        print("Using pandas for CPU-based train-test split...")
        return train_test_split(df, test_size=0.2, random_state=42)


###


def test_models(params):
    """
    Loads and tests pre-trained models (GPU-accelerated cuML or CPU-based scikit-learn) with test data.

    Args:
        params (dict): Dictionary containing all command-line arguments.
    """
    timer = Timer()

    print(f"\nLoading dataset...")
    df = load_dataset(params["dataset_path"])

    results = []

    use_gpu = params["device"] == "gpu"

    if use_gpu:
        gpu_monitor = GPUTensorBoardLogger(log_dir="runs/gpu_monitor")
    model_monitor = ModelTensorBoardLogger(log_dir="runs/model_monitor")

    df = clean_dataset(df)
    
    _, X_test, _, y_test = prepare_data(df, use_gpu=use_gpu)

    # Convert to NumPy for compatibility
    X_test = to_numpy_safe(X_test)
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

                # Get discrete predictions
                y_pred = model.predict(X_test)

                # Get probability estimates (if supported)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = y_pred  # Default to predictions if the model does not support probabilities

                y_proba = to_numpy_safe(y_proba)

                # Evaluate model with selected metric
                metrics = evaluate_model(y_test, y_pred, y_proba, params)

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
        default=0.5,
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
