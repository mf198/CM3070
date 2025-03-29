import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import argparse
import cudf
import cupy as cp
import pandas as pd
import joblib
import torch
import numpy as np
from ccfd.data.dataset import load_dataset, prepare_data
from ccfd.data.preprocess import clean_dataset
from ccfd.evaluation.evaluate_models import evaluate_model
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.type_converter import to_numpy_safe
from ccfd.models.autoencoder import FraudAutoencoder
from ccfd.evaluation.threshold_analysis import compute_curve_values, find_best_threshold

from ccfd.models.autoencoder import FraudAutoencoder
from ccfd.models.vae import FraudVariationalAutoencoder


def load_model(model_path, scaler_path, device, model_type):
    """
    Loads a pre-trained Autoencoder or VAE and its scaler.

    Args:
        model_path (str): Path to model .pth file.
        scaler_path (str): Path to scaler .pkl file.
        device (str): Device to load model on ("cpu" or "cuda").
        model (str): "ae" or "vae"

    Returns:
        tuple: (model, scaler)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]

    if model_type == "ae":
        model = FraudAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    else:
        model = FraudVariationalAutoencoder(
            input_dim=input_dim, latent_dim=latent_dim
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)

    print(f"Loaded {model_type.upper()} model from {model_path}")
    print(f"Loaded Scaler from {scaler_path}")
    return model, scaler


def detect_anomalies(
    model,
    scaler,
    df,
    threshold_method,
    device,
    y_test=None,
    cost_fp=1,
    cost_fn=10,
    percentile=99,
):
    """
    Detects anomalies (fraudulent transactions) using the trained Autoencoder or VAE.

    Returns:
        DataFrame with reconstruction errors and anomaly predictions.
    """
    use_gpu = device == "cuda"

    if use_gpu and isinstance(df, pd.DataFrame):
        df = cudf.from_pandas(df)

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    X_test = (
        scaler.transform(df.to_cupy().get())
        if use_gpu
        else scaler.transform(df.to_numpy())
    )

    X_test_tensor = torch.tensor(
        cp.asarray(X_test) if use_gpu else X_test, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        # Handle VAE vs Autoencoder
        if isinstance(model, FraudVariationalAutoencoder):
            reconstructed, _, _ = model(X_test_tensor)  # VAE returns a tuple
        else:
            reconstructed = model(X_test_tensor)  # Autoencoder returns only the output

        reconstruction_errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1)

    reconstruction_errors = reconstruction_errors.detach().cpu().numpy()

    # Compute threshold
    if (
        threshold_method in ["pr_curve", "roc_curve", "cost_based"]
        and y_test is not None
    ):
        metric1, metric2, thresholds = compute_curve_values(
            y_test, reconstruction_errors, threshold_method, cost_fp, cost_fn
        )
        threshold = find_best_threshold(
            metric1,
            metric2,
            thresholds,
            curve_type=threshold_method,
            percentile=percentile,
        )
    else:
        threshold = np.percentile(reconstruction_errors, percentile)

    print(f"🔍 Best Threshold Found: {threshold:.4f}")

    # Ensure reconstruction_errors is a Series and aligns with df index
    if use_gpu:
        df = df.reset_index(drop=True)  # Reset index in cuDF to match data length
        df["Reconstruction Error"] = cudf.Series(reconstruction_errors).reset_index(
            drop=True
        )
        df["Predicted Fraud"] = (df["Reconstruction Error"] > threshold).astype("int32")
    else:
        df = df.reset_index(drop=True)  # Reset index in pandas
        df["Reconstruction Error"] = pd.Series(reconstruction_errors).reset_index(
            drop=True
        )
        df["Predicted Fraud"] = (df["Reconstruction Error"] > threshold).astype(int)

    return df


def test_autoencoder(params):
    """
    Loads and tests a pre-trained Autoencoder model using only test data.

    Args:
        params (dict): Dictionary containing all command-line arguments.
    """
    timer = Timer()

    use_gpu = params["device"] == "gpu"
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    df = load_dataset(params["dataset_path"], use_gpu=use_gpu)

    df = clean_dataset(df, use_gpu=use_gpu)

    _, X_test, _, y_test = prepare_data(df, use_gpu=use_gpu)
    
    print("\nLoading pre-trained Autoencoder and Scaler...")
    model, scaler = load_model(
        params["model_path"], params["scaler_path"], device, params["model"]
    )

    timer.start()

    print("\nDetecting anomalies...")
    results_df = detect_anomalies(
        model,
        scaler,
        X_test,
        params["threshold_method"],
        device,
        y_test=y_test,
        cost_fp=params["cost_fp"],
        cost_fn=params["cost_fn"],
        percentile=params["percentile"],
    )

    # Extract predictions and reconstruction error
    y_pred = results_df["Predicted Fraud"].values  # Binary predictions (0 or 1)
    y_proba = results_df[
        "Reconstruction Error"
    ].values  # Reconstruction error as probability

    y_test = to_numpy_safe(y_test)
    y_proba = to_numpy_safe(y_proba)

    # Evaluate the model
    metrics = evaluate_model(y_test, y_pred, y_proba, params)

    # Total execution time
    elapsed_time = round(timer.elapsed_final(), 2)
    print(f"Total testing time: {elapsed_time}")

    # Print results
    print("\nAutoencoder Performance Metrics on Test Data:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results efficiently (GPU-compatible)
    output_file = f"{params['results_folder']}/autoencoder_test_results.csv"
    if use_gpu:
        results_df.to_pandas().to_csv(output_file, index=False)
    else:
        results_df.to_csv(output_file, index=False)

    print(f"\nTest results saved to '{output_file}'.")

    # Log metrics using TensorBoard
    model_monitor = ModelTensorBoardLogger(log_dir="runs/autoencoder_test")
    for metric_name, value in metrics.items():
        model_monitor.log_scalar(metric_name, value, step=0)
    model_monitor.close()


###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained Autoencoder or Variational Autoencoder for fraud detection."
    )
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--dataset_path", type=str, default="ccfd/data/creditcard.csv")

    parser.add_argument(
        "--model",
        choices=["ae", "vae"],
        default="ae",
        help="Choose model type: Autoencoder (ae) or Variational Autoencoder (vae)",
    )

    parser.add_argument("--results_folder", type=str, default="results")
    parser.add_argument(
        "--threshold_method",
        choices=["pr_curve", "cost_based", "percentile"],
        default="pr_curve",
    )
    parser.add_argument("--percentile", type=int, default=99)
    parser.add_argument("--cost_fp", type=float, default=1.0)
    parser.add_argument("--cost_fn", type=float, default=5.0)

    # Parse initial arguments
    args = parser.parse_args()

    # Set default model and scaler paths based on selected model type
    if args.model == "vae":
        default_model_path = "ccfd/pretrained_models/pt_vae.pth"
        default_scaler_path = "ccfd/pretrained_models/pt_vae_scaler.pkl"
    else:  # Default to Autoencoder
        default_model_path = "ccfd/pretrained_models/pt_autoencoder.pth"
        default_scaler_path = "ccfd/pretrained_models/pt_autoencoder_scaler.pkl"

    # Allow user-specified paths to override defaults
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model_path,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=default_scaler_path,
        help="Path to the scaler file.",
    )

    # Parse arguments again to include model_path and scaler_path
    args = parser.parse_args()

    params = {
        "dataset_path": args.dataset_path,
        "device": args.device,
        "model_path": args.model_path,
        "scaler_path": args.scaler_path,
        "results_folder": args.results_folder,
        "threshold_method": args.threshold_method,
        "eval_method": args.threshold_method,
        "cost_fp": args.cost_fp,
        "cost_fn": args.cost_fn,
        "percentile": args.percentile,
        "model": args.model,
    }

    test_autoencoder(params)
