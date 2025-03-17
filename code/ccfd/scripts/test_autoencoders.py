import argparse
import cudf
import pandas as pd
import joblib
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.models.autoencoder import FraudAutoencoder  # Ensure this is updated for VAE support

def load_autoencoder(model_path, device):
    """
    Loads a pre-trained Autoencoder model.

    Args:
        model_path (str): Path to the saved Autoencoder model.
        device (str): 'cpu' or 'cuda'.

    Returns:
        Trained Autoencoder model.
    """
    model = joblib.load(model_path).to(device)
    model.eval()
    return model

def detect_anomalies(model, scaler, df, threshold, device):
    """
    Detects anomalies (fraudulent transactions) using the trained Autoencoder.

    Args:
        model (torch.nn.Module): Trained Autoencoder model.
        scaler (MinMaxScaler): Pre-trained MinMaxScaler.
        df (DataFrame): Input dataset (pandas or cuDF).
        threshold (float): Reconstruction error threshold.
        device (str): 'cpu' or 'cuda'.

    Returns:
        DataFrame with reconstruction errors and anomaly predictions.
    """
    df = df.to_pandas() if isinstance(df, cudf.DataFrame) else df
    X_test = scaler.transform(df.drop(columns=["Class"]).values)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        reconstruction_errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

    df["Reconstruction Error"] = reconstruction_errors
    df["Predicted Fraud"] = (df["Reconstruction Error"] > threshold).astype(int)

    return df

def test_autoencoder(params):
    """
    Loads and tests a pre-trained Autoencoder model with test data.

    Args:
        params (dict): Dictionary containing all command-line arguments.
    """
    timer = Timer()
    device = "cuda" if params["device"] == "gpu" and torch.cuda.is_available() else "cpu"

    print(f"\n📌 Loading dataset...")
    df = load_dataset(params["dataset_path"])

    df = clean_dataset(df)

    print("\n🚀 Loading pre-trained Autoencoder...")
    model = load_autoencoder(params["model_path"], device)

    print("\n📊 Loading pre-trained MinMaxScaler...")
    scaler = joblib.load(params["scaler_path"])

    print("\n🔍 Detecting anomalies...")
    results_df = detect_anomalies(model, scaler, df, params["threshold"], device)

    # Save results
    output_file = f"{params['results_folder']}/autoencoder_test_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to '{output_file}'.")

    # Log results
    model_monitor = ModelTensorBoardLogger(log_dir="runs/autoencoder_test")
    model_monitor.log_scalar("Avg Reconstruction Error", results_df["Reconstruction Error"].mean(), step=0)
    model_monitor.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Autoencoder for fraud detection.")
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Choose device for testing: 'gpu' (default) or 'cpu'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ccfd/pretrained_models/pt_autoencoder.pkl",
        help="Path to the trained Autoencoder model.",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="ccfd/pretrained_models/scaler.pkl",
        help="Path to the pre-trained MinMaxScaler.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="ccfd/data/creditcard.csv",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Folder where test results will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Threshold for anomaly detection (reconstruction error).",
    )

    args = parser.parse_args()

    params = {
        "dataset_path": args.dataset_path,
        "device": args.device,
        "model_path": args.model_path,
        "scaler_path": args.scaler_path,
        "results_folder": args.results_folder,
        "threshold": args.threshold,
    }

    test_autoencoder(params)
