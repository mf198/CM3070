import os
import joblib
import optuna
import pandas as pd
from datetime import datetime   


def save_time_performance(train_params, best_value, elapsed_time):
    """
    Saves training performance details to a CSV file with a timestamped filename.

    Args:
        train_params (dict): Dictionary containing training parameters.
        best_value (float): Best value for the selected model, oversampling method and metric
        elapsed_time (float): Training time in seconds.
    """

    # Generate timestamp for the filename (YYYY_MM_DD)
    timestamp_str = datetime.now().strftime("%Y_%m_%d")

    # Construct the filename with a timestamp
    output_file = f"training_time_performance_{timestamp_str}.csv"
    training_path = os.path.join(train_params["results_folder"], output_file)

    # Prepare training data
    training_data = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "device": [train_params["device"]],
        "model": [train_params["model"]],
        "oversampling": [train_params["ovs"] if train_params["ovs"] else "no_ovs"],
        "metric": [train_params["metric"]],
        "value": [best_value],
        "elapsed_time_sec": [elapsed_time],  # Pass elapsed_time instead of a string
    }

    df_results = pd.DataFrame(training_data)

    # If the CSV file exists, append; otherwise, write a new file with headers
    if os.path.exists(training_path):
        df_results.to_csv(training_path, mode="a", header=False, index=False)
    else:
        df_results.to_csv(training_path, mode="w", header=True, index=False)    