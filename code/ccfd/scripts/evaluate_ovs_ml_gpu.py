# scripts/evaluate_ovs_ml_gpu.py
import cudf
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from cuml.model_selection import train_test_split
from ccfd.data.balancer import apply_smote, apply_adasyn, apply_svm_smote
from ccfd.models.cuml_classifiers import (
    train_cuml_random_forest, train_cuml_knn, train_cuml_logistic_regression,
    train_cuml_mbgd, train_cuml_xgboost, evaluate_cuml_model
)
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset
from ccfd.utils.timer import Timer
from ccfd.utils.gpu_monitor import track_gpu_during_training


def prepare_data(df: cudf.DataFrame, target_column: str = "Class"):
    """Splits the dataset into GPU-based training and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_models_with_oversampling(filepath: str):
    """
    Tests all GPU-accelerated models with all oversampling methods.

    Args:
        filepath (str): Path to the dataset.
    """
    # Create the Timer object
    timer = Timer()

    print("\n📌 Loading dataset...")
    df = load_dataset(filepath)

    oversampling_methods = {
        "SMOTE": apply_smote,
        "ADASYN": apply_adasyn,
        "SVM-SMOTE": apply_svm_smote
    }

    models = {
        "LogisticRegression": train_cuml_logistic_regression,
        "RandomForest": train_cuml_random_forest,
        "kNN": train_cuml_knn,        
        "SGD": train_cuml_mbgd,
        "XGBoost": train_cuml_xgboost
    }

    results = []
    total_oversamplings = len(oversampling_methods)
    total_models = len(models)

    # Loops into oversampling methods    
    for oversampling_name, oversampling_function in oversampling_methods.items():
        print(f"\n===============================================")
        print(f"🔄 Applying {oversampling_name} oversampling...")
        print(f"===============================================")
        df_balanced = oversampling_function(df)

        X_train, X_test, y_train, y_test = prepare_data(df_balanced)

        # Loops into ML models
        for model_name, model_function in models.items():
            print(f"\n🚀 Training {model_name} with {oversampling_name}...")

            timer.start()
            model = model_function(X_train, y_train)
            elapsed_time = timer.elapsed_final()            

            print(f"⏱️ Elapsed time: {elapsed_time:.2f} seconds")
            
            model, elapsed_time = track_gpu_during_training(model_function, X_train, y_train)
            
            metrics = evaluate_cuml_model(model, X_test, y_test)
            metrics["Model"] = model_name
            metrics["Oversampling"] = oversampling_name
            metrics["Training Time (s)"] = round(elapsed_time, 2)
            results.append(metrics)        

    results_df = pd.DataFrame(results)
    results_df.to_csv("cuml_oversampling_results.csv", index=False)


if __name__ == "__main__":
    dataset_path = "ccfd/data/creditcard.csv"  # Change to your dataset path
    test_models_with_oversampling(dataset_path)