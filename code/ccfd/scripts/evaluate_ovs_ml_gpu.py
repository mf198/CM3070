# scripts/evaluate_ovs_ml_gpu.py
import pandas as pd
import numpy as np
from cuml.model_selection import train_test_split
from ccfd.data.balancer import ( apply_smote, apply_adasyn, apply_svm_smote, 
                                apply_gan_oversampling, apply_wgan_oversampling
)
from ccfd.models.classifiers_gpu import (
    train_cuml_random_forest, train_cuml_knn, train_cuml_logistic_regression,
    train_cuml_mbgd, train_cuml_xgboost, evaluate_cuml_model
)
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset
from ccfd.utils.timer import Timer
from ccfd.utils.tensorboard_model_logger import ModelTensorBoardLogger
from ccfd.utils.tensorboard_gpu_logger import GPUTensorBoardLogger


def prepare_data(df: pd.DataFrame, target_column: str = "Class"):
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
        "SVM-SMOTE": apply_svm_smote,
        "ADASYN": apply_adasyn,        
        "GAN": apply_gan_oversampling,
        "WGAN": apply_wgan_oversampling,        
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

    # Initialize GPU Monitor    
    # Initialize TensorBoard loggers
    model_monitor =  ModelTensorBoardLogger(log_dir="runs/model_monitor")
    gpu_monitor = GPUTensorBoardLogger(log_dir="runs/gpu_monitor")

    df = clean_dataset(df)

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"X_train: {X_train}")
    print(f"y_train: {y_train}")

    # Loops into oversampling methods    
    for oversampling_name, oversampling_function in oversampling_methods.items():
        print(f"\n===============================================")
        print(f"🔄 Applying {oversampling_name} oversampling...")
        print(f"===============================================")
        df = X_train.copy()
        df["Class"] = y_train
        df_train_balanced = oversampling_function(df)

        # Loops into ML models
        step = 0
        for model_name, model_function in models.items():
            print(f"\n🚀 Training {model_name} with {oversampling_name}...")
            
            timer.start()

            # Extract balanced features and labels
            X_train_balanced = df_train_balanced.drop(columns=["Class"])
            y_train_balanced = df_train_balanced["Class"]
            
            # Call ML function
            model = model_function(X_train_balanced, y_train_balanced)

            metrics = evaluate_cuml_model(model, X_test, y_test)
            metrics["Model"] = model_name
            metrics["Oversampling"] = oversampling_name
            metrics["Training Time (s)"] = round(timer.elapsed_final(), 2)

            # Log GPU usage
            gpu_monitor.log_gpu_stats(step)            

            # Log model 
            model_monitor.log_scalar("Accuracy",    metrics["accuracy"], step)
            model_monitor.log_scalar("Precision",   metrics["precision"], step)
            model_monitor.log_scalar("Recall",      metrics["recall"], step)
            model_monitor.log_scalar("Recall",      metrics["recall"], step)
            model_monitor.log_scalar("F1-score",    metrics["f1_score"], step)
            model_monitor.log_scalar("AUC",         metrics["roc_auc"], step)            

            step += 1

            results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("cuml_oversampling_results.csv", index=False)

    # Close GPU monitor
    gpu_monitor.close()   
    model_monitor.close()

if __name__ == "__main__":
    dataset_path = "ccfd/data/creditcard.csv"  # Change to your dataset path
    test_models_with_oversampling(dataset_path)

