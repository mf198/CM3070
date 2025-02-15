# scripts/evaluate_oversampling.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ccfd.data.balancer import apply_smote, apply_adasyn, apply_svm_smote
from ccfd.models.classifiers_cpu import (
    train_random_forest, train_knn, train_logistic_regression,
    train_sgd, train_xgboost, evaluate_model
)
from ccfd.data.dataset import load_dataset
from ccfd.data.preprocess import clean_dataset
from ccfd.utils.timer import Timer

def prepare_data(df: pd.DataFrame, target_column: str = "Class"):
    """Splits the dataset into training and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=18)

def test_models_with_oversampling(filepath: str):
    """
    Tests all models with all oversampling methods, showing nested progress.

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
        "LogisticRegression": train_logistic_regression,
        #"RandomForest": train_random_forest,
        #"kNN": train_knn,        
        #"SGD": train_sgd,
        #"XGBoost": train_xgboost
    }

    results = []
    total_oversamplings = len(oversampling_methods)
    total_models = len(models)

    # Outer progress bar: Oversampling methods
    with tqdm(total=total_oversamplings, desc="Oversampling Progress", unit="method") as pbar_outer:
        for oversampling_name, oversampling_function in oversampling_methods.items():
            print(f"\n🔄 Applying {oversampling_name} oversampling...")
            df_balanced = oversampling_function(df)

            # Split the balanced dataset
            X_train, X_test, y_train, y_test = prepare_data(df_balanced)

            # Inner progress bar: ML Models
            with tqdm(total=total_models, desc=f"Training Progress ({oversampling_name})", unit="model") as pbar_inner:
                for model_name, model_function in models.items():
                    print(f"🚀 Training {model_name} with {oversampling_name}...")

                    timer.start()
                    model = model_function(X_train, y_train)
                    elapsed_time = timer.elapsed()
                    timer.stop()

                    metrics = evaluate_model(model, X_test, y_test)
                    metrics["Model"] = model_name
                    metrics["Oversampling"] = oversampling_name
                    metrics["Training Time (s)"] = round(elapsed_time, 2)
                    results.append(metrics)

                    pbar_inner.update(1)  # Update model training progress

            pbar_outer.update(1)  # Update oversampling progress

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv("oversampling_results.csv", index=False)
    print("\n✅ Test results saved to 'oversampling_results.csv'")
    print(results_df)

if __name__ == "__main__":
    dataset_path = "ccfd/data/creditcard.csv"  # Change to your dataset path
    test_models_with_oversampling(dataset_path)
