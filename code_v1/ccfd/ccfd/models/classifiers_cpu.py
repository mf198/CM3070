# ccfd/models/classifiers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import xgboost as xgb
from typing import Dict
from sklearn.metrics import confusion_matrix

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    #model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_knn(X_train: pd.DataFrame, y_train: pd.Series, n_neighbors: int = 5) -> KNeighborsClassifier:
    """
    Trains a k-Nearest Neighbors (kNN) classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_neighbors (int): Number of neighbors for kNN.

    Returns:
        KNeighborsClassifier: Trained model.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Trains a Logistic Regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(solver="saga", max_iter=2000)
    model.fit(X_train, y_train)
    return model

def train_sgd(X_train: pd.DataFrame, y_train: pd.Series) -> SGDClassifier:
    """
    Trains a Stochastic Gradient Descent (SGD) classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        SGDClassifier: Trained model.
    """
    model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        xgb.XGBClassifier: Trained XGBoost model.
    """    
    model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def plot_evaluation_curves(y_test: pd.Series, y_proba: np.ndarray):
    """
    Plots ROC, Precision-Recall, and Cost curves.

    Args:
        y_test (pd.Series): True labels.
        y_proba (np.ndarray): Predicted probabilities.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba)))
    axes[0].plot([0, 1], [0, 1], 'r--')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(recall, precision, label="PR Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    # Interpolating to Ensure Same Length or it will generate an error
    min_length = min(len(fpr), len(tpr))
    fpr, recall = fpr[:min_length], tpr[:min_length]

    # Cost Curve (FPR vs Recall, ensuring same length)
    axes[2].plot(fpr, recall, label="Cost Curve")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("Recall")
    axes[2].set_title("Cost Curve")
    axes[2].legend()

    plt.show()


def evaluate_model_cpu(model, X_test: pd.DataFrame, y_test: pd.Series, use_gpu=False) -> Dict[str, float]:
    """
    Evaluates the model on the test set using GPU (cuDF/cuML) or CPU (pandas/scikit-learn).

    Args:
        model: Trained model.
        X_test (cudf.DataFrame or pd.DataFrame): Test features.
        y_test (cudf.Series, np.ndarray, or pd.Series): True labels.
        use_gpu (bool): If True, use cuDF/cuML for evaluation, otherwise use pandas/scikit-learn.

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, F1-score, and AUC.
    """
    # Predict labels
    y_pred = model.predict(X_test)

    print(f"Pred: {y_pred}")

    # Convert y_pred based on type
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)  # Convert NumPy to Pandas

    # Convert y_test to Pandas if needed
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)  # Convert NumPy to Pandas

    # Handle `predict_proba()` for probability-based metrics
    y_proba = None   
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            if isinstance(y_proba, np.ndarray):
                y_proba = pd.Series(y_proba[:, 1])  # Convert NumPy to Pandas
            else:
                y_proba = y_pred  # Default to predictions if empty
        except Exception:
            print(f"Warning: {model.__class__.__name__} does not support predict_proba(). Using raw predictions.")
            y_proba = y_pred  # Default to predictions

    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
            if isinstance(y_proba, np.ndarray):
                y_proba = pd.Series(y_proba)  # Convert NumPy to Pandas
            else:
                y_proba = y_proba.to_pandas()
        except Exception:
            print(f"Warning: {model.__class__.__name__} does not support decision_function(). Using raw predictions.")
            y_proba = y_pred  # Default to predictions

    else:
        print(f"Warning: {model.__class__.__name__} does not support probability outputs. Using raw predictions.")
        y_proba = y_pred  # Default to predictions

    # Compute evaluation metrics (scikit-learn for CPU)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }