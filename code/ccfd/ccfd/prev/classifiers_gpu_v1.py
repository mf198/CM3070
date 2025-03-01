# ccfd/models/cuml_classifiers.py
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.linear_model import LogisticRegression, MBSGDClassifier
from cuml.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from typing import Dict


def train_random_forest_gpu(X_train: cudf.DataFrame, y_train: cudf.Series) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier using RAPIDS cuML.

    Args:
        X_train (cudf.DataFrame): Training features.
        y_train (cudf.Series): Training labels.

    Returns:
        RandomForestClassifier: Trained GPU-based model.
    """
    # Convert data to float32 or it generates a warning
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")

    #model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model = RandomForestClassifier(n_estimators=200, max_depth=15, bootstrap=True, n_bins=128)
    model.fit(X_train, y_train)

    return model


def train_knn_gpu(X_train: cudf.DataFrame, y_train: cudf.Series, n_neighbors: int = 5) -> KNeighborsClassifier:
    """
    Trains a k-Nearest Neighbors (kNN) classifier using RAPIDS cuML.

    Args:
        X_train (cudf.DataFrame): Training features.
        y_train (cudf.Series): Training labels.
        n_neighbors (int): Number of neighbors for kNN.

    Returns:
        KNeighborsClassifier: Trained GPU-based model.
    """
    #model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform", algorithm="brute")

    model.fit(X_train, y_train)
    return model


def train_logistic_regression_gpu(X_train: cudf.DataFrame, y_train: cudf.Series) -> LogisticRegression:
    """
    Trains a Logistic Regression model using RAPIDS cuML.

    Args:
        X_train (cudf.DataFrame): Training features.
        y_train (cudf.Series): Training labels.

    Returns:
        LogisticRegression: Trained GPU-based model.
    """
    #model = LogisticRegression()
    model = LogisticRegression(penalty="l2", C=1.0, solver="qn", max_iter=200)
    
    model.fit(X_train, y_train)
    return model


def train_mbgd_gpu(X_train: cudf.DataFrame, y_train: cudf.Series) -> MBSGDClassifier:
    """
    Trains a Mini-Batch Stochastic Gradient Descent (MBSGD) classifier using RAPIDS cuML.

    Args:
        X_train (cudf.DataFrame): Training features.
        y_train (cudf.Series): Training labels.

    Returns:
        MBSGDClassifier: Trained GPU-based model.
    """
    model = MBSGDClassifier(loss="log", eta0=0.01, batch_size=512, epochs=1000)
    #model = MBSGDClassifier(loss="log", penalty="l2", alpha=0.0001, learning_rate="adaptive")
    model.fit(X_train, y_train)
    return model


def train_xgboost_gpu(X_train, y_train) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier optimized for GPU.

    Args:
        X_train : Training features.
        y_train : Training labels.

    Returns:
        xgb.XGBClassifier: Trained GPU-based model.
    """
    # Ensure X_train and y_train are pandas before passing to XGBoost
    if isinstance(X_train, cudf.DataFrame):
        X_train = X_train.to_pandas()
    if isinstance(y_train, cudf.Series):
        y_train = y_train.to_pandas()

    #model = xgb.XGBClassifier(eval_metric="logloss", device="cuda", random_state=18)
    model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=5, random_state=42, device="cuda")
    model.fit(X_train, y_train)  # XGBoost expects pandas format
    return model


def evaluate_model_gpu(model, X_test: cudf.DataFrame, y_test: cudf.Series) -> Dict[str, float]:
    """
    Evaluates the GPU-based model on the test set.

    Args:
        model: Trained model.
        X_test (cudf.DataFrame): Test features.
        y_test: True labels (cuDF Series or NumPy array).

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, F1-score, and AUC.
    """
    
    y_pred = model.predict(X_test)  # XGBoost returns a NumPy array
    if isinstance(y_pred, np.ndarray):
        y_pred = cudf.Series(y_pred)  # Convert NumPy to cuDF

    # Convert X_test to a numpy array which is compatible with all the models
    if isinstance(y_pred, np.ndarray) == False:
        X_test = X_test.to_numpy()

    # Handle `predict_proba()`
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            y_proba = y_proba.iloc[:, 1].to_pandas()  # Convert cuDF to Pandas
        except Exception:
            print(f"Warning: {model.__class__.__name__} does not support predict_proba(). Using raw predictions.")
            y_proba = y_pred  # Default to predictions

    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            print(f"Warning: {model.__class__.__name__} does not support decision_function(). Using raw predictions.")
            y_proba = y_pred  # Default to predictions

    else:
        print(f"Warning: {model.__class__.__name__} does not support probability outputs. Using raw predictions.")
        y_proba = y_pred  # Default to predictions

    # GPU metrics
    metrics = {}
    metrics["accuracy"]  = accuracy_score(y_test, y_pred)    
    metrics["roc_auc"]   = roc_auc_score(y_test, y_proba)
        
    # Convert to pandas (sklearn functions)    
    y_pred = y_pred.to_pandas()
    y_test = y_test.to_pandas()
    
    metrics["f1_score"]  = f1_score(y_test, y_pred)
    metrics["precision"] = precision_score(y_test, y_pred)
    metrics["recall"]    = recall_score(y_test, y_pred)

    return metrics



def plot_evaluation_curves(y_test: cudf.Series, y_proba: np.ndarray):
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
