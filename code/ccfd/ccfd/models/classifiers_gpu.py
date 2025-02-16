# ccfd/models/cuml_classifiers.py
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.linear_model import LogisticRegression, MBSGDClassifier
import xgboost as xgb
import matplotlib.pyplot as plt


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

    #model = xgb.XGBClassifier(eval_metric="logloss", device="cuda", random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=5, random_state=42, device="cuda")
    model.fit(X_train, y_train)  # XGBoost expects pandas format
    return model


# def plot_evaluation_curves(y_test: cudf.Series, y_proba: np.ndarray):
#     """
#     Plots ROC, Precision-Recall, and Cost curves.

#     Args:
#         y_test (pd.Series): True labels.
#         y_proba (np.ndarray): Predicted probabilities.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     # ROC Curve
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     axes[0].plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba)))
#     axes[0].plot([0, 1], [0, 1], 'r--')
#     axes[0].set_xlabel("False Positive Rate")
#     axes[0].set_ylabel("True Positive Rate")
#     axes[0].set_title("ROC Curve")
#     axes[0].legend()

#     # Precision-Recall Curve
#     precision, recall, _ = precision_recall_curve(y_test, y_proba)
#     axes[1].plot(recall, precision, label="PR Curve")
#     axes[1].set_xlabel("Recall")
#     axes[1].set_ylabel("Precision")
#     axes[1].set_title("Precision-Recall Curve")
#     axes[1].legend()

#     # Interpolating to Ensure Same Length or it will generate an error
#     min_length = min(len(fpr), len(tpr))
#     fpr, recall = fpr[:min_length], tpr[:min_length]

#     # Cost Curve (FPR vs Recall, ensuring same length)
#     axes[2].plot(fpr, recall, label="Cost Curve")
#     axes[2].set_xlabel("False Positive Rate")
#     axes[2].set_ylabel("Recall")
#     axes[2].set_title("Cost Curve")
#     axes[2].legend()

#     plt.show()
