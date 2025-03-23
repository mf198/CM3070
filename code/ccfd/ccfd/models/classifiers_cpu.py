# ccfd/models/classifiers.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import xgboost as xgb
from typing import Dict


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def train_knn(
    X_train: pd.DataFrame, y_train: pd.Series, n_neighbors: int = 5
) -> KNeighborsClassifier:
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


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LogisticRegression:
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
    model = xgb.XGBClassifier(
        eval_metric="logloss", scale_pos_weight=5, random_state=42
    )
    model.fit(X_train, y_train)
    return model
