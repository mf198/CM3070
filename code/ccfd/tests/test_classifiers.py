# tests/test_classifiers.py

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from ccfd.models.classifiers_cpu import (
    train_random_forest,
    train_knn,
    train_logistic_regression,
    train_sgd,
    train_xgboost,
)


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.9, 0.1],
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_train_random_forest(sample_data):
    X, y = sample_data
    model = train_random_forest(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert hasattr(model, "predict_proba")


def test_train_knn(sample_data):
    X, y = sample_data
    model = train_knn(X, y, n_neighbors=3)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert hasattr(model, "predict_proba")


def test_train_logistic_regression(sample_data):
    X, y = sample_data
    model = train_logistic_regression(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert hasattr(model, "predict_proba")


def test_train_sgd(sample_data):
    X, y = sample_data
    model = train_sgd(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_xgboost(sample_data):
    X, y = sample_data
    model = train_xgboost(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert hasattr(model, "predict_proba")
