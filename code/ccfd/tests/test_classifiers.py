# tests/test_classifiers.py
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ccfd.models.classifiers import (
    train_random_forest, train_knn, train_logistic_regression, train_sgd, train_xgboost, evaluate_model
)

@pytest.fixture
def sample_data():
    """Creates a small dataset for testing models."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))  # Imbalanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@pytest.mark.parametrize("train_function", [train_random_forest, train_knn, train_logistic_regression, train_sgd])
def test_train_model(sample_data, train_function):
    """Tests if models train without errors."""
    X_train, _, y_train, _ = sample_data
    model = train_function(X_train, y_train)
    assert model is not None, f"{train_function.__name__} failed to train a model"

def test_model_predictions(sample_data):
    """Tests that models produce predictions of the correct shape."""
    X_train, X_test, y_train, _ = sample_data

    model = train_random_forest(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(X_test), "Model did not return predictions of correct shape"

def test_model_evaluation(sample_data):
    """Tests that model evaluation returns valid metrics."""
    X_train, X_test, y_train, y_test = sample_data

    model = train_random_forest(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert all(0.0 <= v <= 1.0 for v in metrics.values()), "Metrics should be between 0 and 1"

def test_evaluation_curves(sample_data):
    """Tests if ROC, PR, and Cost curves can be plotted without errors."""
    X_train, X_test, y_train, y_test = sample_data

    model = train_xgboost(X_train, y_train)
    evaluate_model(model, X_test, y_test, plot_curves=True)