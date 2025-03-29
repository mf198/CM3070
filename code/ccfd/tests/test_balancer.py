import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from ccfd.data.balancer import apply_smote, apply_adasyn, apply_svm_smote

# Optional: only import cudf and cupy if available
try:
    import cudf
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False

@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.9, 0.1],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=5,
        n_clusters_per_class=1,
        n_samples=1000,
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)

@pytest.mark.parametrize("func", [apply_smote, apply_adasyn, apply_svm_smote])
def test_oversampling_cpu(func, sample_data):
    X, y = sample_data
    X_resampled, y_resampled = func(X, y, use_gpu=False)

    assert isinstance(X_resampled, pd.DataFrame)
    assert isinstance(y_resampled, pd.Series)
    assert len(X_resampled) == len(y_resampled)
    assert y_resampled.value_counts().min() == y_resampled.value_counts().max()

@pytest.mark.skipif(not gpu_available, reason="cuDF not available")
@pytest.mark.parametrize("func", [apply_smote, apply_adasyn, apply_svm_smote])
def test_oversampling_gpu(func, sample_data):
    X, y = sample_data
    X_cudf = cudf.DataFrame.from_pandas(X)
    y_cudf = cudf.Series(y.values)

    X_resampled, y_resampled = func(X_cudf, y_cudf, use_gpu=True)

    assert isinstance(X_resampled, cudf.DataFrame)
    assert isinstance(y_resampled, cudf.Series)
    assert len(X_resampled) == len(y_resampled)
    assert y_resampled.value_counts().min() == y_resampled.value_counts().max()
