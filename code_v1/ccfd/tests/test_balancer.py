# tests/test_balancer.py
import pytest
import pandas as pd
from collections import Counter
from ccfd.data.balancer import apply_smote, apply_adasyn, apply_svm_smote

@pytest.fixture
def imbalanced_dataset():
    """Creates a small imbalanced dataset for testing."""
    df = pd.DataFrame({
        "feature1": list(range(1000)),
        "feature2": list(range(1000, 2000)),
        "Class": [0] * 990 + [1] * 10  # 99% class 0, 1% class 1
    })
    return df

def test_apply_smote(imbalanced_dataset):
    """Test that apply_smote balances the dataset."""
    df_balanced = apply_smote(imbalanced_dataset, target_column="Class")    
    class_counts = Counter(df_balanced["Class"])

    assert class_counts[0] == class_counts[1], "SMOTE failed to balance classes"

def test_apply_adasyn(imbalanced_dataset):
    """Test that apply_adasyn balances the dataset."""
    df_balanced = apply_adasyn(imbalanced_dataset, target_column="Class")
    
    class_counts = Counter(df_balanced["Class"])
    assert class_counts[0] == class_counts[1], "ADASYN failed to balance classes"

def test_apply_svm_smote(imbalanced_dataset):
    """Test that apply_svm_smote balances the dataset."""
    df_balanced = apply_svm_smote(imbalanced_dataset, target_column="Class")
    
    class_counts = Counter(df_balanced["Class"])
    assert class_counts[0] == class_counts[1], "SVM-SMOTE failed to balance classes"
