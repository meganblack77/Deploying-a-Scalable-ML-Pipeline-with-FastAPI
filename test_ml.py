import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def sample_data():
    data = {
        "workclass": ["Private", "Self-emp-not-inc", "Private"],
        "education": ["Bachelors", "HS-grad", "Masters"],
        "marital-status": ["Married-civ-spouse", "Divorced", "Never-married"],
        "occupation": ["Prof-specialty", "Exec-managerial", "Prof-specialty"],
        "relationship": ["Husband", "Not-in-family", "Not-in-family"],
        "race": ["White", "Black", "Asian-Pac-Islander"],
        "sex": ["Male", "Female", "Female"],
        "native-country": ["United-States", "United-States", "India"],
        "salary": [">50K", "<=50K", ">50K"]
    }
    return pd.DataFrame(data)

# Test 1: Verify that process_data returns the correct types
def test_process_data_returns_expected_types(sample_data):
    train, _ = train_test_split(sample_data, test_size=0.20, random_state=42)
    cat_features = [
        "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
    
    assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray), "X should be a DataFrame or ndarray"
    assert isinstance(y, np.ndarray), "y should be an ndarray"
    assert encoder is not None, "encoder should not be None"
    assert lb is not None, "label binarizer should not be None"

# Test 2: Verify that train_model uses LogisticRegression
def test_train_model_uses_logistic_regression(sample_data):
    train, _ = train_test_split(sample_data, test_size=0.20, random_state=42)
    cat_features = [
        "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
    ]
    X_train, y_train, _, _ = process_data(train, categorical_features=cat_features, label="salary", training=True)
    
    model = train_model(X_train, y_train)
    
    assert isinstance(model, LogisticRegression), "Model should be an instance of LogisticRegression"

# Test 3: Verify that compute_model_metrics returns a tuple of floats
def test_compute_model_metrics_returns_expected_values(sample_data):
    train, test = train_test_split(sample_data, test_size=0.20, random_state=42)
    cat_features = [
        "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
    ]
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(fbeta, float), "F1 score should be a float"
