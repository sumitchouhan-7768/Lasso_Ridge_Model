import sys
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Now import Lasso_Ridge_reg
from Lasso_Ridge_reg import X_train, X_test, Y_train, Y_test

# Test 1: Check if dataset is loaded properly
def test_data_shapes():
    assert X_train.shape[0] > 0, "X_train should not be empty"
    assert X_test.shape[0] > 0, "X_test should not be empty"
    assert Y_train.shape[0] > 0, "Y_train should not be empty"
    assert Y_test.shape[0] > 0, "Y_test should not be empty"

# Test 2: Check if Ridge Regression runs without error
def test_ridge_regression():
    model = Ridge(alpha=1.0)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    
    assert predictions.shape == Y_test.shape, "Predictions should match Y_test shape"
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"

# Test 3: Check if Lasso Regression runs without error
def test_lasso_regression():
    model = Lasso(alpha=0.1)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == Y_test.shape, "Predictions should match Y_test shape"
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"

# Test 4: Check R² score is within a reasonable range
def test_model_performance():
    ridge_model = Ridge(alpha=1.0).fit(X_train, Y_train)
    lasso_model = Lasso(alpha=0.1).fit(X_train, Y_train)
    
    ridge_score = ridge_model.score(X_test, Y_test)
    lasso_score = lasso_model.score(X_test, Y_test)

    assert 0.5 <= ridge_score <= 1.0, "Ridge R² score should be reasonable"
    assert 0.5 <= lasso_score <= 1.0, "Lasso R² score should be reasonable"
