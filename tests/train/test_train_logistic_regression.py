import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

from src.train.train_logistic_regression import (
    pipeline_contruction_logistic_regression,
    hyperparameter_tuning_logistic_regression,
    best_logistic_regression
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def dummy_preprocessor():
    """Simple numeric preprocessor."""
    return StandardScaler()

@pytest.fixture
def classification_data():
    """Small dataset to keep tests fast."""
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    return X, y


# -----------------------------
# Tests
# -----------------------------

def test_pipeline_construction_returns_pipeline(dummy_preprocessor):
    pipeline = pipeline_contruction_logistic_regression(dummy_preprocessor)

    assert isinstance(pipeline, Pipeline)
    assert "preprocess" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)


def test_pipeline_model_configuration(dummy_preprocessor):
    pipeline = pipeline_contruction_logistic_regression(dummy_preprocessor)
    model = pipeline.named_steps["model"]

    assert model.max_iter == 1000
    assert model.class_weight == "balanced"


def test_hyperparameter_tuning_returns_gridsearch(
    dummy_preprocessor, classification_data
):
    X, y = classification_data
    pipeline = pipeline_contruction_logistic_regression(dummy_preprocessor)

    grid = hyperparameter_tuning_logistic_regression(pipeline, X, y)

    assert isinstance(grid, GridSearchCV)
    assert hasattr(grid, "best_estimator_")
    assert hasattr(grid, "best_params_")


def test_gridsearch_param_grid(dummy_preprocessor, classification_data):
    X, y = classification_data
    pipeline = pipeline_contruction_logistic_regression(dummy_preprocessor)

    grid = hyperparameter_tuning_logistic_regression(pipeline, X, y)

    expected_C_values = {0.01, 0.1, 1, 10}
    assert set(grid.param_grid["model__C"]) == expected_C_values
    assert grid.param_grid["model__penalty"] == ["l2"]


def test_best_logistic_regression_returns_pipeline(
    dummy_preprocessor, classification_data
):
    X, y = classification_data
    pipeline = pipeline_contruction_logistic_regression(dummy_preprocessor)

    grid = hyperparameter_tuning_logistic_regression(pipeline, X, y)
    best_model = best_logistic_regression(grid)

    assert isinstance(best_model, Pipeline)
    assert isinstance(best_model.named_steps["model"], LogisticRegression)


def test_best_logistic_regression_is_fitted(
    dummy_preprocessor, classification_data
):
    X, y = classification_data
    pipeline = pipeline_contruction_logistic_regression(dummy_preprocessor)

    grid = hyperparameter_tuning_logistic_regression(pipeline, X, y)
    best_model = best_logistic_regression(grid)

    # LogisticRegression sets coef_ after fitting
    assert hasattr(best_model.named_steps["model"], "coef_")
