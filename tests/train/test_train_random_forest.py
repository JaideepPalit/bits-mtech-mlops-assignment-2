import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

from src.train.train_random_forest import (
    pipeline_contruction_random_forest,
    hyperparameter_tuning_random_forest,
    best_random_forest
)

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def dummy_preprocessor():
    """Simple numeric preprocessor."""
    return StandardScaler()

@pytest.fixture
def classification_data():
    """Small dataset to keep GridSearch fast."""
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42
    )
    return X, y


# -------------------------
# Tests
# -------------------------

def test_pipeline_construction_returns_pipeline(dummy_preprocessor):
    pipeline = pipeline_contruction_random_forest(dummy_preprocessor)

    assert isinstance(pipeline, Pipeline)
    assert "preprocess" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_contains_random_forest(dummy_preprocessor):
    pipeline = pipeline_contruction_random_forest(dummy_preprocessor)

    model = pipeline.named_steps["model"]
    assert isinstance(model, RandomForestClassifier)


def test_random_forest_configuration(dummy_preprocessor):
    pipeline = pipeline_contruction_random_forest(dummy_preprocessor)
    model = pipeline.named_steps["model"]

    assert model.random_state == 42
    assert model.class_weight == "balanced"


def test_hyperparameter_tuning_returns_gridsearch(
    dummy_preprocessor, classification_data
):
    X, y = classification_data
    pipeline = pipeline_contruction_random_forest(dummy_preprocessor)

    grid = hyperparameter_tuning_random_forest(pipeline, X, y)

    assert isinstance(grid, GridSearchCV)
    assert hasattr(grid, "best_estimator_")
    assert hasattr(grid, "best_params_")


def test_random_forest_param_grid(
    dummy_preprocessor, classification_data
):
    X, y = classification_data
    pipeline = pipeline_contruction_random_forest(dummy_preprocessor)

    grid = hyperparameter_tuning_random_forest(pipeline, X, y)

    assert set(grid.param_grid["model__n_estimators"]) == {100, 200}
    assert set(grid.param_grid["model__max_depth"]) == {None, 5, 10}
    assert set(grid.param_grid["model__min_samples_split"]) == {2, 5}


def test_best_random_forest_returns_pipeline(
    dummy_preprocessor, classification_data
):
    X, y = classification_data
    pipeline = pipeline_contruction_random_forest(dummy_preprocessor)

    grid = hyperparameter_tuning_random_forest(pipeline, X, y)
    best_model = best_random_forest(grid)

    assert isinstance(best_model, Pipeline)
    assert isinstance(
        best_model.named_steps["model"], RandomForestClassifier
    )


def test_best_random_forest_is_fitted(
    dummy_preprocessor, classification_data
):
    X, y = classification_data
    pipeline = pipeline_contruction_random_forest(dummy_preprocessor)

    grid = hyperparameter_tuning_random_forest(pipeline, X, y)
    best_model = best_random_forest(grid)

    # RandomForest sets estimators_ after fitting
    assert hasattr(best_model.named_steps["model"], "estimators_")
