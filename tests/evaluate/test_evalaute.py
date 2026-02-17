import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluate.evaluate import (
    evaluate_model,
    summarize_cv,
    cross_validate_performance,
    bar_plot_test_metric,
    roc_plot,
    cv_score,
    confusion_matrix
)

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42
    )
    return X, y


@pytest.fixture
def trained_models(classification_data):
    X, y = classification_data

    logreg = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=20, random_state=42)

    logreg.fit(X, y)
    rf.fit(X, y)

    return logreg, rf, X, y


# -------------------------
# evaluate_model
# -------------------------

def test_evaluate_model_returns_all_metrics(trained_models):
    logreg, _, X, y = trained_models

    metrics = evaluate_model(logreg, X, y)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {
        "Accuracy", "Precision", "Recall", "ROC-AUC"
    }

    for value in metrics.values():
        assert 0.0 <= value <= 1.0


# -------------------------
# summarize_cv
# -------------------------

def test_summarize_cv_filters_test_metrics():
    cv_results = {
        "test_accuracy": [0.8, 0.9],
        "test_precision": [0.7, 0.8],
        "fit_time": [0.1, 0.2]
    }

    summary = summarize_cv(cv_results)

    assert "test_accuracy" in summary
    assert "test_precision" in summary
    assert "fit_time" not in summary
    assert summary["test_accuracy"] == pytest.approx(0.85)


# -------------------------
# cross_validate_performance
# -------------------------

def test_cross_validate_performance_outputs(classification_data):
    X, y = classification_data
    model = LogisticRegression(max_iter=300)

    cv_raw, cv_summary = cross_validate_performance(model, X, y)

    assert isinstance(cv_raw, dict)
    assert isinstance(cv_summary, dict)
    assert any("test_" in k for k in cv_summary.keys())


# -------------------------
# Plotting functions
# -------------------------

@patch("src.evaluate.evaluate.plt.show")
def test_bar_plot_test_metric(mock_show):
    logreg_res = {
        "Accuracy": 0.85,
        "Precision": 0.80,
        "Recall": 0.75,
        "ROC-AUC": 0.88
    }
    rf_res = {
        "Accuracy": 0.90,
        "Precision": 0.85,
        "Recall": 0.82,
        "ROC-AUC": 0.92
    }

    bar_plot_test_metric(logreg_res, rf_res)

    mock_show.assert_called_once()


@patch("src.evaluate.evaluate.plt.show")
def test_roc_plot(mock_show, trained_models):
    logreg, rf, X, y = trained_models

    roc_plot(logreg, rf, X, y)

    mock_show.assert_called_once()


@patch("src.evaluate.evaluate.plt.show")
def test_cv_score(mock_show):
    cv_logreg = {
        "test_accuracy": [0.8, 0.85],
        "test_precision": [0.75, 0.8],
        "test_recall": [0.7, 0.78],
        "test_roc_auc": [0.82, 0.88]
    }

    cv_rf = {
        "test_accuracy": [0.85, 0.9],
        "test_precision": [0.8, 0.85],
        "test_recall": [0.78, 0.82],
        "test_roc_auc": [0.87, 0.9]
    }

    cv_score(cv_logreg, cv_rf)

    mock_show.assert_called_once()


@patch("src.evaluate.evaluate.plt.show")
@patch("src.evaluate.evaluate.ConfusionMatrixDisplay.from_estimator")
def test_confusion_matrix(mock_cmd, mock_show, trained_models):
    _, rf, X, y = trained_models

    confusion_matrix(rf, X, y)

    mock_cmd.assert_called_once()
    mock_show.assert_called_once()
