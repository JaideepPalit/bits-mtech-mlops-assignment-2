import importlib
import os
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
import pytest

import evaluate.evaluate as mod


@pytest.fixture(autouse=True)
def stub_show(monkeypatch):
    """Prevent plt.show() from blocking / opening windows during tests."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def _set_module_file_to_tmp(tmp_path, monkeypatch):
    """
    Make module.__file__ point into tmp_path/x/y/cnn_eval.py so that
    Path(__file__).resolve().parents[2] resolves into tmp_path (or inside it).
    """
    fake_path = tmp_path / "x" / "y" / "cnn_eval.py"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    # create an empty file to make resolve() happy
    fake_path.write_text("# fake module file for tests\n")
    monkeypatch.setattr(mod, "__file__", str(fake_path.resolve()))
    return fake_path


def test_evaluate_cnn_basic():
    # Create a fake test dataset: two batches
    y_batch1 = np.array([0, 1])
    y_batch2 = np.array([1])
    test_ds = [
        (np.zeros((2, 224, 224, 3)), y_batch1),
        (np.zeros((1, 224, 224, 3)), y_batch2),
    ]

    # Fake model that returns probability for each sample (as (n,1) array)
    probs = np.array([[0.2], [0.8], [0.6]])
    fake_model = SimpleNamespace(predict=lambda ds: probs)

    y_true, y_probs, y_pred = mod.evaluate_cnn(fake_model, test_ds)

    # Assertions
    assert np.array_equal(y_true, np.array([0, 1, 1]))
    # y_probs should match the fake model's output
    assert np.allclose(y_probs, probs)
    # predictions: >0.5 -> 1
    assert np.array_equal(y_pred.flatten(), np.array([0, 1, 1]))


def test_plot_cnn_confusion_matrix_creates_file(tmp_path, monkeypatch):
    # Make module.__file__ point into tmp so saved output goes to tmp_path/output/evaluate
    _set_module_file_to_tmp(tmp_path, monkeypatch)

    # simple labels
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    # call function (plt.show is stubbed)
    mod.plot_cnn_confusion_matrix(y_true, y_pred)

    out_dir = Path(mod.__file__).resolve().parents[2] / "output" / "evaluate"
    out_file = out_dir / "cnn_confusion_matrix.png"
    assert out_file.exists(), f"Expected {out_file} to be created"
    assert out_file.stat().st_size > 0


def test_plot_cnn_roc_curve_returns_auc_and_creates_file(tmp_path, monkeypatch):
    _set_module_file_to_tmp(tmp_path, monkeypatch)

    # y_true and y_probs valid for roc_curve
    y_true = np.array([0, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.9, 0.8, 0.2, 0.6])

    auc_val = mod.plot_cnn_roc_curve(y_true, y_probs)

    # Basic checks on returned AUC
    assert isinstance(auc_val, float)
    assert 0.0 <= auc_val <= 1.0

    out_dir = Path(mod.__file__).resolve().parents[2] / "output" / "evaluate"
    out_file = out_dir / "cnn_roc.png"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_get_prediction_with_img_array_uses_model_predict():
    # fake model that returns a probability > 0.5
    fake_model = SimpleNamespace(predict=lambda arr: np.array([[0.72]]))

    # provide a fake image array shape (1,224,224,3)
    img_array = np.ones((1, 224, 224, 3), dtype=np.float32)
    label, confidence, raw_prob = mod.get_prediction(fake_model, img_path="unused", img_array=img_array)

    assert label == "Dog"
    # confidence should equal raw_prob for predicted positive
    assert pytest.approx(confidence, rel=1e-6) == raw_prob
    assert pytest.approx(raw_prob, rel=1e-6) == 0.72