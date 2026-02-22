import types
from types import SimpleNamespace
from pathlib import Path
import pytest

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

import src.train.train_cnn as mod
import tensorflow as tf


def test_build_cnn_returns_keras_model():
    """build_cnn should return a tf.keras.Model with a single sigmoid output."""
    model = mod.build_cnn()
    assert isinstance(model, tf.keras.Model)
    # Last layer activation should be sigmoid
    last_layer = model.layers[-1]
    # activation is a function, check its name
    assert getattr(last_layer, "activation", None) is not None
    assert last_layer.activation.__name__ == "sigmoid"


def test_plot_results_draws_and_shows(monkeypatch):
    """
    plot_results should call plotting functions without raising.
    We monkeypatch plt.show to avoid opening windows.
    """
    # Create a fake Keras History-like object
    fake_history = SimpleNamespace(
        history={
            "accuracy": [0.1, 0.3, 0.6],
            "val_accuracy": [0.15, 0.35, 0.55],
            "loss": [1.0, 0.8, 0.6],
            "val_loss": [0.95, 0.85, 0.7],
        }
    )

    # prevent showing UI
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    # Should run without exceptions
    mod.plot_results(fake_history)