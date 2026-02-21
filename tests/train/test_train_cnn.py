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


def test_train_cnn_uses_build_and_passes_callbacks(monkeypatch):
    """
    train_cnn ignores the passed `model` arg and calls build_cnn() internally.
    We monkeypatch build_cnn to return a fake model that records compile() and fit() args.
    """

    class FakeModel:
        def __init__(self):
            self.compiled = False
            self.compile_args = None
            self.fit_called_with = None

        def compile(self, **kwargs):
            self.compiled = True
            self.compile_args = kwargs

        def fit(self, train_ds, validation_data, epochs, callbacks):
            # record inputs so test can assert on them
            self.fit_called_with = dict(
                train_ds=train_ds,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks,
            )
            # return a simple history object like Keras
            return SimpleNamespace(
                history={
                    "accuracy": [0.1, 0.2],
                    "val_accuracy": [0.15, 0.25],
                    "loss": [1.0, 0.8],
                    "val_loss": [0.9, 0.7],
                }
            )

    fake_model = FakeModel()

    # Monkeypatch mod.build_cnn to return our fake model
    monkeypatch.setattr(mod, "build_cnn", lambda: fake_model)

    # Provide dummy train/val dataset objects (can be anything; fake_model just records them)
    dummy_train = "TRAIN_DS"
    dummy_val = "VAL_DS"

    history = mod.train_cnn(model=None, train_ds=dummy_train, val_ds=dummy_val)

    # Ensure compile was called
    assert fake_model.compiled is True
    # Ensure fit was called
    assert fake_model.fit_called_with is not None
    assert fake_model.fit_called_with["train_ds"] == dummy_train
    assert fake_model.fit_called_with["validation_data"] == dummy_val
    # train_cnn uses epochs=7 in the call
    assert fake_model.fit_called_with["epochs"] == 7

    # Ensure a single EarlyStopping callback was passed and has the expected monitor/patience
    callbacks_list = fake_model.fit_called_with["callbacks"]
    assert isinstance(callbacks_list, list)
    assert len(callbacks_list) == 1
    cb = callbacks_list[0]
    # The callback object should be an EarlyStopping or at least expose monitor/patience properties
    assert hasattr(cb, "monitor")
    assert hasattr(cb, "patience")
    assert cb.monitor == "val_loss"
    assert cb.patience == 5

    # The train_cnn should return the history-like object from fit()
    assert isinstance(history, SimpleNamespace)
    assert "accuracy" in history.history


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