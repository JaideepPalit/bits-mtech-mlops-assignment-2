# tests/test_data_io.py
import pickle
from types import SimpleNamespace
from pathlib import Path
import os
import pytest
import tensorflow as tf

# Replace this with your actual module name if it's not `data_io`
import src.train.train_util as mod


@pytest.fixture(autouse=True)
def isolate_module_file(tmp_path, monkeypatch):
    """
    Make module.__file__ point into tmp_path/x/y/data_io.py so that
    Path(__file__).resolve().parents[2] resolves into tmp_path for data/output folders.
    """
    fake_path = tmp_path / "x" / "y" / "data_io.py"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text("# fake module file for tests\n")
    monkeypatch.setattr(mod, "__file__", str(fake_path.resolve()))
    return tmp_path


def test_load_test_dataset_monkeypatched_image_loader(monkeypatch):
    captured = {}

    def fake_image_dataset_from_directory(directory, image_size, batch_size, label_mode, shuffle=None):
        # record what was passed
        captured['directory'] = directory
        captured['image_size'] = image_size
        captured['batch_size'] = batch_size
        captured['label_mode'] = label_mode
        captured['shuffle'] = shuffle
        return "SENTINEL_TEST_DS"

    monkeypatch.setattr(mod.tf.keras.utils, "image_dataset_from_directory", fake_image_dataset_from_directory)

    # Call function (uses default data_dir)
    ds = mod.load_test_dataset()

    # Verify returned sentinel
    assert ds == "SENTINEL_TEST_DS"

    # Verify directory constructed correctly relative to module.__file__
    expected_root = Path(mod.__file__).resolve().parents[2] / "data" / "preprocessed" / "preprocessed_cats_dogs_images"
    expected_dir = f"{expected_root}/test"
    assert Path(captured['directory']).as_posix() == Path(expected_dir).as_posix()

    # Check other args
    assert captured['image_size'] == mod.IMG_SIZE
    assert captured['batch_size'] == mod.BATCH_SIZE
    assert captured['label_mode'] == 'binary'
    # The load_test_dataset sets shuffle=False explicitly
    assert captured['shuffle'] is False


def test_load_training_dataset_prefetches(monkeypatch):
    # A tiny fake dataset that exposes prefetch()
    class FakeDS:
        def __init__(self, name):
            self.name = name

        def prefetch(self, buffer_size):
            # Return something that includes the buffer_size so test can assert it
            return f"prefetched_{self.name}_{buffer_size}"

    def fake_image_dataset_from_directory(directory, image_size, batch_size, label_mode, shuffle=True):
        # If directory path ends with '/train' return train dataset, else val
        if str(directory).endswith("/train") or str(directory).endswith(os.path.sep + "train"):
            return FakeDS("train")
        elif str(directory).endswith("/val") or str(directory).endswith(os.path.sep + "val"):
            return FakeDS("val")
        else:
            # fallback
            return FakeDS("other")

    monkeypatch.setattr(mod.tf.keras.utils, "image_dataset_from_directory", fake_image_dataset_from_directory)

    train_ds, val_ds = mod.load_training_dataset()

    # tf.data.AUTOTUNE is passed to prefetch; ensure our returned strings include that value
    autotune = tf.data.AUTOTUNE
    assert isinstance(train_ds, str)
    assert isinstance(val_ds, str)
    assert f"prefetched_train_{autotune}" in train_ds
    assert f"prefetched_val_{autotune}" in val_ds


def test_save_and_load_model_roundtrip(tmp_path):
    # Prepare a simple object to pickle (a dict)
    model_obj = {"a": 1, "b": [1, 2, 3]}

    # Call save_model (module.__file__ was pointed into tmp by fixture)
    mod.save_model("my_test_model.pkl", model_obj)

    # Check file exists at the expected location
    model_root = Path(mod.__file__).resolve().parents[2] / "output" / "models"
    model_file = model_root / "my_test_model.pkl"
    assert model_file.exists()
    assert model_file.stat().st_size > 0

    # Now load it back via load_model
    loaded = mod.load_model("my_test_model.pkl")
    assert loaded == model_obj


def test_load_model_missing_raises(tmp_path):
    # Ensure non-existent model raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        mod.load_model("definitely_not_there_model.pkl")