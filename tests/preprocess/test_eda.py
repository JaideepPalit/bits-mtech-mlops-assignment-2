import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import preprocess.eda as mod


@pytest.fixture(autouse=True)
def isolate_module_file(tmp_path, monkeypatch):
    """
    Make module.__file__ point into tmp_path/x/y/eda.py so that
    Path(__file__).resolve().parents[2] resolves into tmp_path for output paths.
    """
    fake_path = tmp_path / "x" / "y" / "eda.py"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text("# fake module file for tests\n")
    monkeypatch.setattr(mod, "__file__", str(fake_path.resolve()))
    return tmp_path


def _make_fake_dataset_dirs(tmp_path, n_cat=3, n_dog=3):
    """
    Create structure:
    <tmp>/data_path/Cat/img_0.jpg ...
                 /Dog/img_0.jpg ...
    Return the base data_path (string)
    """
    data_path = tmp_path / "images"
    cat_dir = data_path / "Cat"
    dog_dir = data_path / "Dog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    dog_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_cat):
        (cat_dir / f"cat_{i}.jpg").write_text("fake")
    for i in range(n_dog):
        (dog_dir / f"dog_{i}.jpg").write_text("fake")

    return str(data_path)


def test_eda_class_bal_image_res_channel_check_saves_plot_and_show_monkeypatched(tmp_path, monkeypatch, isolate_module_file):
    # Create fake data dir with some files
    data_path = _make_fake_dataset_dirs(tmp_path, n_cat=4, n_dog=4)

    # Prepare fake imread returning arrays with channels=3
    def fake_imread(path):
        p = str(path).lower()
        if p.endswith(".jpg"):
            # vary heights/widths a bit using hash of filename to ensure variety
            seed = abs(hash(p)) % 100
            h = 100 + (seed % 50)
            w = 120 + (seed % 60)
            # 3 channels
            return np.ones((h, w, 3), dtype=np.uint8)
        return None

    monkeypatch.setattr(mod.cv2, "imread", fake_imread)

    # stub plt.show to avoid UI
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    # stub plt.savefig to actually write small files
    saved = []

    def fake_savefig(path, *args, **kwargs):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok")
        saved.append(str(p))
    monkeypatch.setattr(plt, "savefig", fake_savefig)

    # Call the function (it doesn't return a df)
    mod.eda_class_bal_image_res_channel_check(data_path)

    output_root = Path(mod.__file__).resolve().parents[2] / "output" / "eda"
    expected = output_root / "eda_class_bal_image_res_channel_check.png"

    # Ensure our fake savefig was invoked and the expected file exists
    assert str(expected) in saved
    assert expected.exists()