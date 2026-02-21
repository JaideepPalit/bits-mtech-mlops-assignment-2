import sys
import os
from types import SimpleNamespace
from pathlib import Path
import subprocess
import numpy as np
import pytest

import src.preprocess.preprocess as mod


@pytest.fixture(autouse=True)
def isolate_module_file(tmp_path, monkeypatch):
    """
    Make module.__file__ point into tmp_path/x/y/data_prep.py so that
    Path(__file__).resolve().parents[2] resolves into tmp_path for output paths.
    """
    fake_path = tmp_path / "x" / "y" / "data_prep.py"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text("# fake module file for tests\n")
    monkeypatch.setattr(mod, "__file__", str(fake_path.resolve()))
    return tmp_path


def test_download_dataset_monkeypatched_kaggle(tmp_path, monkeypatch, capsys):
    # Create a fake download folder and PetImages subfolder
    downloads_dir = tmp_path / "downloads"
    petimages_dir = downloads_dir / "PetImages"
    petimages_dir.mkdir(parents=True)

    # Monkeypatch kagglehub.dataset_download to return the downloads directory path
    def fake_dataset_download(handle, output_dir):
        # ensure the function receives the output_dir we expect (string)
        assert isinstance(output_dir, str)
        return str(downloads_dir)

    monkeypatch.setattr(mod.kagglehub, "dataset_download", fake_dataset_download)

    returned = mod.download_dataset(url="some/handle", download_path="my_dataset")

    # The function returns path.join(path, 'PetImages')
    assert returned == os.path.join(str(downloads_dir), "PetImages")

    captured = capsys.readouterr()
    assert "Path to dataset files:" in captured.out


def _create_dummy_files_in_category(base_dir: Path, category: str, n_files: int = 10):
    cat_dir = base_dir / "PetImages" / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        # create empty file names with image extensions
        (cat_dir / f"img_{i}.jpg").write_text("fake")  # content doesn't matter because we monkeypatch cv2.imread
    return cat_dir


def test_pre_process_dataset_creates_splits_and_writes_files(tmp_path, monkeypatch, isolate_module_file):
    # Create fake dataset
    data_root = tmp_path / "data_root"
    _create_dummy_files_in_category(data_root, "Cat", n_files=10)
    _create_dummy_files_in_category(data_root, "Dog", n_files=10)

    # Monkeypatch cv2 to avoid real image reading / writing
    # cv2.imread should return a dummy numpy array (non-None)
    monkeypatch.setattr(mod.cv2, "imread", lambda fp: np.ones((300, 300, 3), dtype=np.uint8))
    # cv2.resize returns a numpy array of target size (224,224,3)
    monkeypatch.setattr(mod.cv2, "resize", lambda img, size: np.ones((size[1], size[0], 3), dtype=np.uint8))
    
    # cv2.imwrite should create a small file (we'll implement a fake writer that writes a marker)
    def fake_imwrite(path, img):
        # create parent directories if not exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("written")
        return True
    monkeypatch.setattr(mod.cv2, "imwrite", fake_imwrite)

    # Run preprocessing using our fake data_root's PetImages parent
    # pre_process_dataset expects data_dir to be path to extracted dataset root / PetImages? The code uses data_dir/category
    # So pass data_root / "PetImages"
    data_dir = str(data_root / "PetImages")
    mod.pre_process_dataset(data_dir=data_dir, output_dir="preprocessed_cats_dogs_images")

    # Output directory is resolved relative to module __file__ parents[2]
    output_root = Path(mod.__file__).resolve().parents[2] / "data" / "preprocessed" / "preprocessed_cats_dogs_images"

    # For each split and category, check files exist and counts match expected splits.
    # With 10 files per category and random_state=42:
    # train = 80% -> 8 files, val = 1, test = 1 (since temp 2 split into 1/1)
    expected_counts = {"train": 8, "val": 1, "test": 1}
    for split in ("train", "val", "test"):
        for cat in ("Cat", "Dog"):
            folder = output_root / split / cat
            assert folder.exists(), f"{folder} should exist"
            files = list(folder.glob("*"))
            assert len(files) == expected_counts[split], (
                f"Expected {expected_counts[split]} files in {folder}, got {len(files)}"
            )


def test_run_cmd_success(monkeypatch, capsys):
    # Simulate subprocess.run returning success
    fake_result = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_result)

    # Should not raise and should print stdout
    mod.run_cmd("echo hello")
    captured = capsys.readouterr()
    assert "Running: echo hello" in captured.out
    assert "ok" in captured.out


def test_run_cmd_failure_raises_systemexit(monkeypatch):
    # Simulate subprocess.run returning non-zero code
    fake_result = SimpleNamespace(returncode=2, stdout="", stderr="some error\n")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_result)

    with pytest.raises(SystemExit) as excinfo:
        mod.run_cmd("bad command")
    assert excinfo.value.code == 1  # code passed to sys.exit(1) in implementation


def test_data_versioning_with_dvc_calls_run_cmd(monkeypatch, isolate_module_file):
    calls = []

    def fake_run_cmd(cmd):
        calls.append(cmd)

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    # Call the function - it will use fake_run_cmd instead of executing real commands
    mod.data_versioning_with_dvc()

    # Check that several expected dvc commands were attempted
    assert any("dvc remote add -f origin" in c for c in calls)
    assert any("dvc remote modify origin endpointurl" in c for c in calls)
    assert any(c.startswith("dvc add") for c in calls)
    assert any("dvc push -r origin" in c for c in calls)


def test_git_dvc_version_calls_run_cmd(monkeypatch, isolate_module_file):
    calls = []

    def fake_run_cmd(cmd):
        calls.append(cmd)

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    # This will not actually run git because run_cmd is replaced
    mod.git_dvc_version()

    # Validate expected git/dvc related commands were attempted
    assert any(cmd.startswith("git remote set-url origin") for cmd in calls)
    assert any(cmd.startswith("git add") for cmd in calls)
    assert any("git commit" in cmd for cmd in calls)
    assert any("git push origin main" in cmd for cmd in calls)