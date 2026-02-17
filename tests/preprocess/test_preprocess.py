import pytest
import pandas as pd
import zipfile
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.preprocess.preprocess import (
    download_dataset,
    load_inspect_dataset,
    handle_missing_value,
    extract_and_transform_categorical_features,
    save_preprocessed_data,
    run_cmd,
)

# -------------------------
# Helpers
# -------------------------

def fake_zip_bytes():
    """Create an in-memory zip file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("test.csv", "a,b,c\n1,2,3")
    buffer.seek(0)
    return buffer.read()



# -------------------------
# handle_missing_value
# -------------------------

def test_handle_missing_value():
    df = pd.DataFrame({
        "ca": ["1", "?", "2"],
        "thal": ["3", "?", "3"],
        "age": ["50", "60", "70"]
    })

    processed = handle_missing_value(df)

    assert processed.isna().sum().sum() == 0
    assert processed["ca"].dtype.kind in "if"
    assert processed["thal"].dtype.kind in "if"


# -------------------------
# extract_and_transform_categorical_features
# -------------------------

def test_extract_and_transform_categorical_features():
    df = pd.DataFrame({
        "sex": [0, 1],
        "cp": [1, 2],
        "fbs": [0, 1],
        "restecg": [0, 1],
        "exang": [0, 1],
        "slope": [1, 2],
        "thal": [3, 6],
        "target": [1, 2]
    })

    out = extract_and_transform_categorical_features(df)

    assert "target" in out.columns
    assert set(out["target"].unique()) == {1}
    assert any(col.startswith("sex_") for col in out.columns)


# -------------------------
# run_cmd
# -------------------------

@patch("src.preprocess.preprocess.subprocess.run")
def test_run_cmd_success(mock_run):
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "OK"
    mock_run.return_value.stderr = ""

    run_cmd("echo test")


@patch("src.preprocess.preprocess.subprocess.run")
def test_run_cmd_failure(mock_run):
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "error"

    with pytest.raises(SystemExit):
        run_cmd("bad command")

