
import requests
import zipfile
import io,os
import pandas as pd
from pathlib import Path
import subprocess
import sys

def download_dataset(url:str,download_path:str):
    download_path = Path(__file__).resolve().parents[2] / "data"/"raw"/download_path
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(download_path)

    print("Downloaded and extracted files:")
    print(z.namelist())


def load_inspect_dataset(file_path:str,df_columns:list=None):
    file_path = Path(__file__).resolve().parents[2] / "data"/"raw"/file_path
    df = pd.read_csv(file_path, header=None)
    if df_columns:
        df.columns=df_columns
    else:
        df.columns = [
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"
        ]
    return df

def handle_missing_value(df):
    # Replace '?' with NaN
    df = df.replace('?', pd.NA)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)

    # Show missing
    print(df.isna().sum())

    df["ca"].fillna(df["ca"].median(), inplace=True)
    df["thal"].fillna(df["thal"].mode()[0], inplace=True)
    return df

def extract_and_transform_categorical_features(df):
    categorical_cols = ["sex","cp","fbs","restecg","exang","slope","thal"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df

def save_preprocessed_data(df,file_name):
    output_root_path = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/"heart_disease_data"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/file_name

    df.to_csv(output_path, index=False)

def run_cmd(cmd):
    """Run shell command safely and exit on failure."""
    print(f"\nRunning: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

def data_versioning_with_dvc():
    run_cmd("dvc remote add -f origin s3://dvc")
    run_cmd("dvc remote modify origin endpointurl https://dagshub.com/jaideep.palit/bits-mtech-mlops-assignment-1.s3")
    

    run_cmd("dvc remote modify origin --local access_key_id d85f991495a6411e956277b0781bd119dfac225d")
    run_cmd("dvc remote modify origin --local secret_access_key d85f991495a6411e956277b0781bd119dfac225d")

    # -------------------------------------------------
    # 4. Track dataset
    # -------------------------------------------------
    processed_file_path = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/"heart_disease_data"/"processed.cleveland.data"

    run_cmd(f"dvc add {processed_file_path}")

    # -------------------------------------------------
    # 5. Push data to DagsHub
    # -------------------------------------------------
    run_cmd("dvc push -r origin")

    print("\nDVC setup completed successfully")

def git_dvc_version():
    processed_file_path_dvc = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/"heart_disease_data"/"processed.cleveland.data.dvc"

    run_cmd("git remote set-url origin https://JaideepPalit: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@github.com/JaideepPalit/bits-mtech-mlops-assignment-1.git")
    # Add specific file
    run_cmd(f"git add {processed_file_path_dvc}")

    # Commit (safe if nothing changed)
    run_cmd(f"git commit -m 'Track {processed_file_path_dvc} using DVC' || echo 'Nothing to commit'")

    # Push to remote
    run_cmd("git push origin main")

    print("✅ File pushed to Git successfully")
    pass