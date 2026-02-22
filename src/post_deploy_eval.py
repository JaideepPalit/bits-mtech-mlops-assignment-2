import os
import glob
import argparse
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from pathlib import Path
from evaluate.experiment_tracking import init_mlflow, mlflow_end_run,mlflow_cnn_post_deploy_eval
from evaluate.evaluate import plot_cnn_confusion_matrix
from preprocess.preprocess import download_dataset
from preprocess.preprocess import pre_process_dataset



# ---------------- CONFIG ----------------
DEFAULT_URL = "http://34.102.150.120/predict"
FIELD_NAME = "file"
TIMEOUT = 60
THRESHOLD = 0.5  # used only if needed
# ----------------------------------------



def get_images(data_dir, limit=None):
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_dir, p), recursive=True))
    files = sorted(files)
    if limit:
        files = files[:limit]
    return files


def send_request(url, image_path):
    with open(image_path, "rb") as f:
        files = {FIELD_NAME: f}
        response = requests.post(url, files=files, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()


def post_deploy_evaluate():
    # url="http://34.102.150.120/predict"
    url="http://34.102.150.120/predict"

    limit=100
    data_dir = Path(__file__).resolve().parents[1] / "data"/ "preprocessed"/"preprocessed_cats_dogs_images"/"test"

    images = get_images(data_dir, limit)

    print(f"Sending {len(images)} images to {url}")

    results = []

    for idx, img_path in enumerate(images, 1):
        true_label = os.path.basename(os.path.dirname(img_path)).lower()

        try:
            response_json = send_request(url, img_path)

            pred_label = response_json["prediction"].lower()
            confidence = float(response_json["confidence"])
            raw_prob = float(response_json["raw_probability"])

            results.append({
                "image": img_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence_percent": confidence,
                "raw_probability": raw_prob
            })

            print(f"[{idx}/{len(images)}] {os.path.basename(img_path)} "
                  f"-> pred={pred_label} ({confidence}%) true={true_label}")

        except Exception as e:
            print(f"[{idx}/{len(images)}] ERROR: {img_path} → {e}")

    df = pd.DataFrame(results)

    # Convert labels to binary (Dog = 1, Cat = 0)
    y_true = (df["true_label"] == "cat").astype(int)
    y_pred = (df["pred_label"] == "cat").astype(int)
    y_prob = df["raw_probability"].astype(float)

    print("\n===== Post-Deployment Metrics =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (Cat):", precision_score(y_true, y_pred))
    print("Recall (Cat):", recall_score(y_true, y_pred))
    print("F1 Score (Cat):", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    metrics={
        "Accuracy": f"{accuracy_score(y_true, y_pred)}",
        "Precision": f"{precision_score(y_true, y_pred)}",
        "Recall": f"{recall_score(y_true, y_pred)}",
        "F1-Score": f"{f1_score(y_true, y_pred)}"
        }
    
    plot_cnn_confusion_matrix(y_true, y_pred, "confusion_matrix_post_deploy_eval.png")

    init_mlflow()
    mlflow_end_run()
    mlflow_cnn_post_deploy_eval(metrics)
    mlflow_end_run()



if __name__ == "__main__":
    data_dir=download_dataset(url="bhavikjikadara/dog-and-cat-classification-dataset",download_path="cats_and_dogs_classification_data")
    pre_process_dataset(data_dir=data_dir)
    post_deploy_evaluate()