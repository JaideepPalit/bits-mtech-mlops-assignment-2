import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from pathlib import Path
import os
import dagshub
import io


def init_mlflow():
    #dagshub.init(repo_owner='jaideep.palit', repo_name='bits-mtech-mlops-assignment-1', mlflow=True)
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = "jaideep.palit"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "d85f991495a6411e956277b0781bd119dfac225d"
    #os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/jaideep.palit/bits-mtech-mlops-assignment-1.mlflow"

    mlflow.set_tracking_uri(
        "https://dagshub.com/jaideep.palit/bits-mtech-mlops-assignment-1.mlflow"
    )
    mlflow.set_experiment("Cats and Dogs Image Classification")



def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


def mlflow_cnn(cnn_model,metrics):
    with mlflow.start_run(run_name="CNN"):
        output_root_path = Path(__file__).resolve().parents[2] / "output"/ "evaluate"
        mlflow.log_param("model", "CNN")
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("optimizer", "adam")

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        mlflow.sklearn.log_model(cnn_model, artifact_path="model")

        stream = io.StringIO()
        cnn_model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()

        # 2. Log the string to MLflow as a .txt file
        mlflow.log_text(summary_str, "model_summary.txt")
        print("Model summary logged to MLflow artifacts.")

        cnn_confusion_matrix_output_path=output_root_path/"cnn_confusion_matrix.png"
        mlflow.log_artifact(cnn_confusion_matrix_output_path)

        cnn_roc_output_path=output_root_path/"cnn_roc.png"
        mlflow.log_artifact(cnn_roc_output_path)

        cnn_summary_output_path=output_root_path/"cnn_summary.png"
        mlflow.log_artifact(cnn_summary_output_path)

        print("Logged CNN run")

def mlflow_eda():
    with mlflow.start_run(run_name="EDA"):
        output_root_path = Path(__file__).resolve().parents[2] / "output"/ "eda"
        class_bal_image_res_channel_check_output_path=output_root_path/"class_bal_image_res_channel_check.png"
        mlflow.log_artifact(class_bal_image_res_channel_check_output_path)

        eda_aspect_ratio_output_path=output_root_path/"eda_aspect_ratio.png"
        mlflow.log_artifact(eda_aspect_ratio_output_path)

        eda_resolution_dist_output_path=output_root_path/"eda_resolution_dist.png"
        mlflow.log_artifact(eda_resolution_dist_output_path)

def mlflow_end_run():
    mlflow.end_run()








