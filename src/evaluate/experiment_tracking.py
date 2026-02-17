import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from pathlib import Path
import os
import dagshub

def init_mlflow():
    #dagshub.init(repo_owner='jaideep.palit', repo_name='bits-mtech-mlops-assignment-1', mlflow=True)
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = "jaideep.palit"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "d85f991495a6411e956277b0781bd119dfac225d"
    #os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/jaideep.palit/bits-mtech-mlops-assignment-1.mlflow"

    mlflow.set_tracking_uri(
        "https://dagshub.com/jaideep.palit/bits-mtech-mlops-assignment-1.mlflow"
    )
    mlflow.set_experiment("Heart Disease Classification")



def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


def mlflow_logreg(logreg_grid,best_logreg,X_test, y_test,logreg_results,rf_results):
    with mlflow.start_run(run_name="Logistic_Regression"):
    
        # Log hyperparameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", logreg_grid.best_params_["model__C"])
        mlflow.log_param("penalty", "l2")
        
        # Train already done via GridSearch
        metrics = compute_metrics(best_logreg, X_test, y_test)
        
        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Log model
        mlflow.sklearn.log_model(best_logreg, artifact_path="model")

        log_roc_curve(best_logreg, X_test, y_test, "logreg_roc.png")

        mlflow_metric_comparision(logreg_results,rf_results)

        print("Logged Logistic Regression run")

def mlflow_rf(rf_grid,best_rf, X_test, y_test,logreg_results,rf_results):
    with mlflow.start_run(run_name="Random_Forest"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", rf_grid.best_params_["model__n_estimators"])
        mlflow.log_param("max_depth", rf_grid.best_params_["model__max_depth"])
        mlflow.log_param("min_samples_split", rf_grid.best_params_["model__min_samples_split"])
        
        metrics = compute_metrics(best_rf, X_test, y_test)
        
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        mlflow.sklearn.log_model(best_rf, artifact_path="model")
        log_roc_curve(best_rf, X_test, y_test, "rf_roc.png")

        mlflow_metric_comparision(logreg_results,rf_results)

        
        print("Logged Random Forest run")

def log_roc_curve(model, X_test, y_test, filename):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "evaluate"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/filename

    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    mlflow.log_artifact(output_path)

def mlflow_metric_comparision(logreg_results,rf_results):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "evaluate"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"model_comparison.png"
    results_df = pd.DataFrame({
        "Logistic Regression": logreg_results,
        "Random Forest": rf_results
    }).T

    plt.figure(figsize=(8,5))
    results_df.plot(kind="bar")
    plt.title("Model Comparison Metrics")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    mlflow.log_artifact(output_path)

def mlflow_eda():
    with mlflow.start_run(run_name="EDA"):
        output_root_path = Path(__file__).resolve().parents[2] / "output"/ "eda"
        data_distribution_output_path=output_root_path/"data_distribution.png"
        mlflow.log_artifact(data_distribution_output_path)

        feature_distribution_output_path=output_root_path/"feature_distribution.png"
        mlflow.log_artifact(feature_distribution_output_path)

        correlation_matrix_output_path=output_root_path/"correlation_matrix.png"
        mlflow.log_artifact(correlation_matrix_output_path)

def mlflow_end_run():
    mlflow.end_run()








