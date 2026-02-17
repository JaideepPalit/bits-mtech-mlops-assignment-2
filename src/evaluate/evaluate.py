from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

def summarize_cv(cv_results):
    return {k: np.mean(v) for k, v in cv_results.items() if "test_" in k}

def cross_validate_performance(model,X_train, y_train,scoring:list=None):
    if not scoring:
        scoring = ["accuracy", "precision", "recall", "roc_auc"]

    cv_res = cross_validate(
        model, X_train, y_train, scoring=scoring, cv=5
    )

    return cv_res, summarize_cv(cv_res)


def bar_plot_test_metric(logreg_results,rf_results):
    # Convert results to DataFrame
    results_df = pd.DataFrame({
        "Logistic Regression": logreg_results,
        "Random Forest": rf_results
    })

    results_df = results_df.T  # models as rows

    # Plot
    results_df.plot(kind="bar", figsize=(6,4))
    plt.title("Model Performance Comparison (Test Set)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def roc_plot(best_logreg,best_rf,X_test,y_test):

    # Predict probabilities
    logreg_probs = best_logreg.predict_proba(X_test)[:, 1]
    rf_probs = best_rf.predict_proba(X_test)[:, 1]

    # ROC data
    fpr_lr, tpr_lr, _ = roc_curve(y_test, logreg_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


def cv_score(cv_logreg,cv_rf):
    cv_summary = pd.DataFrame({
        "Logistic Regression": summarize_cv(cv_logreg),
        "Random Forest": summarize_cv(cv_rf)
    }).T

    cv_summary.columns = [c.replace("test_", "").upper() for c in cv_summary.columns]

    cv_summary.plot(kind="bar", figsize=(6,4))
    plt.title("Cross-Validation Performance (5-Fold Mean)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def confusion_matrix(best_rf,X_test,y_test):
    ConfusionMatrixDisplay.from_estimator(
        best_rf,
        X_test,
        y_test,
        cmap="Blues"
    )

    plt.title("Random Forest – Confusion Matrix")
    plt.show()
