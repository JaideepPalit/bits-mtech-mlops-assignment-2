from preprocess.preprocess import download_dataset
from preprocess.preprocess import load_inspect_dataset
from preprocess.preprocess import handle_missing_value
from preprocess.preprocess import extract_and_transform_categorical_features
from preprocess.preprocess import save_preprocessed_data

from preprocess.preprocess import git_dvc_version
from preprocess.eda import plot_data_distribution
from preprocess.eda import plot_correlation_heatmap
from preprocess.eda import plot_feature_distribution
from train.train_util import separate_feature_and_target
from train.train_util import split_train_set_test_set
from train.train_util import scaling_numerical_feature
from train.train_logistic_regression import pipeline_contruction_logistic_regression
from train.train_logistic_regression import hyperparameter_tuning_logistic_regression
from train.train_logistic_regression import best_logistic_regression
from train.train_random_forest import pipeline_contruction_random_forest
from train.train_random_forest import best_random_forest
from evaluate.evaluate import evaluate_model
from evaluate.evaluate import cross_validate_performance
from train.train_random_forest import hyperparameter_tuning_random_forest
from evaluate.evaluate import cv_score
from evaluate.experiment_tracking import init_mlflow, mlflow_end_run
from evaluate.experiment_tracking import mlflow_logreg
from evaluate.experiment_tracking import mlflow_end_run
from train.train_util import save_model
from evaluate.evaluate import confusion_matrix
from evaluate.evaluate import roc_plot
from evaluate.evaluate import bar_plot_test_metric
from evaluate.experiment_tracking import mlflow_eda
from evaluate.experiment_tracking import mlflow_rf
from train.train_util import save_model
from train.train_util import load_model
from preprocess.preprocess import data_versioning_with_dvc

def preprocess_eda_train_evaluate():
    download_dataset(url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",download_path="heart_disease_data")

    df=load_inspect_dataset(file_path="heart_disease_data/processed.cleveland.data")
    df.head()

    df=handle_missing_value(df)

    df=extract_and_transform_categorical_features(df=df)

    save_preprocessed_data(df,"processed.cleveland.data")

    plot_data_distribution(df=df)

    plot_feature_distribution(df=df)

    plot_correlation_heatmap(df=df)

    X,y=separate_feature_and_target(df=df)

    X_train, X_test, y_train, y_test =split_train_set_test_set(X=X,y=y)

    preprocessor=scaling_numerical_feature(X_train=X_train)

    logreg_pipeline=pipeline_contruction_logistic_regression(preprocessor=preprocessor)

    logreg_grid=hyperparameter_tuning_logistic_regression(logreg_pipeline=logreg_pipeline,X_train=X_train, y_train=y_train)

    best_logreg=best_logistic_regression(logreg_grid=logreg_grid)

    rf_pipeline=pipeline_contruction_random_forest(preprocessor=preprocessor)

    rf_grid=hyperparameter_tuning_random_forest(rf_pipeline=rf_pipeline,X_train=X_train,y_train=y_train)

    best_rf=best_random_forest(rf_grid=rf_grid)

    logreg_results = evaluate_model(best_logreg, X_test, y_test)
    rf_results = evaluate_model(best_rf, X_test, y_test)

    logreg_results, rf_results

    cv_logreg,cv_logreg_res=cross_validate_performance(model=best_logreg,X_train=X_train,y_train=y_train)
    cv_rf,cv_rf_res=cross_validate_performance(model=best_rf,X_train=X_train,y_train=y_train)
    print("Logistic regression", cv_logreg_res)
    print("Random Forest regression", cv_rf_res)

    bar_plot_test_metric(logreg_results,rf_results)

    roc_plot(best_logreg,best_rf,X_test,y_test)

    cv_score(cv_logreg,cv_rf)

    confusion_matrix(best_rf,X_test,y_test)

    init_mlflow()
    mlflow_end_run()

    mlflow_eda()

    mlflow_logreg(logreg_grid,best_logreg,X_test,y_test,logreg_results,rf_results)

    mlflow_rf(rf_grid,best_rf, X_test, y_test,logreg_results,rf_results)

    mlflow_end_run()

    save_model("logistic_regression_pipeline.pkl",best_logreg)
    print("Logistic Regression model saved")

    save_model("random_forest_pipeline.pkl",best_rf)
    print("Random Forest model saved")

    loaded_logreg=load_model("logistic_regression_pipeline.pkl")
    loaded_rf=load_model("random_forest_pipeline.pkl")
    print("Both models loaded successfully")

    sample_X = X_test.iloc[[0]]
    sample_y = y_test.iloc[0]

    sample_X

    logreg_pred = loaded_logreg.predict(sample_X)
    logreg_prob = loaded_logreg.predict_proba(sample_X)

    print("Logistic Regression Prediction:", logreg_pred[0])
    print("Logistic Regression Probability:", logreg_prob[0])

    rf_pred = loaded_rf.predict(sample_X)
    rf_prob = loaded_rf.predict_proba(sample_X)

    print("Random Forest Prediction:", rf_pred[0])
    print("Random Forest Probability:", rf_prob[0])

    def interpret(label):
        return "Heart Disease" if label == 1 else "No Heart Disease"

    print("\nInterpretation:")
    print("Actual          :", interpret(sample_y))
    print("Logistic Reg    :", interpret(logreg_pred))
    print("Random Forest   :", interpret(rf_pred))


if __name__ == "__main__":
    preprocess_eda_train_evaluate()