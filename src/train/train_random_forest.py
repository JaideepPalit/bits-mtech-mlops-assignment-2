
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def pipeline_contruction_random_forest(preprocessor):
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ])


def hyperparameter_tuning_random_forest(rf_pipeline,X_train, y_train):
    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)
    return rf_grid

def best_random_forest(rf_grid):
    best_rf = rf_grid.best_estimator_
    print("Best Random Forest params:", rf_grid.best_params_)
    return best_rf
