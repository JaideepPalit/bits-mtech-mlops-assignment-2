
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def pipeline_contruction_logistic_regression(preprocessor):
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

def hyperparameter_tuning_logistic_regression(logreg_pipeline,X_train, y_train):
    logreg_param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"]
    }

    logreg_grid = GridSearchCV(
        logreg_pipeline,
        logreg_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    logreg_grid.fit(X_train, y_train)
    return logreg_grid

def best_logistic_regression(logreg_grid):
    best_logreg = logreg_grid.best_estimator_
    print("Best Logistic Regression params:", logreg_grid.best_params_)
    return best_logreg
