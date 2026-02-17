from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path
import os

def separate_feature_and_target(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X,y

def split_train_set_test_set(X,y):
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


def scaling_numerical_feature(X_train):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="passthrough"
    )
    return preprocessor

def save_model(model_name:str,model):
    model_root_path = Path(__file__).resolve().parents[2] / "output"/ "models"
    os.makedirs(model_root_path,exist_ok=True)
    model_path=model_root_path/model_name
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def load_model(model_name:str):
    model_path = Path(__file__).resolve().parents[2] / "output"/ "models"/model_name
    
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
        return loaded_model
    