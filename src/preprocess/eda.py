import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
def plot_data_distribution(df):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "eda"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"data_distribution.png"

    sns.countplot(x="target", data=df)
    plt.title("Heart Disease Presence vs Absence")
    plt.savefig(output_path)
    plt.show()

def plot_feature_distribution(df,features:list=None):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "eda"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"feature_distribution.png"

    if not features:
        features = ["age","trestbps","chol","thalach","oldpeak"]
    df[features].hist(bins=15, figsize=(12,6), layout=(2,3))
    plt.suptitle("Feature Distributions")
    plt.savefig(output_path)
    plt.show()

def plot_correlation_heatmap(df):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "eda"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"correlation_matrix.png"

    plt.figure(figsize=(14,10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix")
    plt.savefig(output_path)
    plt.show()
