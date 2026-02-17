import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import cv2
from tqdm import tqdm

def perform_eda_res_disjoint_aspect_ratio(data_path):
    sns.set_theme(style="whitegrid")
    categories = ['Cat', 'Dog']
    stats = []

    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "eda"
    os.makedirs(output_root_path,exist_ok=True)

    for category in categories:
        folder_path = os.path.join(data_path, category)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sampling 1000 images per class for efficiency
        for img_name in tqdm(images[:1000], desc=f"Analyzing {category}"):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                h, w, _ = img.shape
                stats.append({'label': category.capitalize(), 'width': w, 'height': h})

    df = pd.DataFrame(stats)
    df['aspect_ratio'] = df['width'] / df['height']

    # 2. Resolution Joint Plot
    plt.figure(figsize=(10, 8))
    sns.jointplot(data=df, x='width', y='height', hue='label', kind='kde', fill=True, alpha=0.6)
    plt.title('Resolution Joint Plot')
    output_path=output_root_path/"eda_resolution_dist.png"
    plt.savefig(output_path)

    # 3. Aspect Ratio Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='aspect_ratio', hue='label', kde=True, bins=30)
    plt.axvline(1.0, color='red', linestyle='--', label='Square Aspect Ratio')
    plt.title('Distribution of Image Aspect Ratios ($Width/Height$)', fontsize=14)
    plt.legend()
    output_path=output_root_path/"eda_aspect_ratio.png"
    plt.savefig(output_path)

    return df

def eda_class_bal_image_res_channel_check(data_dir:str):
    sns.set_theme(style="whitegrid")
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "eda"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"class_balance.png"

    # Prepare data for EDA
    categories = ['Cat', 'Dog']
    eda_data = []

    for category in categories:
        folder = os.path.join(data_dir, category)
        images = os.listdir(folder)
        for img_name in images[:500]: # Sample 500 from each for speed
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                eda_data.append([category, h, w, c])

    df_eda = pd.DataFrame(eda_data, columns=['label', 'height', 'width', 'channels'])

    # Visualizations
    plt.figure(figsize=(16, 5))

    # 1. Class Balance
    plt.subplot(1, 3, 1)
    sns.countplot(data=df_eda, x='label', palette='magma')
    plt.title('Class Balance (Sampled)')

    # 2. Distribution of Image Resolutions
    plt.subplot(1, 3, 2)
    plt.hist2d(df_eda['width'], df_eda['height'], bins=30, cmap='Blues')
    plt.colorbar(label='Count')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Resolution Heatmap')
    output_path=output_root_path/"image_res_heatmap.png"

    # 3. Channel Check (Confirming RGB)
    plt.subplot(1, 3, 3)
    sns.violinplot(x='label', y='height', data=df_eda)
    plt.title('Height Variance per Class')
    output_path=output_root_path/"eda_class_bal_image_res_channel_check.png"
    plt.savefig(output_path)


    plt.tight_layout()
    plt.show()