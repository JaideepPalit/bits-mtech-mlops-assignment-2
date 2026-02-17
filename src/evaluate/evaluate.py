from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from tensorflow.keras.preprocessing import image

def evaluate_cnn(model,test_ds):
    # Get true labels and predictions
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_probs = model.predict(test_ds)
    y_pred = (y_probs > 0.5).astype(int)

    return y_true, y_probs,  y_pred

def plot_cnn_confusion_matrix(y_true, y_pred):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "evaluate"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"cnn_confusion_matrix.png"

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()

def plot_cnn_roc_curve(y_true, y_probs):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "evaluate"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"cnn_roc.png"

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.show()
    return roc_auc


def plot_cnn_summary(y_true, y_pred, roc_auc):
    output_root_path = Path(__file__).resolve().parents[2] / "output"/ "evaluate"
    os.makedirs(output_root_path,exist_ok=True)
    output_path=output_root_path/"cnn_summary.png"

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = np.mean(y_true == y_pred)

    print("-" * 30)
    print(f"Final Test Evaluation:")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("-" * 30)

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [accuracy, precision, recall, f1, roc_auc]

    # 2. Set the visual style
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", len(metrics_names))

    # 3. Create the Bar Plot
    ax = sns.barplot(x=metrics_names, y=metrics_values, palette=palette)

    # 4. Add data labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=12, fontweight='bold')

    # 5. Final touches
    plt.ylim(0, 1.1) # Scale to 1.1 to leave room for labels
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.title('Final Test Evaluation Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path)

    plt.show()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC":f1,
        "F1-Score": roc_auc
    }

def get_sample_test_data():
    data_dir = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/"preprocessed_cats_dogs_images"
    test_cat_path = f"{data_dir}/test/Cat"
    test_dog_path = f"{data_dir}/test/Dog"

    cat_img = os.path.join(test_cat_path, os.listdir(test_cat_path)[0])
    dog_img = os.path.join(test_dog_path, os.listdir(test_dog_path)[0])

    return cat_img, dog_img

def get_prediction(cnn_model,img_path, img_array=None):
    if img_array is None:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    raw_prob = cnn_model.predict(img_array)[0][0]
    
    label = "Dog" if raw_prob > 0.5 else "Cat"
    confidence = raw_prob if raw_prob > 0.5 else (1 - raw_prob)
    
    return label, confidence, raw_prob