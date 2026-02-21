import pickle
from pathlib import Path
import os
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

def load_test_dataset(data_dir="preprocessed_cats_dogs_images"):
    data_root_path = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/data_dir
    return  tf.keras.utils.image_dataset_from_directory(
        f"{data_root_path}/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False # Crucial for matching predictions to labels
    )

def load_training_dataset(data_dir="preprocessed_cats_dogs_images"):

    data_root_path = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/data_dir
    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_root_path}/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True,
        seed=SEED
    )

    # train_ds = (
    #     train_ds
    #     .unbatch()               
    #     .shuffle(1000, seed=SEED)    
    #     .take(100)               
    #     .batch(BATCH_SIZE)           
    #     .prefetch(tf.data.AUTOTUNE)   
    # )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_root_path}/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # val_ds = (
    #     val_ds
    #     .unbatch()                    
    #     .shuffle(1000, seed=SEED)    
    #     .take(100)                  
    #     .batch(BATCH_SIZE)            
    #     .prefetch(tf.data.AUTOTUNE)   
    # )


    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds,val_ds


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
    