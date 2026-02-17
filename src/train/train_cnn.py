from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

def build_cnn():
    model = models.Sequential([
        # Data Augmentation Layer (Applied only during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        # Rescaling layer (converts [0, 255] to [0, 1])
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(), # Reduces spatial dimensions to a single vector

        # Fully Connected
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Output 0 or 1
    ])
    
    return model

def train_cnn(model,train_ds,val_ds):
    model = build_cnn()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Stop if validation loss doesn't improve for 5 epochs
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1, # Max epochs, but early_stop will likely trigger sooner
        callbacks=[early_stop]
    )

    return history

def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
