from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers

# ----------------- CONFIG -----------------
IMAGE_SIZE = 224             
SEED = 123
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_PATH = "cnn_scratch_best.h5"
MIXED_PRECISION = False      
# ------------------------------------------------

if MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

AUTOTUNE = tf.data.AUTOTUNE
def build_cnn(image_size=IMAGE_SIZE, dropout=0.4, l2=1e-4):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = inputs

    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Efficient separable conv blocks
    for filters in [32, 64, 128]:
        x = layers.SeparableConv2D(filters, 3, padding='same',
                                   depthwise_regularizer=regularizers.l2(l2),
                                   pointwise_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

    # Bottleneck
    x = layers.SeparableConv2D(256, 3, padding='same',
                               depthwise_regularizer=regularizers.l2(l2),
                               pointwise_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)

    final_dense_kwargs = {"dtype": "float32"} if MIXED_PRECISION else {}
    outputs = layers.Dense(1, activation='sigmoid', **final_dense_kwargs)(x)

    model = models.Model(inputs, outputs, name="cnn_scratch")
    return model

def compile_model(model, lr=LEARNING_RATE):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',                        # you'll get val_accuracy automatically
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

def train_cnn(model, train_ds, val_ds, epochs=EPOCHS):
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
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
