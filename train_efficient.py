import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    Activation,
    GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
# Keep using Focal Loss - it's your best tool for this dataset
from focal_loss import SparseCategoricalFocalLoss 

# --- 1. Configuration ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
NUM_CLASSES = 6 # Your 6-class dataset

# --- PATHS ---
BASE_DIR = r"E:\python\New-dataset" 
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "test")

# --- MODEL NAMES ---
# We will save your custom model here
FINAL_MODEL_PATH = "custom_cnn_90_goal.keras"

# --- TRAINING SETTINGS ---
EPOCHS = 30             # <-- UPDATED TO 30
LEARNING_RATE = 1e-3    # Start faster

def create_data_generators():
    print("--- 1. Preparing Data Pipelines ---")

    # Stronger augmentation for training from scratch
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    try:
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="sparse",
            shuffle=True,
            color_mode="rgb"
        )
        validation_generator = validation_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="sparse", 
            shuffle=False,
            color_mode="rgb"
        )
        return train_generator, validation_generator

    except FileNotFoundError:
        print(f"Error: Directory not found at {BASE_DIR}")
        sys.exit(1)

def build_custom_cnn(num_classes):
    print("--- 2. Building Custom Deep CNN (From Scratch) ---")
    
    model = Sequential()
    
    # Input Block
    model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 4 (Deep Learning)
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classification Head
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer (No Softmax for Focal Loss)
    model.add(Dense(num_classes))

    print("✅ Custom Deep CNN built.")
    return model

def main():
    train_gen, val_gen = create_data_generators()
    model = build_custom_cnn(NUM_CLASSES)

    print("--- 3. Compiling Model ---")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        # Using Focal Loss to fix the "confusion" problem
        loss=SparseCategoricalFocalLoss(gamma=2, from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(FINAL_MODEL_PATH, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-7, verbose=1)
    ]

    print("\n--- 4. Starting Training (From Scratch) ---")
    history = model.fit(
        train_gen,
        epochs=EPOCHS, 
        validation_data=val_gen,
        callbacks=callbacks,
    )
    
    print(f"\n✅ SUCCESS! Custom model saved as: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()