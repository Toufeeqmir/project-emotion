#
# NEW, CORRECT fine_tune.py
# This script is for your 70% custom-built CNN
#

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

# --- 1. Configuration ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
NUM_CLASSES = 6 

# --- PATHS ---
# Make sure this path is 100% correct
BASE_DIR = r"E:\python\New-dataset" 
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "test")

# --- MODEL NAMES ---
# This is your 70% model that you already trained
EXISTING_MODEL_PATH = "emotion_final_6class.keras" 
# This is the new, final model we will create
FINAL_MODEL_PATH = "finetuned_70_plus_model.keras" 

# --- 4. TRAINING SETTINGS ---
FINE_TUNE_EPOCHS = 24     # Max 24 epochs for fine-tuning
FINE_TUNE_LR = 1e-6       # Tiny learning rate for fine-tuning
       
def create_data_generators():
    """Creates and returns the train and validation data generators."""
    print("--- 1. Preparing Data Pipelines ---")

    # We must use the same augmentation as your 70% model
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical", # Your model uses categorical
            shuffle=True,
            color_mode="rgb"
        )
        validation_generator = validation_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical", # Your model uses categorical
            shuffle=False,
            color_mode="rgb"
        )
        
        print(f"Found {train_generator.samples} training images.")
        print(f"Found {validation_generator.samples} validation images.")
        print(f"Class Indices: {train_generator.class_indices}")
        print("✅ Data pipelines ready.")
        return train_generator, validation_generator

    except FileNotFoundError:
        print(f"Error: Directory not found. Make sure BASE_DIR is correct.")
        print(f"Current BASE_DIR: {BASE_DIR}")
        print("Please check the path and try again.")
        sys.exit(1)


def main():
    # 1. Prepare Data
    train_gen, val_gen = create_data_generators()
    
    num_classes = len(train_gen.class_indices)
    if num_classes != NUM_CLASSES:
        print(f"Error: Expected {NUM_CLASSES} classes, but found {num_classes} in the folder.")
        sys.exit(1)

    # 2. Load Your 70% Model
    print("\n" + "="*30)
    print(f"--- Loading your 70% model from {EXISTING_MODEL_PATH} ---")
    print("="*30)
    
    try:
        model = load_model(EXISTING_MODEL_PATH)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure your model file is named correctly and is in the E:\python folder.")
        sys.exit(1)

    # 3. --- STARTING FINE-TUNING ---
    # We re-compile with a tiny learning rate to fine-tune
    print("✅ Re-compiling model for fine-tuning...")
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR), # Critical tiny LR
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    
    # 4. Define Callbacks
    callbacks_list = [
        ModelCheckpoint(FINAL_MODEL_PATH, monitor="val_loss", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-8, mode="min", verbose=1)
    ]
    
    # 5. Start Fine-Tuning!
    print("\n" + "="*30)
    print("--- STARTING FINE-TUNING ---")
    print("="*30)

    history_phase2 = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS, # This will run for 24 epochs
        initial_epoch=0,        # This is a new run, so we start at 0
        validation_data=val_gen,
        callbacks=callbacks_list,
    )
    
    print("\n" + "="*30)
    print("✅✅✅ FULL TRAINING COMPLETE. ✅✅✅")
    print(f"Final, best model saved to: {FINAL_MODEL_PATH}")
    print("="*30)

if __name__ == "__main__":
    main()