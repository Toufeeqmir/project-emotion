# Code for check_labels.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Point to your TRAINING data
TRAIN_DIR = r"E:\python\split_data\train"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

try:
    print("Reading class order from training data...")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    
    print("\n--- YOUR MODEL'S LABELS ---")
    print("This is the correct order for your CLASS_LABELS list:")
    print(train_generator.class_indices)
    print("----------------------------")

except Exception as e:
    print(f"Error: {e}")