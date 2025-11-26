import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
BASE_DIR = r"E:\python\New-dataset"
VAL_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = "emotion_detector_final.h5"

print("=" * 60)
print("MODEL DIAGNOSIS")
print("=" * 60)

# Load model
print(f"\n1. Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    exit(1)

model = load_model(MODEL_PATH)
print("✅ Model loaded")

# Check model input shape
print(f"\n2. Model Input Shape: {model.input_shape}")
print(f"   Expected: (None, height, width, channels)")

channels = model.input_shape[-1]
if channels == 1:
    print("   ✅ Model expects GRAYSCALE images (1 channel)")
    color_mode = 'grayscale'
elif channels == 3:
    print("   ✅ Model expects RGB images (3 channels)")
    color_mode = 'rgb'
else:
    print(f"   ❌ Unexpected channel count: {channels}")
    exit(1)

# Try loading data with correct format
print(f"\n3. Loading validation data with color_mode='{color_mode}'...")

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(model.input_shape[1], model.input_shape[2]),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    color_mode=color_mode
)

print(f"✅ Found {val_generator.samples} images")

# Evaluate with correct format
print(f"\n4. Evaluating model with {color_mode} data...")
results = model.evaluate(val_generator, verbose=1)

print(f"\n{'=' * 60}")
print("RESULTS")
print("=" * 60)
print(f"Loss:     {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")

if results[1] > 0.55:
    print("\n✅ GOOD! Accuracy matches training (~60%)")
    print("The model is working correctly!")
elif results[1] > 0.40:
    print("\n⚠️  Moderate accuracy. Model might need more training.")
else:
    print("\n❌ LOW accuracy. There's a data/model mismatch!")
    print("\nPossible issues:")
    print("1. Model was trained on different data format")
    print("2. Data preprocessing doesn't match training")
    print("3. Model file is corrupted")
    
print("\n" + "=" * 60)