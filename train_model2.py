import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- CONFIGURATION ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
NUM_CLASSES = 6         # 👈 Removed 'disgust'
EPOCHS = 24             # 👈 Limited to 24, tuned for fast convergence
INITIAL_LR = 0.001

# --- PATHS ---
BASE_DIR = r"E:\python\New-dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = "emotion_final_6class.keras"

def verify_dataset():
    print("=" * 70)
    print("STEP 1: VERIFYING DATASET")
    print("=" * 70)

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("❌ Dataset directories missing.")
        sys.exit(1)

    train_classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    if "disgust" in train_classes:
        print("⚠️  Remove 'disgust' folder from dataset!")
        sys.exit(1)

    print(f"✅ Classes detected: {train_classes}")
    return True

def create_data_generators():
    print("\n📂 Loading Data...")

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

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )

    print(f"\n✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    return train_gen, val_gen

def build_cnn():
    print("\n🧠 Building CNN Model...")

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

def setup_callbacks():
    return [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

def train_model(model, train_gen, val_gen):
    print("\n🚀 Starting Training...")

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=setup_callbacks(),
        verbose=1
    )

    print("\n✅ Training Completed.")
    return history

def evaluate_model(val_gen):
    print("\n📊 Evaluating Best Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"\n🎯 Final Validation Accuracy: {acc*100:.2f}%")
    print(f"📉 Validation Loss: {loss:.4f}")

    if acc >= 0.70:
        print("🚀 Excellent! Model accuracy improved in just 24 epochs.")
    elif acc >= 0.60:
        print("✅ Good accuracy, slightly more tuning may boost it further.")
    else:
        print("⚠️ Try a few more epochs or check data balance.")

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    verify_dataset()
    train_gen, val_gen = create_data_generators()
    model = build_cnn()
    train_model(model, train_gen, val_gen)
    evaluate_model(val_gen)

if __name__ == "__main__":
    main()
