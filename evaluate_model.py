from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# === Load trained model ===
model_path = "emotion_final_6class.keras"

if not os.path.exists(model_path):
    print(f"❌ Error: Model file '{model_path}' not found!")
    exit()

model = load_model(model_path)

# === Get model input size ===
# This automatically detects if your model needs (48,48) or (224,224)
input_shape = model.input_shape[1:3]
print(f"Model expects input size: {input_shape}")

# === Dataset paths (FIXED) ===
# We use a relative path now. It looks for "New-dataset" inside the current folder.
base_dir = "New-dataset"
test_dir = os.path.join(base_dir, "test")

if not os.path.exists(test_dir):
    print(f"❌ Error: Test folder not found at: {os.path.abspath(test_dir)}")
    print("Please make sure the 'New-dataset' folder is inside 'E:\\Aicoach\\python'")
    exit()

# === Data generator ===
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# === Evaluate model ===
print("\nEvaluating model on test data...")
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Model Evaluation Complete")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Loss: {loss:.4f}")

# === Predict classes ===
print("\nGenerating predictions for Report...")
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# === Classification Report ===
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
print("\n🌀 Confusion Matrix:")
print(cm)