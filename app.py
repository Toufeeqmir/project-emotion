from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import os

app = Flask(__name__)
CORS(app)

# CONFIGURATION
MODEL_FILENAME = 'emotion_final_6class.keras'
CLASS_NAMES = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# LOAD MODEL
print(f"Loading model: {MODEL_FILENAME}...")
if os.path.exists(MODEL_FILENAME):
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("Model loaded successfully!")
else:
    print(f"CRITICAL ERROR: {MODEL_FILENAME} not found!")
    model = None

# LOAD FACE DETECTOR
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def prepare_image(image_bytes):
    # 1. Load Image
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR
    
    # 2. Detect Faces
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 3. SMART CROP (The Fix)
    if len(faces) > 0:
        # Get the largest face
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # Calculate a SQUARE crop to prevent distortion
        center_x, center_y = x + w // 2, y + h // 2
        max_dim = max(w, h) # Use the largest dimension
        
        # Add a little padding (10%) so we don't cut off the chin/forehead
        max_dim = int(max_dim * 1.2) 
        
        # Calculate new coordinates
        new_x = max(center_x - max_dim // 2, 0)
        new_y = max(center_y - max_dim // 2, 0)
        new_w = max_dim
        new_h = max_dim
        
        # Crop the square region
        img = img.crop((new_x, new_y, new_x + new_w, new_y + new_h))
        print("✅ Face cropped (Square)!")
    else:
        print("⚠️ No face detected. Using full image.")

    # 4. Resize to 48x48 (Now it won't be distorted!)
    img = img.resize((48, 48))
    
    # 5. Normalize
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/', methods=['GET'])
def home():
    return "<h1>Smart AI (Distortion Fixed) is Online! 🚀</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        file = request.files['file']
        processed_image = prepare_image(file.read())
        
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        predicted_label = CLASS_NAMES[predicted_class_index]
        
        return jsonify({
            'emotion': predicted_label,
            'confidence': f"{confidence*100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
