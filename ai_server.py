import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
from focal_loss import SparseCategoricalFocalLoss
from sentiment_fusion import SentimentEngine

app = Flask(__name__)
CORS(app) # Allows your Node.js server to talk to this Python script

# --- 1. CONFIGURATION ---
# Make sure this matches your actual model filename
MODEL_PATH = "emotion_final_6class.keras"

# The 6 classes your model knows (Alphabetical order)
CLASS_LABELS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_HEIGHT, IMG_WIDTH = 128, 128

# --- 2. LOAD MODEL (Run once at startup) ---
print("⏳ Loading AI Model... (This may take a few seconds)")
try:
    # We must tell Keras about the custom Focal Loss function used during training
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
    )
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model.\nDetails: {e}")
    print(f"Make sure '{MODEL_PATH}' is in this same folder.")
    exit(1)

# Initialize your Logic Engine (The class we made in sentiment_fusion.py)
fusion_engine = SentimentEngine()

# --- 3. THE API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # A. Get Data from the POST request
        data = request.json
        image_b64 = data.get('image')  # The face image
        user_text = data.get('text', '') # What the user said (optional)

        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        # B. Decode Image (Base64 -> OpenCV Image)
        # The frontend usually sends a header like "data:image/jpeg;base64,".
        # If your Node.js code doesn't strip it, we might need to handle it here.
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
            
        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
             return jsonify({"error": "Failed to decode image"}), 400

        # C. Preprocess for the Model
        # 1. Resize to 128x128
        resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # 2. Expand dimensions (128,128,3) -> (1,128,128,3)
        input_batch = np.expand_dims(resized, axis=0)
        
        # 3. Use EfficientNet's specific preprocessing (This handles 0-255 vs 0-1 scaling)
        input_batch = tf.keras.applications.efficientnet.preprocess_input(input_batch)

        # D. Run Inference (The "Brain")
        preds = model.predict(input_batch, verbose=0)[0]
        
        # Get the highest probability
        class_idx = np.argmax(preds)
        confidence = float(np.max(preds) * 100)
        face_emotion = CLASS_LABELS[class_idx]

        # E. Run Logic (The "Coach")
        # Combine the face emotion with the text sentiment
        verdict, advice = fusion_engine.analyze_multimodal(face_emotion, user_text)

        # F. Return JSON result
        return jsonify({
            "status": "success",
            "face_emotion": face_emotion,
            "confidence": round(confidence, 2),
            "speech_text": user_text,
            "ai_verdict": verdict,
            "coaching_tip": advice
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 5001 to avoid conflict with Node.js (which usually uses 3000 or 8080)
    app.run(host='0.0.0.0', port=5001, debug=True)