from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# =========================================================
# CONFIGURATION
# =========================================================
# This matches the filename in your screenshot
MODEL_FILENAME = 'emotion_final_6class.keras'
CLASS_NAMES = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# =========================================================
# LOAD MODEL
# =========================================================
print(f"Loading model: {MODEL_FILENAME}...")
if os.path.exists(MODEL_FILENAME):
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("Model loaded successfully!")
else:
    print(f"CRITICAL ERROR: {MODEL_FILENAME} not found!")
    model = None

def prepare_image(image_bytes):
    """Prepares the image to match the model's expected input"""
    img = Image.open(io.BytesIO(image_bytes))
    # Ensure image is RGB (converts PNG/JPEG)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize to 224x224
    img = img.resize((224, 224))
    # Convert to array and normalize (0-1)
    img_array = np.array(img)
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Prepare image
        processed_image = prepare_image(file.read())
        
        # Make prediction
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
    # Run the server
    app.run(debug=True, port=5000)