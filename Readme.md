# 🎭 AI Emotion Recognition System

A Deep Learning model that detects human facial expressions in real-time. 
Built with **MobileNetV2**, **TensorFlow**, and deployed on **Hugging Face**.

## 🚀 Live Demo
**API URL:** `https://toufeeq04-emotion-backend.hf.space/predict`

## 🧠 Model Details
* **Architecture:** MobileNetV2 (Transfer Learning) + Custom Dense Layers
* **Input Size:** 48x48 pixels (RGB)
* **Classes:** 6 Emotions (Anger, Fear, Happy, Neutral, Sad, Surprise)
* **Accuracy:** ~96% on Happy/Neutral classes.

## 🛠️ Tech Stack
* **Backend:** Python, Flask, Gunicorn
* **AI Engine:** TensorFlow/Keras
* **Deployment:** Docker & Hugging Face Spaces

## 🔌 How to Connect (For Frontend)
Send a **POST** request to the API with the image file.

### JavaScript Example:
```javascript
const API_URL = "[https://toufeeq04-emotion-backend.hf.space/predict](https://toufeeq04-emotion-backend.hf.space/predict)";

const formData = new FormData();
formData.append("file", imageFile); 

fetch(API_URL, {
    method: "POST",
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log("Detected:", data.emotion);
    console.log("Confidence:", data.confidence);
});