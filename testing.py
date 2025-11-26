import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Load trained model ===
model_path = "emotion_final_6class.keras"
model = load_model(model_path)
print("✅ Model loaded successfully!")

# === Class labels (must match training order) ===
class_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === Detect model input type and shape ===
input_shape = model.input_shape[1:3]
channels = model.input_shape[-1]
print(f"Model expects input size: {input_shape} and channels: {channels}")

# === Initialize Haar cascade for face detection ===
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Open webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Starting live camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = frame[y:y + h, x:x + w]

        # === Preprocess ROI ===
        if channels == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, input_shape)
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        if channels == 1:
            roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        # === Prediction ===
        preds = model.predict(roi, verbose=0)[0]
        label_index = np.argmax(preds)
        label = class_labels[label_index]
        confidence = preds[label_index]

        # === Display results ===
        text = f"{label} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection (Press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed.")
