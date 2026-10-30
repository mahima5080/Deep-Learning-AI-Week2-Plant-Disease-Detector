"""
Deep Learning & AI Internship — Week 2
Flask Web Application for Plant Disease Detection
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load Model and Classes
MODEL_PATH = "plant_disease_model.h5"
CLASS_PATH = "class_names.json"

model = None
class_names = []

def load_ai_model():
    global model, class_names
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[✔] MobileNetV2 Plant Disease Model loaded!")
    else:
        print("[!] Model file missing. Please run train.py first!")

    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH, "r") as f:
            class_names = json.load(f)

load_ai_model()

def preprocess_leaf_image(image_file) -> np.ndarray:
    """Resize and preprocess leaf upload to 224x224 RGB tensor."""
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        tensor = preprocess_leaf_image(file)
        predictions = model.predict(tensor)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])

        disease_label = class_names[predicted_idx] if class_names else f"Class {predicted_idx}"
        is_healthy = "Healthy" in disease_label

        return jsonify({
            "prediction": disease_label,
            "confidence": f"{confidence * 100:.2f}%",
            "status": "Healthy" if is_healthy else "Disease Detected"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)