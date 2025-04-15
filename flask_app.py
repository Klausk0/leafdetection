from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import numpy as np
import json
import os
import cv2
from io import BytesIO
from PIL import Image
import time
import psutil

app = Flask(__name__)

# ======= CONFIG =======
MODEL_DIR = "models"
DEFAULT_MODEL = "cnn"
MODEL_FILES = {
    "cnn": "plant_disease_model.keras",
    "resnet50": "resnet50_model.keras",
    "mobilenet": "mobilenet_model.keras"
}

CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# ======= Load Models (Lazy Load on First Use) =======
loaded_models = {}

def load_selected_model(model_name):
    if model_name not in loaded_models:
        print(f"📦 Loading model: {model_name}")
        model_path = os.path.join(MODEL_DIR, MODEL_FILES.get(model_name, MODEL_FILES[DEFAULT_MODEL]))
        loaded_models[model_name] = load_model(model_path)
    return loaded_models[model_name]

# ======= Load Class Indices =======
try:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
        class_labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    print("❌ Error loading class indices:", e)
    class_labels = {}

# ======= Preprocessing Selector =======
def get_preprocess_function(model_name):
    return {
        "resnet50": resnet_preprocess,
        "mobilenet": mobilenet_preprocess
    }.get(model_name, lambda x: x / 255.0)  # default for cnn

# ======= Utility Functions =======
def logistic_growth(t, a, b, c):
    return c / (1 + np.exp(-a * (t - b)))

def predict_days_to_full_infection(current_severity, a=0.2, b=10, c=1.0):
    t_range = np.linspace(0, 100, 1000)
    y_pred = logistic_growth(t_range, a, b, c)
    current_day = t_range[np.argmin(np.abs(y_pred - current_severity))]
    full_day = t_range[np.argmax(y_pred >= 0.99)]
    return max(full_day - current_day, 0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# ======= Model Metrics Functions =======
def get_model_size(model_name):
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    return model_size

def get_inference_time(model, input_array):
    start_time = time.time()
    model.predict(input_array)
    inference_time = time.time() - start_time  # Time in seconds
    return inference_time

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    return memory_usage

# ======= Routes =======
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    # Get model from query param, fallback to default
    selected_model = request.args.get('model', DEFAULT_MODEL).lower()
    
    # Check if the model is valid
    if selected_model not in MODEL_FILES:
        return jsonify({'error': f"Unsupported model: {selected_model}. Supported models are 'cnn', 'resnet50', 'mobilenet'."})

    model = load_selected_model(selected_model)
    preprocess = get_preprocess_function(selected_model)

    # Load and preprocess image
    img = Image.open(BytesIO(file.read())).convert("RGB")
    
    # Resize based on model requirement
    if selected_model == "cnn":
        img = img.resize((128, 128))  # CNN model expects 128x128 images
    else:
        img = img.resize((224, 224))  # ResNet50 and MobileNetV2 expect 224x224 images

    img_array = np.array(img)
    input_array = preprocess(img_array.astype(np.float32))
    input_array = np.expand_dims(input_array, axis=0)

    # Model Metrics
    model_size = get_model_size(selected_model)
    inference_time = get_inference_time(model, input_array)
    memory_usage = get_memory_usage()

    # Prediction
    preds = model.predict(input_array)
    pred_class = int(np.argmax(preds, axis=1)[0])
    raw_label = class_labels.get(pred_class, 'Unknown')

    # Decode label
    if "___" in raw_label:
        plant, disease = raw_label.split("___", 1)
    else:
        plant, disease = "Unknown", raw_label

    # Image Analysis (disease %)
    bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    diseased_mask = cv2.bitwise_not(green_mask)
    diseased_mask = (diseased_mask > 0).astype(np.uint8)

    diseased_pixels = np.sum(diseased_mask)
    total_pixels = diseased_mask.size
    percentage_disease = (diseased_pixels / total_pixels) * 100

    # Days left estimate
    days_left = predict_days_to_full_infection(percentage_disease / 100.0)

    return jsonify({
        'prediction': {
            'model': selected_model,
            'plant': plant,
            'disease': disease,
            'label': f"{plant} - {disease}",
            'percentage_disease': f"{percentage_disease:.2f}%",
            'days_left': f"{days_left:.1f} days",
            'model_size_MB': f"{model_size:.2f} MB",
            'inference_time_sec': f"{inference_time:.4f} sec",
            'memory_usage_MB': f"{memory_usage:.2f} MB"
        }
    })

if __name__ == '__main__':
    from waitress import serve
    print("🚀 Running on http://localhost:5000")
    serve(app, host="0.0.0.0", port=5000)
