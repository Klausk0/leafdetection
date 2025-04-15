import os
import random
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

# ============================
#        CONFIGURATION
# ============================

MODEL_PATHS = {
    "cnn": "models/plant_disease_model.keras",
    "resnet50": "models/resnet50_model.keras",
    "mobilenet": "models/mobilenet_model.keras"
}

CLASS_INDICES_PATH = 'class_indices.json'
BASE_DIR = 'plant-doctor'  # Dataset root

# ============================
#     ARGUMENT PARSING
# ============================

parser = argparse.ArgumentParser(description="Leaf Disease Prediction")
parser.add_argument('--model', type=str, choices=['cnn', 'resnet50', 'mobilenet'], default='cnn',
                    help='Choose the model to use for prediction.')
parser.add_argument('--class_name', type=str, default='Apple',
                    help='Plant class name to pick an image from.')

args = parser.parse_args()

# ============================
#   UTILITY FUNCTIONS
# ============================

def get_all_images_from_directory(directory):
    image_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(subdir, file))
    return image_files

def get_random_image_from_class(base_dir, class_name):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        raise FileNotFoundError(f"Class folder '{class_name}' not found in {base_dir}")
    
    image_files = get_all_images_from_directory(class_dir)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {class_dir} or its subdirectories")
    
    return random.choice(image_files)

# ============================
#        MAIN PROCESS
# ============================

# Load a random image
try:
    image_path = get_random_image_from_class(BASE_DIR, args.class_name)
    print(f"✅ Selected image: {image_path}")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# Load model
model_path = MODEL_PATHS[args.model]
try:
    model = load_model(model_path)
    print(f"✅ Loaded {args.model.upper()} model from: {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Load class indices
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    print(f"✅ Class indices loaded.")
except Exception as e:
    print(f"❌ Error loading class indices: {e}")
    exit(1)

# Load and preprocess image
img = image.load_img(image_path, target_size=(128, 128))  # Ensure this size matches all model inputs
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
try:
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = list(class_indices.keys())[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    print(f"🔍 Prediction ({args.model}): {predicted_class} ({confidence:.2f}% confidence)")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    exit(1)
