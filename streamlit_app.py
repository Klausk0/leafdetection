from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import json
import os
import io  # ✅ Needed for BytesIO

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/plant_disease_model.keras'
model = load_model(MODEL_PATH)

# Load class indices
class_indices = {}
try:
    with open("class_indices.json", "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "{":
            class_indices = json.load(f)
        else:
            for line in f:
                if ":" in line:
                    class_name, class_index = line.strip().split(":")
                    class_indices[int(class_index.strip())] = class_name.strip()
except Exception as e:
    print("❌ Error loading class indices:", e)

# Reverse key-value if needed
if isinstance(list(class_indices.keys())[0], str):
    class_labels = {v: k for k, v in class_indices.items()}
else:
    class_labels = class_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            # ✅ Fix: Read image from file as BytesIO
            img = image.load_img(io.BytesIO(file.read()), target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds = model.predict(img_array)
            pred_class = np.argmax(preds, axis=1)[0]
            pred_label = class_labels.get(pred_class, 'Unknown')
            return jsonify({'prediction': pred_label})
        else:
            return jsonify({'error': 'Invalid file format'})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    from waitress import serve
    print("🚀 Running on http://localhost:5000")
    serve(app, host="0.0.0.0", port=5000)
