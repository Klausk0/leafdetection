import os
import random
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------- CONFIG ----------------
MODEL_PATH = 'models/plant_disease_model.keras'
SEGMENTATION_MODEL_PATH = 'models/plant_disease_model.keras'
CLASS_INDICES_PATH = 'models/class_indices.json'
BASE_DIR = 'plant-doctor'

all_classes = ['Apple', 'Bell_Pepper', 'Cherry', 'Corn_Maize', 'Grape', 'Peach', 'Potato', 'Strawberry', 'Tomato']

# ------------- FUNCTIONS ----------------

def is_leaf_present(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found.")
        return False

    img = cv2.resize(img, (640, 480))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_like_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    if debug:
        debug_img = img.copy()
        cv2.drawContours(debug_img, leaf_like_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Leaf Detection", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bool(leaf_like_contours)

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
        raise FileNotFoundError(f"No image files found in {class_dir}")
    return random.choice(image_files)

def logistic_growth(t, a, b, c):
    return c / (1 + np.exp(-a * (t - b)))

def predict_days_to_full_infection(current_severity, a=0.2, b=10, c=1.0):
    t_range = np.linspace(0, 100, 1000)
    y_pred = logistic_growth(t_range, a, b, c)
    current_day = t_range[np.argmin(np.abs(y_pred - current_severity))]
    full_day = t_range[np.argmax(y_pred >= 0.99)]
    return max(full_day - current_day, 0)

# ------------ MAIN PIPELINE -------------

image_path = None
leaf_found = False
max_attempts = 10

# New: User upload option
custom_path = input("📤 Enter path to your leaf image or press Enter to use a random one: ").strip()

if custom_path and os.path.exists(custom_path):
    print(f"📸 Using custom image: {custom_path}")
    if is_leaf_present(custom_path):
        image_path = custom_path
        leaf_found = True
        print("🌿 Leaf detected in custom image.")
    else:
        print("🚫 No leaf detected in custom image.")
        exit(1)
else:
    print("🔄 No valid custom image provided. Selecting random leaf image from dataset...")
    for attempt in range(max_attempts):
        class_name = random.choice(all_classes)
        try:
            image_path = get_random_image_from_class(BASE_DIR, class_name)
            print(f"✅ Trying image from class '{class_name}': {image_path}")
            leaf_found = is_leaf_present(image_path)
            if leaf_found:
                print("🌿 Leaf detected.")
                break
            else:
                print("🚫 No leaf detected. Trying another image...")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            continue
    else:
        print("⛔ No valid leaf image found after multiple attempts.")
        exit(1)

# --- Load Models ---
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Classification model loaded from: {MODEL_PATH}")
    segmentation_model = load_model(SEGMENTATION_MODEL_PATH)
    print(f"✅ Segmentation model loaded from: {SEGMENTATION_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --- Load Class Indices ---
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    print(f"✅ Class indices loaded from: {CLASS_INDICES_PATH}")
except Exception as e:
    print(f"❌ Error loading class indices: {e}")
    exit(1)

# --- Classification ---
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

try:
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = list(class_indices.keys())[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    print(f"🔍 Prediction: {predicted_class} with {confidence:.2f}% confidence.")
except Exception as e:
    print(f"❌ Prediction error: {e}")
    exit(1)

plt.imshow(img)
plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.axis('off')
plt.show()

# --- Segmentation and Disease Area ---
original_img = image.load_img(image_path, target_size=(128, 128))
original_img_array = np.array(original_img)

bgr_img = cv2.cvtColor(original_img_array, cv2.COLOR_RGB2BGR)
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

diseased_mask = cv2.bitwise_not(green_mask)
diseased_mask = (diseased_mask > 0).astype(np.uint8)

diseased_pixels = np.sum(diseased_mask)
total_pixels = diseased_mask.size
percentage_disease = (diseased_pixels / total_pixels) * 100
print(f"🦠 Percentage of diseased area: {percentage_disease:.2f}%")

# --- Infection Forecast ---
days_left = predict_days_to_full_infection(percentage_disease / 100.0)
print(f"⏳ Estimated days to full infection: {days_left:.1f} days")

# --- Visualization ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title('Original Image\n' + ('🌿 Leaf Detected' if leaf_found else '🚫 No Leaf Detected'))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(diseased_mask, cmap='gray')
plt.title('Non-Green (HSV) = Diseased Area')
plt.axis('off')

plt.tight_layout()
plt.show()
