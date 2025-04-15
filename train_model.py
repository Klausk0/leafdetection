import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras import layers

# Paths and setup
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "plant-doctor"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Image size can be kept larger for MobileNet and ResNet
BATCH_SIZE = 32
EPOCHS = 2  # Keep it minimal for testing purposes

# Data generators with augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Save class indices
combined_class_indices = {}
for class_name, index in train_generator.class_indices.items():
    if "_" in class_name:
        plant_name, disease_name = class_name.split("_", 1)
        combined_label = f"{plant_name}___{disease_name}"
    else:
        combined_label = class_name
    combined_class_indices[combined_label] = index

with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(combined_class_indices, f, indent=2)
print(f"✅ Class indices saved at {CLASS_INDICES_PATH}")

# --- MobileNetV2 Model with Transfer Learning ---
def build_mobilenet_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False  # Freeze the pre-trained layers

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    return model

# --- ResNet50 Model with Transfer Learning ---
def build_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False  # Freeze the pre-trained layers

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    return model

# --- Compile and Train Function ---
def compile_and_train(model, name):
    print(f"\n🚀 Training {name}...")
    
    model.compile(optimizer=SGD(momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    model_path = MODEL_DIR / f"{name}_model.keras"
    try:
        model.save(model_path)
        print(f"✅ {name} model saved at: {model_path}")
    except Exception as e:
        print(f"❌ Error saving {name} model: {e}")
    return history

# Train MobileNetV2
mobilenet_model = build_mobilenet_model()
compile_and_train(mobilenet_model, "MobileNetV2")

# Train ResNet50
resnet_model = build_resnet_model()
compile_and_train(resnet_model, "ResNet50")
