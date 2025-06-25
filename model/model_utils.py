"""
Module for model-related utilities for the face mask detection system.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from typing import Tuple, Union, List, Dict, Any

# Constants
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_detector.h5")
IMG_SIZE = 224
CLASS_LABELS = ["with_mask", "without_mask"]

def load_mask_detection_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Load the pre-trained mask detection model.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        Loaded TensorFlow/Keras model
        
    Raises:
        FileNotFoundError: If the model file is not found
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = load_model(model_path)
    return model

def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """
    Preprocess a face image for model prediction.
    
    Args:
        face_img: Face image as numpy array
        
    Returns:
        Preprocessed image ready for model prediction
    """
    # Resize to the expected input size
    face = tf.image.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face

def predict_mask(model: tf.keras.Model, face_img: np.ndarray) -> Dict[str, Any]:
    """
    Predict whether a face is wearing a mask or not.
    
    Args:
        model: Loaded mask detection model
        face_img: Preprocessed face image
        
    Returns:
        Dictionary containing prediction class and confidence score
    """
    # Preprocess the face image
    processed_face = preprocess_face(face_img)
    
    # Make prediction
    prediction = model.predict(processed_face, verbose=0)[0]
    
    # Get the class with highest probability
    max_index = np.argmax(prediction)
    label = CLASS_LABELS[max_index]
    confidence = float(prediction[max_index])
    
    return {
        "class": label,
        "confidence": confidence,
        "raw_predictions": prediction.tolist()
    }

def create_model() -> tf.keras.Model:
    """
    Create and compile the mask detection model architecture.
    This function is used for model training, not for inference.
    
    Returns:
        Compiled TensorFlow/Keras model
    """
    # Load the MobileNetV2 network without the top layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the top layers for classification
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: with_mask, without_mask
    ])
    
    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    
    return model 