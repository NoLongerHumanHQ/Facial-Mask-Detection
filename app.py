"""
Facial Mask Detection Application

This Streamlit application detects whether a person is wearing a face mask or not.
It supports both image upload and webcam inputs.
"""

import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, Union, Optional

# Import custom modules
import sys
sys.path.append(os.path.dirname(__file__))

from model.model_utils import load_mask_detection_model, predict_mask
from utils.face_detection import FaceDetector, draw_face_box
from utils.image_processing import (
    convert_bytes_to_image, convert_pil_to_cv2, convert_cv2_to_pil,
    resize_image, extract_face_roi
)

# Constants
HEADER_TITLE = "Facial Mask Detection"
HEADER_DESCRIPTION = """
    This application detects whether people in images are wearing face masks or not.
    Upload an image or use your webcam to get started.
"""
DEFAULT_CONF_THRESHOLD = 0.5
MAX_IMAGE_WIDTH = 800

# Define colors for visualization
COLOR_WITH_MASK = (0, 255, 0)     # Green
COLOR_WITHOUT_MASK = (0, 0, 255)  # Red

@st.cache_resource
def load_model():
    """
    Load the mask detection model with caching for better performance.
    """
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model", "mask_detector.h5")
        model = load_mask_detection_model(model_path)
        model_loaded = True
    except FileNotFoundError:
        st.warning("Model file not found. The application will attempt to use a placeholder model.")
        # Create a placeholder model if the real model is not available
        from model.model_utils import create_model
        model = create_model()
        model_loaded = False
    
    return model, model_loaded

def process_image(image: np.ndarray, detector: FaceDetector, model, 
                 conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> Tuple[np.ndarray, Dict]:
    """
    Process an image for mask detection.
    
    Args:
        image: Input image as numpy array
        detector: Face detector instance
        model: Loaded mask detection model
        conf_threshold: Minimum confidence threshold for predictions
    
    Returns:
        Annotated image and statistics dictionary
    """
    # Resize large images for better performance
    h, w = image.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        image = resize_image(image, width=MAX_IMAGE_WIDTH)
    
    # Detect faces
    faces = detector.detect_faces(image)
    
    # Initialize statistics
    stats = {
        "total_faces": len(faces),
        "with_mask": 0,
        "without_mask": 0,
        "avg_confidence": 0.0,
        "processing_time_ms": 0
    }
    
    # Start timing
    start_time = time.time()
    
    # If no faces detected, return original image
    if not faces:
        stats["processing_time_ms"] = (time.time() - start_time) * 1000
        return image, stats
    
    # Process each face
    confidence_sum = 0
    result_image = image.copy()
    
    for face in faces:
        # Extract face region
        face_roi = extract_face_roi(image, face["bbox"], padding=0.0)
        
        # Skip if face ROI is empty
        if face_roi.size == 0:
            continue
            
        # Predict mask
        prediction = predict_mask(model, face_roi)
        
        # Check confidence threshold
        if prediction["confidence"] < conf_threshold:
            continue
        
        # Update statistics
        confidence_sum += prediction["confidence"]
        if prediction["class"] == "with_mask":
            stats["with_mask"] += 1
            color = COLOR_WITH_MASK
        else:
            stats["without_mask"] += 1
            color = COLOR_WITHOUT_MASK
        
        # Draw bounding box
        result_image = draw_face_box(
            result_image, face, 
            prediction["class"].replace("_", " "), 
            color, prediction["confidence"]
        )
    
    # Calculate average confidence
    total_predictions = stats["with_mask"] + stats["without_mask"]
    if total_predictions > 0:
        stats["avg_confidence"] = confidence_sum / total_predictions
    
    # Calculate processing time
    stats["processing_time_ms"] = (time.time() - start_time) * 1000
    
    return result_image, stats

def display_statistics(stats: Dict[str, Any]):
    """
    Display detection statistics in the sidebar.
    
    Args:
        stats: Dictionary containing detection statistics
    """
    st.sidebar.subheader("Detection Statistics")
    
    cols = st.sidebar.columns(2)
    
    cols[0].metric("Total Faces", stats["total_faces"])
    cols[1].metric("Processing Time", f"{stats['processing_time_ms']:.1f} ms")
    
    if stats["total_faces"] > 0:
        # Create a progress bar for mask vs no mask
        st.sidebar.subheader("Mask Detection")
        
        total = max(1, stats["with_mask"] + stats["without_mask"])  # Avoid division by zero
        
        with_mask_percentage = (stats["with_mask"] / total) * 100
        without_mask_percentage = (stats["without_mask"] / total) * 100
        
        st.sidebar.progress(with_mask_percentage / 100)
        st.sidebar.text(f"With Mask: {stats['with_mask']} ({with_mask_percentage:.1f}%)")
        
        st.sidebar.progress(without_mask_percentage / 100)
        st.sidebar.text(f"Without Mask: {stats['without_mask']} ({without_mask_percentage:.1f}%)")
        
        # Average confidence
        if stats["avg_confidence"] > 0:
            st.sidebar.text(f"Avg. Confidence: {stats['avg_confidence']:.2f}")
            
        # Add a pie chart
        if stats["with_mask"] > 0 or stats["without_mask"] > 0:
            fig, ax = plt.subplots(figsize=(4, 4))
            labels = ['With Mask', 'Without Mask']
            sizes = [stats["with_mask"], stats["without_mask"]]
            colors = ['#4CAF50', '#F44336']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.sidebar.pyplot(fig)

def webcam_detection(detector: FaceDetector, model, conf_threshold: float):
    """
    Detect masks in webcam feed.
    
    Args:
        detector: Face detector instance
        model: Loaded mask detection model
        conf_threshold: Confidence threshold for predictions
    """
    # Create a placeholder for the webcam image
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Create a stop button
    stop_button_col, status_col = st.columns([1, 3])
    with stop_button_col:
        stop_button = st.button("Stop", key="stop_webcam")
    
    with status_col:
        status_text = st.empty()
    
    # Initialize webcam
    status_text.text("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Stats for display
    stats = {
        "total_faces": 0,
        "with_mask": 0,
        "without_mask": 0,
        "avg_confidence": 0.0,
        "processing_time_ms": 0
    }
    
    try:
        status_text.text("Webcam is running. Click 'Stop' to end the session.")
        
        while not stop_button:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            # Process frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame, stats = process_image(frame, detector, model, conf_threshold)
            
            # Display the result
            frame_placeholder.image(result_frame, channels="RGB", use_column_width=True)
            
            # Update stop button status
            stop_button = st.session_state.get("stop_webcam", False)
            
            # Short sleep to reduce CPU usage
            time.sleep(0.03)
    
    finally:
        # Release resources
        status_text.text("Stopping webcam...")
        cap.release()
        status_text.text("Webcam stopped.")

def main():
    """
    Main function to run the Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="Mask Detection",
        page_icon="😷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title(HEADER_TITLE)
    st.markdown(HEADER_DESCRIPTION)
    
    # Load model
    with st.spinner("Loading face mask detection model..."):
        model, model_loaded = load_model()
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Select detection method
    detection_method = st.sidebar.selectbox(
        "Face Detection Method",
        options=["mediapipe", "haar"],
        index=0
    )
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_CONF_THRESHOLD,
        step=0.05
    )
    
    # Initialize face detector
    detector = FaceDetector(method=detection_method, min_confidence=conf_threshold)
    
    # Display model information in sidebar
    with st.sidebar.expander("Model Information", expanded=False):
        st.write("Model: MobileNetV2")
        st.write("Input Size: 224x224 RGB")
        st.write("Classes: with_mask, without_mask")
        if not model_loaded:
            st.warning("Using placeholder model - real model file not found.")
    
    # Input method selection
    input_method = st.sidebar.radio("Input Method", ["Upload Image", "Webcam"])
    
    # Main area - Process based on input method
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image containing faces",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Detect Masks"):
                with st.spinner("Processing image..."):
                    # Convert to OpenCV format
                    cv2_image = convert_pil_to_cv2(image)
                    
                    # Process image
                    result_image, stats = process_image(
                        cv2_image, detector, model, conf_threshold
                    )
                    
                    # Display result
                    st.image(
                        result_image, 
                        caption="Detection Result",
                        channels="BGR",
                        use_column_width=True
                    )
                    
                    # Display statistics
                    display_statistics(stats)
                    
                    # Download button for result
                    pil_result = convert_cv2_to_pil(result_image)
                    buf = io.BytesIO()
                    pil_result.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.download_button(
                            label="Download Result",
                            data=byte_im,
                            file_name="mask_detection_result.png",
                            mime="image/png"
                        )
    
    else:  # Webcam mode
        st.write("Webcam mode will detect masks in real-time.")
        start_webcam = st.button("Start Webcam")
        
        if start_webcam:
            webcam_detection(detector, model, conf_threshold)

if __name__ == "__main__":
    main() 