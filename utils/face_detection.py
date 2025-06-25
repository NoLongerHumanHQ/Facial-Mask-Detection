"""
Module for face detection utilities using various methods.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict, Any, Optional, Union

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Constants for Haar cascade face detector
CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

class FaceDetector:
    """
    Class for detecting faces in images using different methods.
    Supports OpenCV Haar Cascade and MediaPipe Face Detection.
    """
    
    def __init__(self, method: str = "mediapipe", min_confidence: float = 0.5):
        """
        Initialize the face detector.
        
        Args:
            method: Detection method, either 'haar' or 'mediapipe'
            min_confidence: Minimum confidence threshold for detections
        """
        self.method = method
        self.min_confidence = min_confidence
        
        # Initialize the appropriate detector
        if method == "haar":
            try:
                self.detector = cv2.CascadeClassifier(CASCADE_PATH)
                if self.detector.empty():
                    raise ValueError("Error loading Haar cascade classifier")
            except Exception as e:
                raise ValueError(f"Error loading Haar cascade classifier: {e}")
        elif method == "mediapipe":
            self.detector = mp_face_detection.FaceDetection(min_detection_confidence=min_confidence)
        else:
            raise ValueError("Unsupported detection method. Use 'haar' or 'mediapipe'")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image as numpy array (BGR for OpenCV)
            
        Returns:
            List of dictionaries containing face information (bounding box coordinates and confidence)
        """
        if self.method == "haar":
            return self._detect_haar(image)
        else:  # mediapipe
            return self._detect_mediapipe(image)
    
    def _detect_haar(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using Haar cascade classifier.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of dictionaries with face information
        """
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Format results
        results = []
        for (x, y, w, h) in faces:
            face_dict = {
                "bbox": (x, y, w, h),
                "confidence": 1.0,  # Haar doesn't provide confidence scores
                "landmarks": None
            }
            results.append(face_dict)
        
        return results
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using MediaPipe Face Detection.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of dictionaries with face information
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        
        # Process the image
        results = self.detector.process(image_rgb)
        
        # Format results
        detected_faces = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] < self.min_confidence:
                    continue
                    
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * image_width)
                y = int(bbox.ymin * image_height)
                w = int(bbox.width * image_width)
                h = int(bbox.height * image_height)
                
                # Ensure the box is within image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, image_width - x)
                h = min(h, image_height - y)
                
                face_dict = {
                    "bbox": (x, y, w, h),
                    "confidence": float(detection.score[0]),
                    "landmarks": detection.location_data.relative_keypoints
                }
                detected_faces.append(face_dict)
        
        return detected_faces

def draw_face_box(image: np.ndarray, 
                  face: Dict[str, Any], 
                  label: str, 
                  color: Tuple[int, int, int],
                  confidence: Optional[float] = None) -> np.ndarray:
    """
    Draw bounding box and label on the image.
    
    Args:
        image: Input image
        face: Face dictionary containing bbox coordinates
        label: Label to display
        color: Color for the bounding box (BGR)
        confidence: Optional confidence score to display
        
    Returns:
        Image with bounding box and label
    """
    # Make a copy of the image to avoid modifying the original
    output = image.copy()
    
    # Get bounding box coordinates
    (x, y, w, h) = face["bbox"]
    
    # Draw bounding box
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    
    # Prepare label text with confidence if provided
    label_text = label
    if confidence is not None:
        label_text = f"{label}: {confidence:.2f}"
    
    # Draw label background rectangle
    cv2.rectangle(output, (x, y - 25), (x + len(label_text) * 12, y), color, -1)
    
    # Add label text
    cv2.putText(output, label_text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output 