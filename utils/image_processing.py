"""
Module for image processing utilities.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from PIL import Image
import io

def convert_bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to OpenCV image format.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Image as numpy array in BGR format (OpenCV)
    """
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image from bytes")
        
    return image

def convert_pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image (numpy array in BGR format)
    """
    # Convert to RGB numpy array
    image = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    return image

def convert_cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image to PIL Image.
    
    Args:
        cv2_image: OpenCV image (numpy array in BGR format)
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
                max_size: Optional[int] = None) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        max_size: Maximum dimension (width or height, whichever is larger)
        
    Returns:
        Resized image
        
    Note:
        If both width and height are provided, aspect ratio will not be preserved.
        If max_size is provided, width and height will be ignored.
    """
    h, w = image.shape[:2]
    
    # If max_size is provided, use it to determine new dimensions
    if max_size is not None:
        if h > w:
            # Height is the larger dimension
            height = max_size
            width = int(w * (height / h))
        else:
            # Width is the larger dimension
            width = max_size
            height = int(h * (width / w))
    else:
        # If only one dimension is provided, calculate the other to maintain aspect ratio
        if width is None and height is not None:
            width = int(w * (height / h))
        elif height is None and width is not None:
            height = int(h * (width / w))
        elif width is None and height is None:
            # If neither is provided, return the original image
            return image
    
    # Ensure dimensions are at least 1 pixel
    width = max(1, width)
    height = max(1, height)
    
    # Resize the image
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized

def extract_face_roi(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     padding: float = 0.0) -> np.ndarray:
    """
    Extract face region of interest (ROI) from image.
    
    Args:
        image: Input image
        bbox: Bounding box coordinates (x, y, width, height)
        padding: Additional padding around the face (percentage of bbox size)
        
    Returns:
        Face image (ROI)
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Get bbox coordinates
    x, y, width, height = bbox
    
    # Calculate padding
    pad_w = int(width * padding)
    pad_h = int(height * padding)
    
    # Calculate new coordinates with padding
    start_x = max(0, x - pad_w)
    start_y = max(0, y - pad_h)
    end_x = min(w, x + width + pad_w)
    end_y = min(h, y + height + pad_h)
    
    # Extract face ROI
    face_roi = image[start_y:end_y, start_x:end_x]
    
    return face_roi

def blend_transparent_image(background: np.ndarray, overlay: np.ndarray, 
                            position: Tuple[int, int]) -> np.ndarray:
    """
    Blend a transparent overlay image onto a background image.
    
    Args:
        background: Background image (BGR)
        overlay: Overlay image with transparency (BGRA)
        position: Position to place overlay (x, y)
        
    Returns:
        Blended image
    """
    x, y = position
    
    # Get dimensions of overlay
    h, w = overlay.shape[:2]
    
    # Check if overlay has an alpha channel
    if overlay.shape[2] != 4:
        # If not, just place it on top
        if y + h <= background.shape[0] and x + w <= background.shape[1]:
            background[y:y+h, x:x+w] = overlay
        return background
    
    # Get the alpha channel
    alpha = overlay[:, :, 3] / 255.0
    
    # Calculate region of overlay
    y_end = min(background.shape[0], y + h)
    x_end = min(background.shape[1], x + w)
    y_overlay_end = y_end - y
    x_overlay_end = x_end - x
    
    # For each color channel
    for c in range(3):
        background[y:y_end, x:x_end, c] = (
            alpha[:y_overlay_end, :x_overlay_end] * overlay[:y_overlay_end, :x_overlay_end, c] + 
            (1 - alpha[:y_overlay_end, :x_overlay_end]) * background[y:y_end, x:x_end, c]
        )
    
    return background

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to range [0, 1].
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized image back to uint8 format.
    
    Args:
        image: Normalized image ([0, 1])
        
    Returns:
        Image with pixel values in range [0, 255]
    """
    return (image * 255.0).astype(np.uint8) 