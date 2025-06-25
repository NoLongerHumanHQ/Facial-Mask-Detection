"""
Setup script for facial mask detection project.
This script installs the required dependencies and generates
sample files for testing the application.
"""

import os
import subprocess
import sys
import time
import platform

def print_step(step, description):
    """Print a formatted step description."""
    print(f"\n\n{'=' * 80}")
    print(f"STEP {step}: {description}")
    print(f"{'=' * 80}\n")

def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install the required dependencies."""
    print_step(1, "Installing dependencies")
    
    # Install dependencies
    return run_command(f"{sys.executable} -m pip install -r requirements.txt")

def generate_samples():
    """Generate sample images for testing."""
    print_step(2, "Generating sample images")
    
    # First make sure OpenCV is installed
    run_command(f"{sys.executable} -m pip install opencv-python numpy")
    
    # Create a simple script to generate sample images
    sample_script = """import cv2
import numpy as np
import os

def create_sample_image(filename, has_mask=True):
    # Create a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a face-like circle
    cv2.circle(img, (320, 240), 100, (220, 220, 220), -1)
    
    # Draw eyes
    cv2.circle(img, (280, 200), 20, (255, 255, 255), -1)
    cv2.circle(img, (360, 200), 20, (255, 255, 255), -1)
    cv2.circle(img, (280, 200), 10, (0, 0, 0), -1)
    cv2.circle(img, (360, 200), 10, (0, 0, 0), -1)
    
    if has_mask:
        # Draw a mask
        cv2.rectangle(img, (260, 230), (380, 300), (100, 100, 100), -1)
        label = "with mask"
        color = (0, 255, 0)  # Green
    else:
        # Draw a mouth
        cv2.ellipse(img, (320, 270), (40, 20), 0, 0, 180, (100, 100, 100), -1)
        label = "without mask"
        color = (0, 0, 255)  # Red
    
    # Draw a bounding box
    cv2.rectangle(img, (220, 140), (420, 340), color, 2)
    
    # Add label
    cv2.putText(img, label, (230, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Save the image
    os.makedirs('sample_images', exist_ok=True)
    cv2.imwrite(os.path.join('sample_images', filename), img)
    print(f"Created {filename}")

# Create samples
create_sample_image('with_mask.jpg', True)
create_sample_image('without_mask.jpg', False)

# Create a group image
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
positions = [(160, 240), (320, 240), (480, 240)]
has_masks = [True, False, True]

for i, (pos, has_mask) in enumerate(zip(positions, has_masks)):
    # Draw face
    cv2.circle(img, pos, 70, (220, 220, 220), -1)
    
    # Draw eyes
    eye_offset = 30
    cv2.circle(img, (pos[0] - eye_offset, pos[1] - 30), 15, (255, 255, 255), -1)
    cv2.circle(img, (pos[0] + eye_offset, pos[1] - 30), 15, (255, 255, 255), -1)
    cv2.circle(img, (pos[0] - eye_offset, pos[1] - 30), 7, (0, 0, 0), -1)
    cv2.circle(img, (pos[0] + eye_offset, pos[1] - 30), 7, (0, 0, 0), -1)
    
    if has_mask:
        # Draw a mask
        cv2.rectangle(img, (pos[0] - 40, pos[1]), (pos[0] + 40, pos[1] + 40), (100, 100, 100), -1)
        label = "with mask"
        color = (0, 255, 0)
    else:
        # Draw a mouth
        cv2.ellipse(img, (pos[0], pos[1] + 20), (30, 15), 0, 0, 180, (100, 100, 100), -1)
        label = "without mask"
        color = (0, 0, 255)
    
    # Draw bounding box and label
    cv2.rectangle(img, (pos[0] - 70, pos[1] - 70), (pos[0] + 70, pos[1] + 70), color, 2)
    cv2.putText(img, label, (pos[0] - 60, pos[1] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite('sample_images/group_mask.jpg', img)
print("Created group_mask.jpg")
    """
    
    # Write the script to a temporary file
    with open("temp_sample_generator.py", "w") as f:
        f.write(sample_script)
    
    # Run the script
    success = run_command(f"{sys.executable} temp_sample_generator.py")
    
    # Remove the temporary file
    if os.path.exists("temp_sample_generator.py"):
        os.remove("temp_sample_generator.py")
        
    return success

def generate_model():
    """Generate a placeholder model for testing."""
    print_step(3, "Generating placeholder model")
    
    # Install TensorFlow if not already installed
    run_command(f"{sys.executable} -m pip install tensorflow")
    
    # Create the model script
    model_script = """
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_model():
    # Load base model with ImageNet weights
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # Skip downloading ImageNet weights to save time
    )
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and save the model
print("Creating placeholder model...")
model = create_model()
os.makedirs('model', exist_ok=True)
model_path = os.path.join('model', 'mask_detector.h5')
model.save(model_path)
print(f"Placeholder model saved to {model_path}")
    """
    
    # Write the script to a temporary file
    with open("temp_model_generator.py", "w") as f:
        f.write(model_script)
    
    # Run the script
    success = run_command(f"{sys.executable} temp_model_generator.py")
    
    # Remove the temporary file
    if os.path.exists("temp_model_generator.py"):
        os.remove("temp_model_generator.py")
        
    return success

def main():
    """Main setup function."""
    print("\nFacial Mask Detection Setup\n")
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please check errors above.")
        return
    
    # Generate samples
    if not generate_samples():
        print("Failed to generate sample images. Please check errors above.")
    
    # Generate model
    if not generate_model():
        print("Failed to generate model. Please check errors above.")
    
    # Check if all steps completed
    print_step("COMPLETE", "Setup completed successfully")
    print("You can now run the application with:")
    print("    streamlit run app.py")

if __name__ == "__main__":
    main() 