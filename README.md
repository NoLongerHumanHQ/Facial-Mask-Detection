# Facial Mask Detection

A lightweight facial mask detection system that can classify whether a person is wearing a mask or not, deployed on Streamlit for easy web interface access.

## Features

- Real-time face mask detection from webcam feed
- Image upload functionality for mask detection
- Multiple face detection in a single image
- Confidence score display for predictions
- Bounding box visualization around detected faces
- Simple and intuitive user interface

## Technical Stack

- **Framework**: TensorFlow/Keras for model development
- **Model Architecture**: MobileNetV2 (lightweight CNN) for classification
- **Face Detection**: OpenCV Haar Cascade and MediaPipe
- **Web Interface**: Streamlit
- **Image Processing**: OpenCV, Pillow

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/facial-mask-detection.git
cd facial-mask-detection
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit application:
```
streamlit run app.py
```

2. Open your browser and go to:
```
http://localhost:8501
```

## Usage

### Webcam Mode
1. Select "Webcam" from the sidebar
2. Allow camera access when prompted
3. The application will detect faces and classify mask usage in real time

### Image Upload Mode
1. Select "Upload Image" from the sidebar
2. Upload an image containing one or more faces
3. The application will process the image and display results with bounding boxes

### Settings
- Adjust the confidence threshold using the slider in the sidebar
- View model information in the sidebar

## Model Information

- Base Architecture: MobileNetV2
- Input Size: 224x224 RGB
- Classes: "with_mask", "without_mask"
- Transfer Learning: ImageNet weights
- Custom Classification Layer: Yes
- Regularization: Dropout
- Optimizer: Adam with learning rate scheduling

## Project Structure

```
facial-mask-detection/
├── app.py                 # Main Streamlit application
├── model/
│   ├── mask_detector.h5   # Trained model file
│   └── model_utils.py     # Model loading and preprocessing functions
├── utils/
│   ├── face_detection.py  # Face detection utilities
│   └── image_processing.py # Image preprocessing functions
├── requirements.txt       # Dependencies
├── README.md             # Project documentation
└── sample_images/        # Test images folder
```

## Performance

- Accuracy: >90% on test dataset
- Inference Time: <2 seconds per image
- Model Size: <50MB

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the pre-trained models
- Streamlit for the amazing web framework
- OpenCV for computer vision capabilities 