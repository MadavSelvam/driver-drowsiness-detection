# 🚗 Driver Drowsiness Detection System

## 📌 Project Overview

The **Driver Drowsiness Detection System** is an image-based computer vision application designed to identify visual indicators associated with driver drowsiness by analyzing eye conditions and yawning behavior.

The project uses Deep Learning and Computer Vision techniques to detect facial landmarks, classify eye states, and identify possible yawning.

Two separate **MobileNetV2-based Convolutional Neural Network (CNN) models** are used:
- **Eye State Classification:** Identifies whether the driver's eyes are open or closed.
- **Mouth State Classification:** Identifies whether the driver is yawning or not yawning.

The application is developed using Python and Streamlit, providing both single-image analysis and live webcam monitoring.

---

## 🎯 Project Objectives

- Detect whether the driver's eyes are open or closed.
- Identify possible yawning behavior.
- Use facial landmarks to locate the eye regions.
- Analyze uploaded images and live webcam frames.
- Display prediction results through an interactive dashboard.
- Generate a downloadable prediction report.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development and prediction |
| MobileNetV2 | Transfer learning for image classification |
| CNN | Image feature extraction and classification |
| MediaPipe | Face detection and facial landmark extraction |
| OpenCV | Image processing and video frame handling |
| NumPy | Numerical operations |
| Pandas | Prediction report generation |
| Streamlit | Interactive web application |
| Streamlit-WebRTC | Live webcam streaming |
| Pillow (PIL) | Image loading and processing |

---

## 🧠 Model Architecture

The project uses two independently trained MobileNetV2 models.

### 1. Eye State Classification Model

**Objective:** Classify the driver's eye condition.

- Classes: `Open` and `Closed`.
- Architecture: MobileNetV2 with transfer learning.
- Input size: 224 × 224 pixels.
- MediaPipe facial landmarks are used to extract the left and right eye regions.
- The trained model predicts the probability of the eyes being open.
- The application uses the predicted probability to determine whether each eye is possibly closed.

### 2. Mouth State Classification Model

**Objective:** Identify possible yawning behavior.

- Classes: `Yawn` and `No_Yawn`.
- Architecture: MobileNetV2 with transfer learning.
- Input size: 224 × 224 pixels.
- The complete input image is resized and passed to the trained classifier.
- The model generates a yawn probability.
- A configurable probability threshold is used to classify a possible yawn.

### Model Training Approach

The models are developed using Transfer Learning.

1. Load the pre-trained MobileNetV2 architecture.
2. Use ImageNet pre-trained weights.
3. Freeze the backbone layers initially.
4. Add new classification layers.
5. Train the classification head.
6. Fine-tune selected MobileNetV2 layers.
7. Evaluate model performance using the test dataset.
8. Save the trained models for application integration.

---

## ✨ Application Features

### 1. Dashboard

- Displays the project overview.
- Provides information about the classification models.
- Shows the available application features.

### 2. Image Analysis

- Upload a JPG, JPEG, or PNG image.
- Detect facial landmarks using MediaPipe.
- Extract and analyze left and right eye regions.
- Predict eye conditions.
- Estimate yawn probability.
- Display annotated images and prediction results.
- Download the prediction report in CSV format.

### 3. Live Facecam

- Access the webcam through the browser.
- Process live video frames.
- Detect facial landmarks.
- Analyze left and right eye conditions.
- Estimate yawn probability.
- Display rolling eye-closure observations and yawn counts.


## ⚙️ Installation and Setup

### Prerequisites

- Python installed.
- Visual Studio Code (VS Code).
- Git.
- A webcam for testing the Live Facecam feature.

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

Navigate to the project directory:

```bash
cd driver-drowsiness-detection
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the main dependencies:

```bash
pip install streamlit tensorflow mediapipe opencv-python numpy pandas pillow streamlit-webrtc av
```

### Step 4: Verify Model Files

Ensure the following files are available in their respective locations:

```text
models/eye_state/mobilenetv2_eye_finetuned.keras
models/mouth_state/mobilenetv2_mouth_finetuned.keras
models/face_landmarker.task
```

The application requires these trained models and the MediaPipe Face Landmarker task file to run.

### Step 5: Run the Application

Execute the following command:

```bash
streamlit run app.py
```

The application will open in your default browser.


### Mouth Model – Test Results

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 86.98% |
| Precision (Yawn) | 97.56% |
| Recall (Yawn) | 75.47% |
| F1-Score (Yawn) | 85.11% |
| ROC-AUC | 95.82% |

**Note:** These results correspond to the evaluation at a probability threshold of 0.50. The application currently uses a provisional yawn threshold of 0.30. Application predictions and evaluation metrics may therefore differ.


