# Driver Drowsiness Detection Using Deep Learning

## Project Overview

Driver Drowsiness Detection is a deep learning-based application designed to identify visual indicators of driver fatigue by analyzing eye conditions and yawning behavior.

The project uses two independently trained **MobileNetV2 models** to classify eye states and mouth states. The predictions from both models are used to identify potential signs of drowsiness.

The application provides an image-based analysis interface through Streamlit.

## Objectives

- Detect whether the driver's eyes are Open or Closed.
- Identify whether the driver is Yawning or Not Yawning.
- Combine eye-state and mouth-state predictions to assess potential drowsiness.
- Provide an easy-to-use interface for image-based analysis.
- Explore the application of CNN-based deep learning models in driver safety.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development |
| MobileNetV2 | Eye-state and mouth-state classification |
| OpenCV | Image processing |
| MediaPipe | Facial landmark detection |
| NumPy | Numerical operations |
| Streamlit | Web application interface |
| Matplotlib | Model evaluation and visualization |
| Scikit-learn | Model evaluation metrics |

## Project Architecture

The application follows this general workflow:

1. Input: The user uploads a driver image.
2. Face Processing: Facial landmarks are detected using MediaPipe.
3. Eye Analysis: Eye regions are extracted and passed to the trained eye-state model.
4. Mouth Analysis: The input image is passed to the trained mouth-state model.
5. Classification: The models predict the eye and mouth conditions.
6. Fatigue Assessment: The predictions are processed using the fatigue assessment logic.
7. Output: The application displays the detected conditions and assessment result.

## Deep Learning Models

### 1. Eye-State Detection – MobileNetV2

The eye-state classification model identifies two categories:

- Open
- Closed

The model analyzes the extracted eye regions and predicts the corresponding eye condition.

### 2. Yawning Detection – MobileNetV2

The mouth-state classification model identifies two categories:

- Yawn
- No_Yawn

The model analyzes the input image and predicts whether yawning is present.

### 3. Fatigue Assessment

The predictions from the eye-state and mouth-state models are processed through the fatigue assessment logic.

The application uses these visual indicators to identify potential drowsiness.

## Project Structure

```text
driver-drowsiness-detection/
│
├── models/
│   ├── eye_state/
│   │   └── mobilenetv2_eye_finetuned.keras
│   │
│   ├── mouth_state/
│   │   └── mobilenetv2_mouth_finetuned.keras
│   │
│   └── face_landmarker.task
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_mobilenetv2_eye.ipynb
│   └── 04_mobilenetv2_mouth.ipynb
│
├── app.py
├── fatigue_logic.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The `data/` and `outputs/` directories are excluded from the GitHub repository.*

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- Visual Studio Code.
- Git.

### Step 1: Clone the Repository

```bash
git clone https://github.com/MadavSelvam/driver-drowsiness-detection.git
```

### Step 2: Navigate to the Project Directory

```bash
cd driver-drowsiness-detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## Application Features

- Image Upload: Upload an image for driver condition analysis.
- Eye-State Classification: Detect open and closed eye conditions.
- Yawning Classification: Identify yawning behavior.
- Fatigue Assessment: Combine visual predictions to identify potential drowsiness.
- Prediction Results: Display the classification results through the Streamlit interface.

## Model Evaluation

The trained models are evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
