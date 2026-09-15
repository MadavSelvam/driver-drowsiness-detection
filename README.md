# Driver Drowsiness Detection

A deep learning-based computer vision application that detects driver fatigue using eye-state and yawning analysis.

## Project Overview

The application classifies driver images into four categories:

- `Closed`
- `Open`
- `no_yawn`
- `yawn`

The predictions are then mapped to three fatigue levels:

- **Alert**
- **Mild Fatigue**
- **Severe Fatigue**

Two deep learning models were developed and compared:

1. Custom CNN
2. MobileNetV2 Transfer Learning

MobileNetV2 was selected as the final model based on its superior performance.

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Custom CNN | 48.74% | 36.84% | 48.74% | 32.84% |
| **MobileNetV2** | **84.14%** | **89.90%** | **84.14%** | **82.62%** |

MobileNetV2 improved accuracy by approximately **35.40 percentage points** compared with the Custom CNN.

## Fatigue Mapping

| Detected Class | Fatigue Level | Score |
|---|---|---:|
| Open | Alert | 0 |
| no_yawn | Alert | 0 |
| yawn | Mild Fatigue | 1 |
| Closed | Severe Fatigue | 2 |

The fatigue score is generated using an explainable rule-based mapping and does not directly represent the model's class index.

## Project Structure

```text
driver-drowsiness-detection/
│
├── dataset/
│   └── train/
│       ├── Closed/
│       ├── Open/
│       ├── no_yawn/
│       └── yawn/
│
├── models/
│   ├── custom_cnn_final.keras
│   └── mobilenetv2_final.keras
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_mobilenetv2.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_fatigue_analysis.ipynb
│
├── outputs/
│   ├── custom_cnn_metrics.csv
│   ├── mobilenetv2_metrics.csv
│   ├── model_comparison.csv
│   ├── fatigue_predictions.csv
│   └── fatigue_progression.csv
│
├── app.py
├── requirements.txt
└── README.md