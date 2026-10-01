# Driver Drowsiness Detection Using Eye Closure and Yawning Analysis with Deep Learning

A deep learning-based driver drowsiness detection application that analyzes **eye closure** and **yawning behavior** from driver images to identify different levels of fatigue.

The project uses **MediaPipe facial landmarks** for facial feature localization, **MobileNetV2 transfer learning** for eye and mouth state classification, and a **confidence-aware decision fusion** approach to determine the driver's fatigue state.

The project also includes temporal analysis for sequences of observations, allowing repeated eye-closure and yawning observations to be evaluated over time.

---

## Project Overview

Driver drowsiness is a major safety concern, particularly during long-distance and monotonous driving.

This project focuses on two observable visual indicators of fatigue:

- **Eye state**
  - Open
  - Closed

- **Mouth state**
  - No Yawn
  - Yawn

The application extracts the relevant facial regions and uses separate deep learning models to classify the eye and mouth states.

The predictions are then combined using a confidence-aware decision mechanism.

### Overall Pipeline

```text
Driver Image
      ↓
Face Detection / Facial Landmarks
      ↓
 ┌───────────────┬────────────────┐
 ↓               ↓
Eye ROI        Mouth ROI
 ↓               ↓
Eye Model      Mouth Model
 ↓               ↓
Open/Closed    Yawn/No Yawn
 └───────────────┴────────────────┘
              ↓
     Confidence-Aware Fusion
              ↓
      Fatigue Classification
              ↓
 Alert / Mild Fatigue /
 Severe Fatigue / Uncertain