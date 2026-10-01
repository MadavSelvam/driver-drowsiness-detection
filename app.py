
import json
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EYE_MODEL_PATH = MODEL_DIR / "eye_mobilenetv2_best.keras"
MOUTH_MODEL_PATH = MODEL_DIR / "mouth_mobilenetv2_best.keras"
LANDMARK_MODEL_PATH = PROJECT_ROOT / "face_landmarker.task"

FUSION_CONFIG_PATH = OUTPUT_DIR / "final_fusion_config.json"
TEMPORAL_CONFIG_PATH = OUTPUT_DIR / "temporal_fatigue_config.json"

IMG_SIZE = (224, 224)
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

EYE_CLASSES = ["Closed", "Open"]

# MediaPipe Face Landmarker eye contours.
# Each eye is cropped separately because the training data for
# Closed/Open consists of eye-focused images.
LEFT_EYE_LANDMARKS = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_LANDMARKS = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]


# ============================================================
# 2. CONFIGURATION
# ============================================================

@st.cache_data
def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


fusion_config = load_json(FUSION_CONFIG_PATH)
temporal_config_file = load_json(TEMPORAL_CONFIG_PATH)

CONFIDENCE_THRESHOLD = float(
    fusion_config.get(
        "confidence_threshold",
        DEFAULT_CONFIDENCE_THRESHOLD,
    )
)

TEMPORAL_CONFIG = {
    "history_size": 15,
    "severe_closed_count": 3,
    "mild_yawn_count": 3,
    "severe_ratio": 0.50,
    "mild_ratio": 0.40,
    "severe_window": 5,
    "mild_window": 7,
    **temporal_config_file,
}


# ============================================================
# 3. LOAD MODELS + MEDIAPIPE
# ============================================================

@st.cache_resource
def load_models_and_landmarker():
    required = [
        EYE_MODEL_PATH,
        MOUTH_MODEL_PATH,
        LANDMARK_MODEL_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n\n"
            + "\n".join(missing)
        )

    eye_model = tf.keras.models.load_model(
        EYE_MODEL_PATH
    )

    mouth_model = tf.keras.models.load_model(
        MOUTH_MODEL_PATH
    )

    base_options = python.BaseOptions(
        model_asset_path=str(LANDMARK_MODEL_PATH)
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.50,
        min_face_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    return eye_model, mouth_model, landmarker


try:
    eye_model, mouth_model, face_landmarker = (
        load_models_and_landmarker()
    )
    models_loaded = True
except Exception as exc:
    models_loaded = False
    load_error = str(exc)


# ============================================================
# 4. TEMPORAL ANALYZER
# ============================================================

class TemporalFatigueAnalyzer:

    def __init__(self, config):
        self.config = config
        self.history = deque(
            maxlen=int(config["history_size"])
        )

    def reset(self):
        self.history.clear()

    def add_observation(self, observation):

        self.history.append(observation)

        severe_window = list(self.history)[
            -int(self.config["severe_window"]):
        ]

        mild_window = list(self.history)[
            -int(self.config["mild_window"]):
        ]

        closed_count = sum(
            1
            for item in severe_window
            if (
                item["eye_state"] == "Closed"
                and item["eye_confidence"]
                >= CONFIDENCE_THRESHOLD
            )
        )

        yawn_count = sum(
            1
            for item in mild_window
            if (
                item["mouth_state"] == "yawn"
                and item["mouth_confidence"]
                >= CONFIDENCE_THRESHOLD
            )
        )

        closed_ratio = (
            closed_count / len(severe_window)
            if severe_window
            else 0.0
        )

        yawn_ratio = (
            yawn_count / len(mild_window)
            if mild_window
            else 0.0
        )

        if (
            closed_count
            >= self.config["severe_closed_count"]
            and closed_ratio
            >= self.config["severe_ratio"]
        ):
            state = "Severe Fatigue"
            reason = (
                "Repeated confident closed-eye "
                "observations detected."
            )

        elif (
            yawn_count
            >= self.config["mild_yawn_count"]
            and yawn_ratio
            >= self.config["mild_ratio"]
        ):
            state = "Mild Fatigue"
            reason = (
                "Repeated confident yawn "
                "observations detected."
            )

        else:
            recent_states = [
                item["fatigue_state"]
                for item in list(self.history)[-3:]
            ]

            if (
                len(recent_states) == 3
                and all(
                    state == "Alert"
                    for state in recent_states
                )
            ):
                state = "Alert"
                reason = (
                    "Recent observations consistently "
                    "indicate alert state."
                )
            else:
                state = "Uncertain"
                reason = (
                    "Evidence is not persistent enough "
                    "for temporal fatigue escalation."
                )

        return {
            "fatigue_state": state,
            "reason": reason,
            "history_length": len(self.history),
            "closed_count": closed_count,
            "yawn_count": yawn_count,
            "closed_ratio": round(closed_ratio, 4),
            "yawn_ratio": round(yawn_ratio, 4),
        }


# ============================================================
# 5. SESSION STATE
# ============================================================

if "temporal_analyzer" not in st.session_state:
    st.session_state.temporal_analyzer = (
        TemporalFatigueAnalyzer(TEMPORAL_CONFIG)
    )

if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "history_records" not in st.session_state:
    st.session_state.history_records = []


def reset_analysis():
    st.session_state.temporal_analyzer.reset()
    st.session_state.frame_counter = 0
    st.session_state.last_result = None
    st.session_state.history_records = []


# ============================================================
# 6. FACE LANDMARK DETECTION
# ============================================================

def detect_landmarks(image_rgb):

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ============================================================
# 7. LANDMARK CROP HELPERS
# ============================================================

def landmark_box(
    landmarks,
    indices,
    image_width,
    image_height,
    pad_x_ratio,
    pad_y_ratio,
    min_pad_x=4,
    min_pad_y=4,
):
    points = []

    for index in indices:
        if index >= len(landmarks):
            continue

        point = landmarks[index]

        points.append(
            (
                point.x * image_width,
                point.y * image_height,
            )
        )

    if len(points) < 4:
        return None

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_y = float(points[:, 1].min())
    max_y = float(points[:, 1].max())

    width = max_x - min_x
    height = max_y - min_y

    if width < 3 or height < 3:
        return None

    pad_x = max(
        min_pad_x,
        width * pad_x_ratio,
    )

    pad_y = max(
        min_pad_y,
        height * pad_y_ratio,
    )

    x1 = max(
        0,
        int(round(min_x - pad_x)),
    )

    y1 = max(
        0,
        int(round(min_y - pad_y)),
    )

    x2 = min(
        image_width,
        int(round(max_x + pad_x)),
    )

    y2 = min(
        image_height,
        int(round(max_y + pad_y)),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def crop_box(image_rgb, box):

    if box is None:
        return None

    x1, y1, x2, y2 = box

    crop = image_rgb[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# 8. EYE ROI EXTRACTION
# ============================================================

def extract_eye_rois(image_rgb, landmarks):

    height, width = image_rgb.shape[:2]

    left_box = landmark_box(
        landmarks,
        LEFT_EYE_LANDMARKS,
        width,
        height,
        pad_x_ratio=0.45,
        pad_y_ratio=0.65,
        min_pad_x=6,
        min_pad_y=6,
    )

    right_box = landmark_box(
        landmarks,
        RIGHT_EYE_LANDMARKS,
        width,
        height,
        pad_x_ratio=0.45,
        pad_y_ratio=0.65,
        min_pad_x=6,
        min_pad_y=6,
    )

    left_crop = crop_box(
        image_rgb,
        left_box,
    )

    right_crop = crop_box(
        image_rgb,
        right_box,
    )

    return {
        "left_crop": left_crop,
        "right_crop": right_crop,
        "left_box": left_box,
        "right_box": right_box,
    }


# ============================================================
# 9. MOUTH ROI EXTRACTION
# ============================================================

def extract_mouth_roi(image_rgb, landmarks):

    height, width = image_rgb.shape[:2]

    mouth_box = landmark_box(
        landmarks,
        MOUTH_LANDMARKS,
        width,
        height,
        pad_x_ratio=0.35,
        pad_y_ratio=0.60,
        min_pad_x=8,
        min_pad_y=8,
    )

    mouth_crop = crop_box(
        image_rgb,
        mouth_box,
    )

    return mouth_crop, mouth_box


# ============================================================
# 10. MODEL PREPROCESSING
# ============================================================

def preprocess_image(image_rgb):

    image = cv2.resize(
        image_rgb,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(
        np.float32
    )

    return np.expand_dims(
        image,
        axis=0,
    )


# ============================================================
# 11. EYE PREDICTION
# ============================================================

def predict_single_eye(eye_crop):

    batch = preprocess_image(
        eye_crop
    )

    probabilities = np.asarray(
        eye_model.predict(
            batch,
            verbose=0,
        )
    ).reshape(-1)

    if probabilities.size != 2:
        raise ValueError(
            "Unexpected eye model output shape: "
            f"{probabilities.shape}"
        )

    index = int(
        np.argmax(probabilities)
    )

    return {
        "state": EYE_CLASSES[index],
        "confidence": float(
            probabilities[index]
        ),
        "probabilities": probabilities.tolist(),
    }


def combine_eye_predictions(
    left_result,
    right_result,
):
    """
    Combine two independently classified eye ROIs.

    Both eyes must agree confidently before the application
    reports a definitive Open/Closed eye state.

    Mixed or low-confidence observations become Uncertain
    instead of forcing the eye model to choose one state.
    """

    threshold = CONFIDENCE_THRESHOLD

    if left_result is None or right_result is None:
        return {
            "state": "Uncertain",
            "confidence": 0.0,
            "left": left_result,
            "right": right_result,
            "reason": "One or both eye ROIs could not be extracted.",
        }

    left_confident = (
        left_result["confidence"] >= threshold
    )

    right_confident = (
        right_result["confidence"] >= threshold
    )

    if (
        left_confident
        and right_confident
        and left_result["state"] == "Closed"
        and right_result["state"] == "Closed"
    ):
        state = "Closed"
        confidence = min(
            left_result["confidence"],
            right_result["confidence"],
        )
        reason = "Both eyes confidently classified as closed."

    elif (
        left_confident
        and right_confident
        and left_result["state"] == "Open"
        and right_result["state"] == "Open"
    ):
        state = "Open"
        confidence = min(
            left_result["confidence"],
            right_result["confidence"],
        )
        reason = "Both eyes confidently classified as open."

    else:
        state = "Uncertain"
        confidence = min(
            left_result["confidence"],
            right_result["confidence"],
        )
        reason = (
            "Eye predictions are mixed or do not both "
            "meet the confidence threshold."
        )

    return {
        "state": state,
        "confidence": float(confidence),
        "left": left_result,
        "right": right_result,
        "reason": reason,
    }


# ============================================================
# 12. MOUTH PREDICTION
# ============================================================

def predict_mouth(mouth_crop):

    batch = preprocess_image(
        mouth_crop
    )

    probability = float(
        np.asarray(
            mouth_model.predict(
                batch,
                verbose=0,
            )
        ).reshape(-1)[0]
    )

    if probability >= 0.50:
        state = "yawn"
        confidence = probability
    else:
        state = "no_yawn"
        confidence = 1.0 - probability

    return {
        "state": state,
        "confidence": confidence,
        "yawn_probability": probability,
    }


# ============================================================
# 13. CONFIDENCE-AWARE FUSION
# ============================================================

def confidence_aware_fusion(
    eye_result,
    mouth_result,
):

    eye_confident = (
        eye_result["state"] in ["Open", "Closed"]
        and eye_result["confidence"]
        >= CONFIDENCE_THRESHOLD
    )

    mouth_confident = (
        mouth_result is not None
        and mouth_result["confidence"]
        >= CONFIDENCE_THRESHOLD
    )

    if (
        eye_result["state"] == "Closed"
        and eye_confident
    ):
        return {
            "fatigue_state": "Severe Fatigue",
            "reason": (
                "Both eye ROIs were confidently "
                "classified as closed."
            ),
        }

    if (
        eye_result["state"] == "Open"
        and eye_confident
        and mouth_result is not None
        and mouth_result["state"] == "yawn"
        and mouth_confident
    ):
        return {
            "fatigue_state": "Mild Fatigue",
            "reason": (
                "Both eyes are confidently open and "
                "a confident yawn was detected."
            ),
        }

    if (
        eye_result["state"] == "Open"
        and eye_confident
        and mouth_result is not None
        and mouth_result["state"] == "no_yawn"
        and mouth_confident
    ):
        return {
            "fatigue_state": "Alert",
            "reason": (
                "Both eyes are confidently open and "
                "no yawn was detected."
            ),
        }

    return {
        "fatigue_state": "Uncertain",
        "reason": (
            "Eye or mouth evidence is mixed, unavailable, "
            "or below the confidence threshold."
        ),
    }


# ============================================================
# 14. PROCESS ONE FRAME
# ============================================================

def process_frame(
    image_rgb,
    frame_number,
    use_temporal=True,
):

    landmarks = detect_landmarks(
        image_rgb
    )

    annotated = image_rgb.copy()

    if landmarks is None:

        observation = {
            "frame": frame_number,
            "fatigue_state": "Uncertain",
            "eye_state": "Uncertain",
            "mouth_state": "Unavailable",
            "eye_confidence": 0.0,
            "mouth_confidence": 0.0,
        }

        if use_temporal:
            temporal = (
                st.session_state.temporal_analyzer
                .add_observation(
                    observation
                )
            )
        else:
            temporal = {
                "fatigue_state": "Uncertain",
                "reason": "No face detected.",
                "history_length": 0,
                "closed_count": 0,
                "yawn_count": 0,
                "closed_ratio": 0.0,
                "yawn_ratio": 0.0,
            }

        observation["temporal_state"] = (
            temporal["fatigue_state"]
        )

        return {
            "observation": observation,
            "temporal": temporal,
            "annotated": annotated,
            "left_eye_crop": None,
            "right_eye_crop": None,
            "mouth_crop": None,
            "face_detected": False,
        }

    eye_rois = extract_eye_rois(
        image_rgb,
        landmarks,
    )

    left_crop = eye_rois["left_crop"]
    right_crop = eye_rois["right_crop"]

    if left_crop is not None:
        left_eye_result = predict_single_eye(
            left_crop
        )
    else:
        left_eye_result = None

    if right_crop is not None:
        right_eye_result = predict_single_eye(
            right_crop
        )
    else:
        right_eye_result = None

    eye_result = combine_eye_predictions(
        left_eye_result,
        right_eye_result,
    )

    mouth_crop, mouth_box = extract_mouth_roi(
        image_rgb,
        landmarks,
    )

    if mouth_crop is not None:
        mouth_result = predict_mouth(
            mouth_crop
        )
    else:
        mouth_result = {
            "state": "Unavailable",
            "confidence": 0.0,
        }

    fusion_result = confidence_aware_fusion(
        eye_result,
        mouth_result,
    )

    observation = {
        "frame": frame_number,
        "fatigue_state": fusion_result[
            "fatigue_state"
        ],
        "eye_state": eye_result["state"],
        "mouth_state": mouth_result["state"],
        "eye_confidence": eye_result[
            "confidence"
        ],
        "mouth_confidence": mouth_result[
            "confidence"
        ],
    }

    if use_temporal:
        temporal = (
            st.session_state.temporal_analyzer
            .add_observation(
                observation
            )
        )
    else:
        temporal = {
            "fatigue_state": fusion_result[
                "fatigue_state"
            ],
            "reason": fusion_result[
                "reason"
            ],
            "history_length": 1,
            "closed_count": (
                1
                if eye_result["state"] == "Closed"
                else 0
            ),
            "yawn_count": (
                1
                if mouth_result["state"] == "yawn"
                else 0
            ),
            "closed_ratio": (
                1.0
                if eye_result["state"] == "Closed"
                else 0.0
            ),
            "yawn_ratio": (
                1.0
                if mouth_result["state"] == "yawn"
                else 0.0
            ),
        }

    observation["temporal_state"] = (
        temporal["fatigue_state"]
    )

    # Draw eye boxes.
    for box, label in [
        (
            eye_rois["left_box"],
            "Left Eye",
        ),
        (
            eye_rois["right_box"],
            "Right Eye",
        ),
    ]:

        if box is None:
            continue

        x1, y1, x2, y2 = box

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw mouth box.
    if mouth_box is not None:

        x1, y1, x2, y2 = mouth_box

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            annotated,
            "Mouth",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Overlay prediction.
    cv2.putText(
        annotated,
        (
            f"Eyes: {eye_result['state']} "
            f"({eye_result['confidence']:.2f})"
        ),
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        (
            f"Mouth: {mouth_result['state']} "
            f"({mouth_result['confidence']:.2f})"
        ),
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        (
            f"State: "
            f"{temporal['fatigue_state']}"
        ),
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return {
        "observation": observation,
        "temporal": temporal,
        "annotated": annotated,
        "left_eye_crop": left_crop,
        "right_eye_crop": right_crop,
        "mouth_crop": mouth_crop,
        "face_detected": True,
    }


# ============================================================
# 15. UI HELPERS
# ============================================================

def state_icon(state):

    return {
        "Alert": "🟢",
        "Mild Fatigue": "🟡",
        "Severe Fatigue": "🔴",
        "Uncertain": "⚪",
    }.get(
        state,
        "⚪",
    )


def display_result(
    result,
    show_temporal=True,
):

    observation = result["observation"]
    temporal = result["temporal"]

    final_state = (
        temporal["fatigue_state"]
    )

    st.markdown(
        f"# {state_icon(final_state)} "
        f"{final_state}"
    )

    st.caption(
        temporal["reason"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Eye State",
            observation["eye_state"],
            (
                f"{observation['eye_confidence']:.1%}"
            ),
        )

    with col2:
        st.metric(
            "Mouth State",
            observation["mouth_state"],
            (
                f"{observation['mouth_confidence']:.1%}"
            ),
        )

    with col3:
        st.metric(
            "Confidence Threshold",
            f"{CONFIDENCE_THRESHOLD:.0%}",
        )

    if show_temporal:

        t1, t2, t3, t4 = st.columns(4)

        with t1:
            st.metric(
                "History",
                temporal["history_length"],
            )

        with t2:
            st.metric(
                "Closed Eyes",
                temporal["closed_count"],
            )

        with t3:
            st.metric(
                "Yawns",
                temporal["yawn_count"],
            )

        with t4:
            st.metric(
                "Frame",
                observation["frame"],
            )


# ============================================================
# 16. APPLICATION HEADER
# ============================================================

st.title(
    "🚗 Driver Drowsiness Detection"
)

st.markdown(
    "### Eye Closure + Yawning Analysis with Deep Learning"
)

st.write(
    "The application uses separate eye and mouth regions, "
    "MobileNetV2 models, confidence-aware fusion, and "
    "temporal fatigue analysis."
)

if not models_loaded:

    st.error(
        "Models could not be loaded.\n\n"
        + load_error
    )

    st.stop()

st.success(
    "Models loaded: Eye MobileNetV2 + Mouth MobileNetV2"
)


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.metric(
        "Confidence Threshold",
        f"{CONFIDENCE_THRESHOLD:.0%}",
    )

    st.metric(
        "Temporal History",
        TEMPORAL_CONFIG["history_size"],
    )

    st.metric(
        "Severe Window",
        TEMPORAL_CONFIG["severe_window"],
    )

    st.metric(
        "Mild Window",
        TEMPORAL_CONFIG["mild_window"],
    )

    st.divider()

    st.write("**Selected Models**")
    st.write("👁️ Eye: MobileNetV2")
    st.write("👄 Mouth: MobileNetV2")

    st.divider()

    st.write("**Eye ROI**")
    st.caption(
        "Each eye is detected separately using "
        "MediaPipe facial landmarks before "
        "classification."
    )

    st.divider()

    if st.button(
        "🔄 Reset Analysis",
        use_container_width=True,
    ):
        reset_analysis()
        st.rerun()


# ============================================================
# 18. INPUT MODE
# ============================================================

mode = st.radio(
    "Select input",
    [
        "📷 Camera Snapshot",
        "🖼️ Image Upload",
    ],
    horizontal=True,
)


# ============================================================
# 19. CAMERA SNAPSHOT
# ============================================================

if mode == "📷 Camera Snapshot":

    st.subheader(
        "Camera Snapshot"
    )

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_rgb = np.asarray(
            image
        )

        st.session_state.frame_counter += 1

        result = process_frame(
            image_rgb,
            st.session_state.frame_counter,
            use_temporal=True,
        )

        st.session_state.last_result = result

        st.session_state.history_records.append(
            result["observation"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                result["annotated"],
                caption="Processed frame",
                use_container_width=True,
            )

        with col2:

            if result["left_eye_crop"] is not None:
                st.image(
                    result["left_eye_crop"],
                    caption="Left eye ROI",
                    use_container_width=True,
                )

            if result["right_eye_crop"] is not None:
                st.image(
                    result["right_eye_crop"],
                    caption="Right eye ROI",
                    use_container_width=True,
                )

            if result["mouth_crop"] is not None:
                st.image(
                    result["mouth_crop"],
                    caption="Mouth ROI",
                    use_container_width=True,
                )

        display_result(
            result,
            show_temporal=True,
        )


# ============================================================
# 20. IMAGE UPLOAD
# ============================================================

elif mode == "🖼️ Image Upload":

    st.subheader(
        "Single Image Analysis"
    )

    st.info(
        "Single images use direct eye + mouth fusion. "
        "Temporal escalation is reserved for sequences."
    )

    uploaded_image = st.file_uploader(
        "Upload a driver image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        image_rgb = np.asarray(
            image
        )

        if st.button(
            "Analyze Image",
            type="primary",
        ):

            # Single-image analysis should not carry over
            # history from a previous camera session.
            st.session_state.temporal_analyzer.reset()
            st.session_state.history_records = []

            st.session_state.frame_counter += 1

            result = process_frame(
                image_rgb,
                st.session_state.frame_counter,
                use_temporal=False,
            )

            st.session_state.last_result = result

            st.session_state.history_records = [
                result["observation"]
            ]

        if st.session_state.last_result is not None:

            result = (
                st.session_state.last_result
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    result["annotated"],
                    caption="Processed frame",
                    use_container_width=True,
                )

            with col2:

                eye_col1, eye_col2 = st.columns(2)

                with eye_col1:
                    if (
                        result["left_eye_crop"]
                        is not None
                    ):
                        st.image(
                            result["left_eye_crop"],
                            caption="Left eye ROI",
                            use_container_width=True,
                        )

                with eye_col2:
                    if (
                        result["right_eye_crop"]
                        is not None
                    ):
                        st.image(
                            result["right_eye_crop"],
                            caption="Right eye ROI",
                            use_container_width=True,
                        )

                if result["mouth_crop"] is not None:
                    st.image(
                        result["mouth_crop"],
                        caption="Mouth ROI",
                        use_container_width=True,
                    )

            display_result(
                result,
                show_temporal=False,
            )


# ============================================================
# 21. TEMPORAL HISTORY
# ============================================================

if (
    mode != "🖼️ Image Upload"
    and st.session_state.history_records
):

    st.divider()

    st.subheader(
        "📈 Fatigue Progression"
    )

    history = (
        st.session_state.history_records
    )

    state_values = {
        "Alert": 0,
        "Mild Fatigue": 1,
        "Severe Fatigue": 2,
        "Uncertain": np.nan,
    }

    chart_df = pd.DataFrame(
        {
            "Frame": [
                item["frame"]
                for item in history
            ],
            "Fatigue Level": [
                state_values.get(
                    item.get(
                        "temporal_state",
                        item["fatigue_state"],
                    ),
                    np.nan,
                )
                for item in history
            ],
        }
    )

    st.line_chart(
        chart_df,
        x="Frame",
        y="Fatigue Level",
    )

    st.caption(
        "0 = Alert, 1 = Mild Fatigue, "
        "2 = Severe Fatigue. Uncertain observations "
        "are not forced into a fatigue level."
    )

    with st.expander(
        "View frame-level observations"
    ):
        st.dataframe(
            history,
            use_container_width=True,
        )


# ============================================================
# 22. FOOTER
# ============================================================

st.divider()

st.caption(
    "Driver Drowsiness Detection | "
    "Eye ROI + Mouth ROI | "
    "MobileNetV2 + MediaPipe Face Landmarks + "
    "Confidence-Aware Fusion + Temporal Analysis"
)
