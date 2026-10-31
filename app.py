from pathlib import Path
from collections import deque
import threading
import time

import av

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

EYE_MODEL_PATH = (
    ROOT / "models" / "eye_state" / "mobilenetv2_eye_finetuned.keras"
)
MOUTH_MODEL_PATH = (
    ROOT / "models" / "mouth_state" / "mobilenetv2_mouth_finetuned.keras"
)
LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"

IMG_SIZE = (224, 224)

# Existing eye-model decision threshold.
EYE_CLOSED_THRESHOLD = 0.75

# Provisional operating threshold selected from the threshold comparison.
YAWN_THRESHOLD = 0.30

RIGHT_EYE = [33, 160, 158, 133, 153, 144, 145, 163, 7, 246]
LEFT_EYE = [362, 385, 387, 263, 373, 380, 374, 386, 249, 466]


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_models():
    """Load the trained eye and mouth models once per Streamlit session."""
    if not EYE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Eye model not found:\n{EYE_MODEL_PATH}"
        )

    if not MOUTH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Mouth model not found:\n{MOUTH_MODEL_PATH}"
        )

    eye_model = tf.keras.models.load_model(
        str(EYE_MODEL_PATH),
        compile=False,
    )

    mouth_model = tf.keras.models.load_model(
        str(MOUTH_MODEL_PATH),
        compile=False,
    )

    return eye_model, mouth_model


# --------------------------------------------------
# MODEL-SPECIFIC PREPROCESSING
# --------------------------------------------------

def preprocess_eye(image_rgb):
    """
    Preprocessing retained for the existing eye-state model.
    """
    image = tf.convert_to_tensor(image_rgb, dtype=tf.float32)

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method="bilinear",
        antialias=True,
    )

    image = preprocess_input(image)

    return tf.expand_dims(image, axis=0)


def preprocess_mouth(image_rgb):
    """
    Match the mouth-model training notebook:
    - use the complete uploaded image (not a face-only crop)
    - TensorFlow bilinear resize with its default antialias setting
    - MobileNetV2 preprocess_input
    """
    image = tf.convert_to_tensor(image_rgb, dtype=tf.float32)

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method="bilinear",
    )

    image = preprocess_input(image)

    return tf.expand_dims(image, axis=0)


# --------------------------------------------------
# FACE AND EYE HELPERS
# --------------------------------------------------

def get_eye_crop(image_rgb, landmarks, indices):
    """Extract a padded eye crop using MediaPipe eye contour landmarks."""
    height, width = image_rgb.shape[:2]

    points = np.array(
        [
            [landmarks[index].x * width, landmarks[index].y * height]
            for index in indices
        ],
        dtype=np.float32,
    )

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    box_width = max(1.0, x_max - x_min)
    box_height = max(1.0, y_max - y_min)

    # Preserve the existing eye-crop padding.
    pad_x = box_width * (2.2 - 1.0) / 2
    pad_y = box_height * (3.0 - 1.0) / 2

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(width, int(x_max + pad_x))
    y2 = min(height, int(y_max + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image_rgb[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


def get_face_box(image_shape, landmarks):
    """Get the bounding box around one MediaPipe face."""
    height, width = image_shape[:2]

    x_coords = [point.x for point in landmarks]
    y_coords = [point.y for point in landmarks]

    x1 = max(0, int(min(x_coords) * width))
    y1 = max(0, int(min(y_coords) * height))
    x2 = min(width, int(max(x_coords) * width))
    y2 = min(height, int(max(y_coords) * height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def predict_eye_state(image_rgb, landmarks, eye_model):
    """Predict left and right eye state for a detected face."""
    left_crop, left_box = get_eye_crop(
        image_rgb, landmarks, LEFT_EYE
    )
    right_crop, right_box = get_eye_crop(
        image_rgb, landmarks, RIGHT_EYE
    )

    if left_crop is None or right_crop is None:
        return None

    # The eye model output represents probability of OPEN eyes.
    left_open_probability = float(
        np.asarray(
            eye_model.predict(
                preprocess_eye(left_crop),
                verbose=0,
            )
        ).reshape(-1)[0]
    )

    right_open_probability = float(
        np.asarray(
            eye_model.predict(
                preprocess_eye(right_crop),
                verbose=0,
            )
        ).reshape(-1)[0]
    )

    left_open_probability = float(
        np.clip(left_open_probability, 0.0, 1.0)
    )
    right_open_probability = float(
        np.clip(right_open_probability, 0.0, 1.0)
    )

    left_closed_probability = 1.0 - left_open_probability
    right_closed_probability = 1.0 - right_open_probability

    left_closed = (
        left_closed_probability >= EYE_CLOSED_THRESHOLD
    )
    right_closed = (
        right_closed_probability >= EYE_CLOSED_THRESHOLD
    )

    return {
        "left_box": left_box,
        "right_box": right_box,
        "left_open_probability": left_open_probability,
        "right_open_probability": right_open_probability,
        "left_closed": bool(left_closed),
        "right_closed": bool(right_closed),
        "both_closed": bool(left_closed and right_closed),
    }


# --------------------------------------------------
# STILL-IMAGE ANALYSIS
# --------------------------------------------------

def analyze_image(image_rgb, eye_model, mouth_model):
    """
    Analyze one uploaded driver image.

    Important:
    The mouth classifier receives the complete uploaded image because
    the training notebook trained it on complete dataset images.
    The eye classifier continues to receive landmark-based eye crops.
    """
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            "MediaPipe Face Landmarker model was not found:\n"
            f"{LANDMARKER_PATH}"
        )

    image_rgb = np.ascontiguousarray(image_rgb)
    annotated = image_rgb.copy()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARKER_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:
        detection = landmarker.detect(mp_image)

    if not detection.face_landmarks:
        return annotated, [], None

    # Mouth model: predict from the complete uploaded image, matching
    # the full-image training pipeline in the mouth notebook.
    yawn_probability = float(
        np.asarray(
            mouth_model.predict(
                preprocess_mouth(image_rgb),
                verbose=0,
            )
        ).reshape(-1)[0]
    )
    yawn_probability = float(
        np.clip(yawn_probability, 0.0, 1.0)
    )
    yawning = yawn_probability >= YAWN_THRESHOLD

    results = []

    for face_number, landmarks in enumerate(
        detection.face_landmarks,
        start=1,
    ):
        face_box = get_face_box(image_rgb.shape, landmarks)

        if face_box is None:
            continue

        eye_prediction = predict_eye_state(
            image_rgb,
            landmarks,
            eye_model,
        )

        if eye_prediction is None:
            continue

        both_closed = eye_prediction["both_closed"]

        if both_closed and yawning:
            status = "Eyes possibly closed + possible yawn"
            color = (255, 80, 80)
        elif both_closed:
            status = "Both eyes possibly closed"
            color = (255, 80, 80)
        elif yawning:
            status = "Possible yawn"
            color = (255, 180, 0)
        else:
            status = "No configured threshold crossed"
            color = (60, 200, 90)

        x1, y1, x2, y2 = face_box

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            annotated,
            status,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA,
        )

        for eye_box in (
            eye_prediction["left_box"],
            eye_prediction["right_box"],
        ):
            if eye_box is not None:
                ex1, ey1, ex2, ey2 = eye_box
                cv2.rectangle(
                    annotated,
                    (ex1, ey1),
                    (ex2, ey2),
                    (0, 200, 255),
                    2,
                )

        results.append(
            {
                "Face": face_number,
                "Left-eye open probability": round(
                    eye_prediction["left_open_probability"], 3
                ),
                "Right-eye open probability": round(
                    eye_prediction["right_open_probability"], 3
                ),
                "Left eye possibly closed": eye_prediction["left_closed"],
                "Right eye possibly closed": eye_prediction["right_closed"],
                "Both eyes possibly closed": both_closed,
                "Yawn probability": round(yawn_probability, 3),
                "Yawn threshold": YAWN_THRESHOLD,
                "Yawn detected": bool(yawning),
                "Snapshot observation": status,
            }
        )

    return annotated, results, yawn_probability


# --------------------------------------------------
# LIVE WEBCAM PROCESSOR
# --------------------------------------------------

class LiveVideoProcessor(VideoProcessorBase):
    """Analyze sampled browser webcam frames with the existing models."""

    def __init__(self, eye_model, mouth_model):
        self.eye_model = eye_model
        self.mouth_model = mouth_model
        self.lock = threading.Lock()
        self.last_sample_time = 0.0
        self.last_timestamp_ms = 0
        self.last_annotated = None
        self.last_message = "Starting camera..."
        self.eye_history = deque(maxlen=20)  # 10 seconds at 2 samples/second
        self.yawn_events = deque(maxlen=20)  # timestamps of yawn onsets
        self.previous_yawning = False

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(LANDMARKER_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )

    def recv(self, frame):
        image_rgb = frame.to_ndarray(format="rgb24")
        now = time.monotonic()

        # Run inference at approximately 2 frames per second to reduce load.
        if now - self.last_sample_time >= 0.5:
            self.last_sample_time = now
            annotated = image_rgb.copy()
            try:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(image_rgb),
                )
                timestamp_ms = time.monotonic_ns() // 1_000_000
                timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
                self.last_timestamp_ms = timestamp_ms
                detection = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                if not detection.face_landmarks:
                    message = "No face detected - face the camera"
                    cv2.putText(annotated, message, (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 80, 80), 2, cv2.LINE_AA)
                    self.previous_yawning = False
                else:
                    landmarks = detection.face_landmarks[0]
                    face_box = get_face_box(image_rgb.shape, landmarks)
                    eye_prediction = predict_eye_state(
                        image_rgb, landmarks, self.eye_model
                    )

                    # Mouth model is evaluated on the complete frame, just
                    # as the training notebook used complete images.
                    yawn_probability = float(np.asarray(
                        self.mouth_model.predict(
                            preprocess_mouth(image_rgb), verbose=0
                        )
                    ).reshape(-1)[0])
                    yawn_probability = float(np.clip(yawn_probability, 0.0, 1.0))
                    yawning = yawn_probability >= YAWN_THRESHOLD

                    if eye_prediction is not None:
                        both_closed = eye_prediction["both_closed"]
                        self.eye_history.append(bool(both_closed))
                        if yawning and not self.previous_yawning:
                            self.yawn_events.append(now)
                        self.previous_yawning = yawning

                        # Keep only yawn events from the last 60 seconds.
                        while self.yawn_events and now - self.yawn_events[0] > 60:
                            self.yawn_events.popleft()

                        if face_box:
                            x1, y1, x2, y2 = face_box
                            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                          (255, 180, 0) if both_closed or yawning
                                          else (60, 200, 90), 2)

                        lines = [
                            "Eyes: " + ("both possibly closed" if both_closed else "not both closed"),
                            f"Yawn probability: {yawn_probability:.1%}",
                            f"Closed-eye samples: {sum(self.eye_history)}/{len(self.eye_history)}",
                            f"Yawn onsets (last 60s): {len(self.yawn_events)}",
                        ]
                        for i, line in enumerate(lines):
                            cv2.putText(annotated, line, (15, 30 + i * 27),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                                        (255, 255, 255), 2, cv2.LINE_AA)

                        for eye_box in (eye_prediction["left_box"], eye_prediction["right_box"]):
                            if eye_box:
                                ex1, ey1, ex2, ey2 = eye_box
                                cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2),
                                              (0, 200, 255), 1)
                        message = "Live analysis running"
                    else:
                        message = "Face detected; adjust position for eye analysis"
                        cv2.putText(annotated, message, (15, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 180, 0), 2, cv2.LINE_AA)

                with self.lock:
                    self.last_annotated = annotated
                    self.last_message = message if 'message' in locals() else "Analyzing..."
            except Exception as exc:
                cv2.putText(annotated, f"Processing error: {str(exc)[:55]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 80, 80), 2, cv2.LINE_AA)
                with self.lock:
                    self.last_annotated = annotated
                    self.last_message = "Frame processing error"

        with self.lock:
            output = self.last_annotated.copy() if self.last_annotated is not None else image_rgb
        return av.VideoFrame.from_ndarray(output, format="rgb24")

    def __del__(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Image Analysis", "Live Facecam", "About"],
)

st.sidebar.caption(
    "Driver Drowsiness Detection • Image-based prototype"
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":
    st.title("🚗 Driver Drowsiness Detection")

    st.write(
        "A prototype that uses trained MobileNetV2 models to estimate eye "
        "state and yawning from an uploaded image or live webcam frames."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Eye-state model", "MobileNetV2")

    with col2:
        st.metric("Mouth-state model", "MobileNetV2")

    with col3:
        st.metric("Yawn threshold", "0.30")

    st.subheader("Available features")

    st.markdown(
        """
        - Upload a JPG, JPEG or PNG image.
        - Detect a face using MediaPipe FaceLandmarker.
        - Estimate left-eye and right-eye state.
        - Estimate yawn probability.
        - Display annotated image and a prediction report.
        - Download the prediction report as a CSV file.
        """
    )

    st.info(
        "Image Analysis evaluates one still image. Live Facecam samples "
        "webcam frames over time and displays rolling eye-closure and yawn "
        "observations. These are experimental indicators, not a validated "
        "fatigue or driving-safety assessment."
    )

    st.warning(
        "This is an experimental academic prototype, not a medical or "
        "driver-safety device. Do not use its output to decide whether "
        "someone is safe to drive."
    )


# --------------------------------------------------
# IMAGE ANALYSIS PAGE
# --------------------------------------------------

elif page == "Image Analysis":
    st.title("Image Analysis")

    st.write(
        "Upload one clear, front-facing driver image with the face visible."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_rgb = np.asarray(image)

            st.image(
                image_rgb,
                caption="Uploaded image",
                width="stretch",
            )

            if st.button(
                "Analyze Image",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner(
                    "Loading models and analyzing the uploaded image..."
                ):
                    eye_model, mouth_model = load_models()

                    annotated, results, yawn_probability = analyze_image(
                        image_rgb,
                        eye_model,
                        mouth_model,
                    )

                st.subheader("Analysis result")

                if not results:
                    st.warning(
                        "No usable face or eye landmarks were detected. "
                        "Try a clear, front-facing image with good lighting."
                    )
                else:
                    st.image(
                        annotated,
                        caption="Face detection and prediction annotations",
                        width="stretch",
                    )

                    result_df = pd.DataFrame(results)

                    st.subheader("Prediction report")
                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    first_result = results[0]

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric(
                            "Left eye",
                            (
                                "Possibly closed"
                                if first_result["Left eye possibly closed"]
                                else "Open"
                            ),
                        )

                    with metric_col2:
                        st.metric(
                            "Right eye",
                            (
                                "Possibly closed"
                                if first_result["Right eye possibly closed"]
                                else "Open"
                            ),
                        )

                    with metric_col3:
                        st.metric(
                            "Yawn probability",
                            f"{yawn_probability:.1%}",
                        )

                    if first_result["Yawn detected"]:
                        st.warning(
                            "Possible yawn detected. The model's yawn "
                            f"probability is {yawn_probability:.1%}, which "
                            f"meets the configured threshold of "
                            f"{YAWN_THRESHOLD:.2f}."
                        )
                    else:
                        st.success(
                            "The yawn threshold was not crossed. "
                            f"Yawn probability: {yawn_probability:.1%}."
                        )

                    if first_result["Both eyes possibly closed"]:
                        st.warning(
                            "Both eyes are possibly closed in this image. "
                            "A still image cannot establish how long the "
                            "eyes have remained closed."
                        )

                    csv_data = result_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download prediction report (CSV)",
                        data=csv_data,
                        file_name="driver_drowsiness_prediction_report.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.caption(
                        "The yawn classifier uses the complete uploaded "
                        "image, matching the input format in its training "
                        "notebook. The eye classifier uses separate "
                        "landmark-based eye crops."
                    )

        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")
            st.exception(exc)


# --------------------------------------------------
# LIVE FACECAM PAGE
# --------------------------------------------------

elif page == "Live Facecam":
    st.title("Live Driver Drowsiness Monitoring")
    st.write(
        "Allow browser camera access and position your face clearly in "
        "front of the camera. The app samples frames at approximately 2 Hz."
    )
    st.info(
        "The live view reports eye-closure samples and yawn onsets. It does "
        "not diagnose fatigue or determine whether a person is safe to drive."
    )

    if not LANDMARKER_PATH.exists():
        st.error(f"Face Landmarker model not found: {LANDMARKER_PATH}")
    else:
        try:
            eye_model, mouth_model = load_models()
            webrtc_streamer(
                key="driver-drowsiness-live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: LiveVideoProcessor(
                    eye_model, mouth_model
                ),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640},
                              "height": {"ideal": 480},
                              "facingMode": "user"},
                    "audio": False,
                },
                async_processing=True,
            )
            st.caption(
                "If camera access is blocked, allow it in your browser site "
                "settings. Webcam access generally requires localhost or HTTPS."
            )
        except Exception as exc:
            st.error(f"Unable to start live webcam analysis: {exc}")
            st.exception(exc)


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":
    st.title("About This Project")

    st.markdown(
        """
        This prototype uses the following components:

        - **MobileNetV2 eye-state model:** estimates the probability that
          each eye is open.
        - **MobileNetV2 mouth-state model:** estimates the probability
          that the uploaded image belongs to the yawning class.
        - **MediaPipe FaceLandmarker:** detects facial landmarks and
          supports eye-region extraction.
        - **OpenCV:** draws face and eye-region annotations.
        - **TensorFlow:** loads the trained models and preprocesses images.
        - **Streamlit:** provides the web application interface.
        - **Streamlit WebRTC:** streams browser webcam frames to the app.

        ### Model input handling

        - The eye-state model receives separate eye-region crops.
        - The mouth-state model receives the complete uploaded image,
          matching the full-image preprocessing used in its training notebook.
        - Both models use 224 × 224 input images and MobileNetV2
          preprocessing.

        ### Yawn threshold

        The current yawn threshold is **0.30**. It is a provisional
        operating threshold selected from the threshold comparison on
        the available test set. It should be validated on a separate
        validation set before making final performance claims.

        ### Limitation

        Image Analysis evaluates one still image. Live Facecam samples
        frames over time and displays experimental rolling observations;
        these are not a validated fatigue or driving-safety assessment.
        """
    )

    st.warning(
        "This application is an experimental academic prototype. Model "
        "performance on prepared test images does not guarantee reliable "
        "performance across different people, cameras, lighting conditions "
        "or camera angles. It is not a substitute for a validated driver "
        "monitoring or safety system."
    )
