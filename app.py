import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/mobilenetv2_final.keras"

CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]

FATIGUE_MAPPING = {
    "Closed": ("Severe Fatigue", 2),
    "yawn": ("Mild Fatigue", 1),
    "Open": ("Alert", 0),
    "no_yawn": ("Alert", 0),
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection")
st.write(
    "Upload an image to detect eye closure/yawning and estimate the fatigue level."
)

uploaded_file = st.file_uploader(
    "Upload a driver image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_array = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    fatigue_level, fatigue_score = FATIGUE_MAPPING[predicted_class]

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Detected State",
            predicted_class
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

    st.subheader("Fatigue Assessment")

    if fatigue_level == "Severe Fatigue":
        st.error("🔴 Severe Fatigue")
    elif fatigue_level == "Mild Fatigue":
        st.warning("🟠 Mild Fatigue")
    else:
        st.success("🟢 Alert")

    st.write(f"Fatigue Score: **{fatigue_score}**")

    st.subheader("Class Probabilities")

    for class_name, probability in zip(CLASS_NAMES, predictions[0]):
        st.write(
            f"{class_name}: {probability:.2%}"
        )