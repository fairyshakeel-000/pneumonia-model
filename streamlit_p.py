import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------
# 🧭 Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 Pneumonia Detection using CNN")
st.write("Upload a chest X-ray image to check if it shows signs of Pneumonia or Normal lungs.")

# ----------------------------------------------------------
# 🧩 Load Model
# ----------------------------------------------------------
MODEL_PATH = "pneumonia_cnn_fixed.h5"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ----------------------------------------------------------
# 🖼️ Image Upload
# ----------------------------------------------------------
uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ----------------------------------------------------------
    # 🧠 Preprocess Image
    # ----------------------------------------------------------
    img = image.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)

    # ----------------------------------------------------------
    # 🔍 Make Prediction
    # ----------------------------------------------------------
    prediction = model.predict(img_array)[0][0]

    st.subheader("🩻 Prediction Result:")
    if prediction > 0.5:
        st.error(f"⚠️ Pneumonia Detected! (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ Normal Lungs (Confidence: {1 - prediction:.2f})")

    # ----------------------------------------------------------
    # 📊 Confidence Gauge
    # ----------------------------------------------------------
    st.write("---")
    st.write("### Prediction Confidence")
    st.progress(float(prediction) if prediction < 1 else 1.0)
