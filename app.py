import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Breast Cancer Image Classification", layout="centered")
st.title("Mammography ROI Classifier (ResNet50)")
st.write("Upload a pre-cropped region-of-interest (ROI) mammogram image for classification.")

# 1. Load trained model (cached so it only loads once)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("resnet50_breast_cancer.keras")

model = load_model()

# 2. Decision Threshold Adjustment
threshold = st.slider(
    "Decision Threshold (Lower = Higher Sensitivity/Recall for Malignancy)",
    min_value=0.1, max_value=0.9, value=0.3, step=0.05
)

# 3. Image Upload Interface
uploaded_file = st.file_uploader("Choose an ROI mammogram crop...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image to match training pipeline (224x224 + ResNet50 preprocess_input)
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Make Prediction
    if st.button("Run Inference"):
        prob = float(model.predict(img_batch)[0][0])
        is_malignant = prob >= threshold
        
        st.subheader("Prediction Results")
        st.write(f"**Malignancy Probability:** `{prob:.2%}`")
        
        if is_malignant:
            st.error(f"**Classification: MALIGNANT** (Score exceeds threshold of {threshold:.2f})")
        else:
            st.success(f"**Classification: BENIGN** (Score below threshold of {threshold:.2f})")