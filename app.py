import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
import numpy as np


# Welcome page state
# ---------------------------------------------------------
if "started" not in st.session_state:
    st.session_state.started = False
 
def start_app():
    st.session_state.started = True
 
# Welcome screen
if not st.session_state.started:
    st.markdown("<h1 style='text-align: center;'>Welcome!</h1>", unsafe_allow_html=True)
    st.write("")
    st.write(
        "This is an AI-assisted healthcare platform that helps patients "
        "better understand their mammogram results, using three different machine "
        "learning models to offer a range of perspectives."
    )
    st.write("")
    st.caption(
        "AI models can make mistakes and should not be relied on as a diagnosis. "
        "Always consult your doctor to discuss results."
    )
    st.write("")
    st.button("Get Started", on_click=start_app)
    st.stop()
 
#Page configuration
st.set_page_config(page_title="Breast Cancer Image Classification", layout="centered")
st.title("Breast Cancer Image Classifier")
st.write("Upload a pre-cropped region-of-interest (ROI) mammogram image for classification.")

IMG_SIZE = 224

# 1. Load all models once, cached
@st.cache_resource

def load_models():
    # Model 1: fine-tuned ResNet50 (this IS the "CNN" from Colab)
    cnn_model = tf.keras.models.load_model("models/resnet50_breast_cancer.keras")

    # Feature extractor for the SVM: same as Colab ->
    # model.layers[-3].output = the GlobalAveragePooling2D layer,
    # i.e. everything up to (but not including) Dropout + final Dense
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-3].output
    )

    # Model 2: SVM + its scaler (fit on CNN-extracted features)
    svm_model = joblib.load("models/svm_model.pkl")
    svm_scaler = joblib.load("models/svm_scaler.pkl")

    # Model 3: fine-tuned ViT
    vit_model = ViTForImageClassification.from_pretrained("models/vit_model")
    vit_processor = ViTImageProcessor.from_pretrained("models/vit_model")
    vit_model.eval()

    return cnn_model, feature_extractor, svm_model, svm_scaler, vit_model, vit_processor

cnn_model, feature_extractor, svm_model, svm_scaler, vit_model, vit_processor = load_models()


# 2. Decision threshold (applies to CNN's sigmoid output)
threshold = st.slider(
    "Decision Threshold (Lower = Higher Sensitivity/Recall for Malignancy)",
    min_value=0.1, max_value=0.9, value=0.3, step=0.05
)


# 3. Image Upload Interface
uploaded_file = st.file_uploader("Choose an ROI mammogram crop...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Inference"):
        results = {}

        # Shared preprocessing for the ResNet50-based CNN and the SVM
        # (SVM features come from this same CNN, so same preprocessing applies)
        img_resized = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        img_preprocessed = preprocess_input(img_resized.copy())
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        #  Model 1: ResNet50-based CNN (sigmoid) 
        prob = float(cnn_model.predict(img_batch)[0][0])
        label = "MALIGNANT" if prob >= threshold else "BENIGN"
        confidence = prob if prob >= threshold else 1 - prob
        results["CNN (ResNet50)"] = (label, confidence)

        # Model 2: SVM on CNN-extracted features 
        cnn_features = feature_extractor.predict(img_batch)
        cnn_features_scaled = svm_scaler.transform(cnn_features)
        proba = svm_model.predict_proba(cnn_features_scaled)[0]
        idx = int(np.argmax(proba))
        label = "MALIGNANT" if idx == 1 else "BENIGN"
        results["SVM"] = (label, float(proba[idx]))

        # Model 3: ViT (softmax) 
        vit_inputs = vit_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = vit_model(**vit_inputs).logits
        vit_probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(vit_probs).item())
        label = "MALIGNANT" if idx == 1 else "BENIGN"
        results["ViT"] = (label, float(vit_probs[idx].item()))

        
        # 4. Display results: model name + confidence only
        st.subheader("Prediction Results")
        for model_name, (label, confidence) in results.items():
            col1, col2, col3 = st.columns([2, 2, 2])
            col1.write(f"**{model_name}**")
            col2.write(f"{confidence:.2%}")
            if label == "MALIGNANT":
                col3.error(label)
            else:
                col3.success(label)