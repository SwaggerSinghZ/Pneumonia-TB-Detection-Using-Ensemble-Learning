import streamlit as st
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(page_title="Chest X-ray Diagnosis AI", layout="wide")

st.title("AI-Based Chest X-ray Diagnosis System")

st.markdown("""
### Deep Learning Detection of Pneumonia & Tuberculosis

This system uses **5 deep learning models with ensemble learning** to classify chest X-rays into:

- **NORMAL**
- **PNEUMONIA**
- **TUBERCULOSIS**

The system provides:

- Model predictions  
- Confidence scores  
- Model agreement analysis  
- Explainable AI using **Grad-CAM**
""")


# ------------------------------------------------
# SIDEBAR MODEL SELECTION
# ------------------------------------------------

st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Prediction Mode",
    [
        "Ensemble (All Models)",
        "EfficientNetB0",
        "ResNet50",
        "DenseNet121",
        "InceptionV3",
        "MobileNetV2"
    ]
)


# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = "../Models"

MODEL_PATHS = {
    "EfficientNetB0": os.path.join(BASE_DIR, "EfficientNetB0/best_model.h5"),
    "ResNet50": os.path.join(BASE_DIR, "ResNet50/best_model.h5"),
    "DenseNet121": os.path.join(BASE_DIR, "DenseNet121/best_model.h5"),
    "InceptionV3": os.path.join(BASE_DIR, "InceptionV3/best_model.h5"),
    "MobileNetV2": os.path.join(BASE_DIR, "MobileNetV2/best_model.h5"),
}

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

@st.cache_resource
def load_models():
    models = {}

    for name, path in MODEL_PATHS.items():
        models[name] = load_model(path)

    return models


models = load_models()


# ------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------

def preprocess_image(img, target_size):

    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ------------------------------------------------
# ENSEMBLE PREDICTION
# ------------------------------------------------

def weighted_ensemble_predict(img):

    predictions = []
    confidences = []
    model_names = []

    for name, model in models.items():

        if name == "InceptionV3":
            input_img = preprocess_image(img, (299,299))
        else:
            input_img = preprocess_image(img, (224,224))

        probs = model.predict(input_img, verbose=0)[0]

        predictions.append(probs)
        confidences.append(np.max(probs))
        model_names.append(name)

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    weights = confidences / np.sum(confidences)

    weighted_probs = np.average(predictions, axis=0, weights=weights)

    final_class_idx = np.argmax(weighted_probs)
    final_class = CLASS_NAMES[final_class_idx]
    confidence = np.max(weighted_probs)

    best_model_index = np.argmax(confidences)
    best_model = model_names[best_model_index]

    return final_class, confidence, weighted_probs, predictions, best_model


# ------------------------------------------------
# UNCERTAINTY
# ------------------------------------------------

def calculate_uncertainty(predictions):

    variance = np.var(predictions, axis=0)
    uncertainty = np.mean(variance)

    return uncertainty


# ------------------------------------------------
# GRAD-CAM
# ------------------------------------------------

def generate_gradcam(model, img_array, class_index):

    gradcam = Gradcam(model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)

    score = CategoricalScore(class_index)

    cam = gradcam(score, img_array)

    heatmap = cam[0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap


def overlay_gradcam(original_img, heatmap):

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return overlay


# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Uploaded X-ray",
             use_container_width=True)

    # Spinner animation
    with st.spinner("Running AI diagnosis..."):

        # ------------------------------------------------
        # MODEL SELECTION LOGIC
        # ------------------------------------------------

        if model_choice == "Ensemble (All Models)":

            label, confidence, probs, raw_predictions, best_model = weighted_ensemble_predict(img)

            uncertainty = calculate_uncertainty(raw_predictions)

        else:

            model = models[model_choice]

            if model_choice == "InceptionV3":
                img_array = preprocess_image(img,(299,299))
            else:
                img_array = preprocess_image(img,(224,224))

            probs = model.predict(img_array)[0]

            label = CLASS_NAMES[np.argmax(probs)]
            confidence = np.max(probs)

            best_model = model_choice
            uncertainty = None


    # ------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------

    st.subheader("Prediction")

    st.write(f"Diagnosis: **{label}**")
    st.write(f"Confidence: **{round(confidence*100,2)}%**")
    st.write(f"Model Used: **{best_model}**")


    # ------------------------------------------------
    # PROBABILITY BAR CHART (NEW)
    # ------------------------------------------------

    st.subheader("Class Probabilities")

    prob_dict = {cls: float(p*100) for cls, p in zip(CLASS_NAMES, probs)}

    st.bar_chart(prob_dict)


    # ------------------------------------------------
    # UNCERTAINTY DISPLAY
    # ------------------------------------------------

    if uncertainty is not None:

        st.subheader("Model Agreement")

        st.write(f"Uncertainty Score: {round(uncertainty,4)}")

        if uncertainty > 0.02:
            st.warning("Models disagree — Expert review recommended")
        else:
            st.success("Models show strong agreement")


    # ------------------------------------------------
    # GRAD-CAM
    # ------------------------------------------------

    st.subheader("Explainable AI — Grad-CAM")

    model = models[best_model]

    if best_model == "InceptionV3":
        img_array = preprocess_image(img,(299,299))
    else:
        img_array = preprocess_image(img,(224,224))

    pred = model.predict(img_array)
    class_index = np.argmax(pred)

    heatmap = generate_gradcam(model, img_array, class_index)

    overlay = overlay_gradcam(img, heatmap)

    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption="Grad-CAM Explanation",
             use_container_width=True)


# ------------------------------------------------
# MEDICAL DISCLAIMER
# ------------------------------------------------

st.markdown("---")

st.caption(
"This AI system is intended for research and educational purposes only. "
"It should not be used as a substitute for professional medical diagnosis."
)