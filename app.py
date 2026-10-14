
import os
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Retina AI Dashboard",
    page_icon="👁️",
    layout="wide",
)

st.markdown("""
<style>
.main { background-color: #f5f8fc; }
h1, h2, h3 { color: #17365d; }
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce5f0;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# Required display order
MODEL_FILES = {
    "Retinal Age": ("retina_age_model.pth", "regression"),
    "Retinal Sex": ("retina_sex_model_v2.pth", "classification"),
    "Retinal Disease": ("retina_disease_model.pth", "multilabel"),
    "CIMT Estimate": ("cimt_thickness_model.pth", "regression"),
}

DISEASE_LABELS = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other",
]

SEX_CLASS_LABELS = ["Female", "Male"]

# Temporary threshold; validate with training data before clinical use.
DIABETES_THRESHOLD = 0.50

# Additional model filenames expected if you add these models later.
ADDITIONAL_MODELS = {
    "CAD / Heart Disease": "cad_heart_disease_model.pth",
    "Risk Prediction": "risk_prediction_model.pth",
    "CAP Grade": "cap_grade_model.pth",
}


# =====================================================
# IMAGE TRANSFORM
# =====================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =====================================================
# MODEL LOADING
# =====================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
        ):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]

        if checkpoint and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise ValueError("Unsupported checkpoint format.")


def clean_state_dict(state):
    cleaned = {}

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        while key.startswith(("module.", "model.", "net.")):
            key = key.split(".", 1)[1]

        cleaned[key] = value

    return cleaned


def infer_output_size(state, default):
    for key in (
        "classifier.1.weight",
        "classifier.weight",
        "fc.weight",
    ):
        if key in state and state[key].ndim == 2:
            return int(state[key].shape[0])

    return default


@st.cache_resource(show_spinner=False)
def load_model(path, task):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file missing: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = clean_state_dict(extract_state_dict(checkpoint))

        default_outputs = 8 if task == "multilabel" else 1
        output_size = infer_output_size(state, default_outputs)

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            output_size,
        )

        model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()
    return model


# =====================================================
# PREDICTION HELPERS
# =====================================================

def run_prediction(model, image, task):
    tensor = TRANSFORM(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        elif "output" in output:
            output = output["output"]
        else:
            output = next(iter(output.values()))

    values = output.detach().float().cpu().reshape(-1).numpy()

    if not np.all(np.isfinite(values)):
        raise ValueError("Model returned invalid numeric values.")

    if task == "regression":
        return float(values[0])

    if task == "multilabel":
        if np.all((values >= 0) & (values <= 1)):
            return values

        return 1.0 / (
            1.0 + np.exp(-np.clip(values, -50, 50))
        )

    if len(values) == 1:
        probability = float(
            1.0 / (
                1.0 + np.exp(-np.clip(values[0], -50, 50))
            )
        )
        return {
            "Class 0": 1.0 - probability,
            "Class 1": probability,
        }

    shifted = values - np.max(values)
    exp_values = np.exp(np.clip(shifted, -50, 50))
    probabilities = exp_values / exp_values.sum()

    return {
        f"Class {index}": float(probability)
        for index, probability in enumerate(probabilities)
    }


# =====================================================
# MAIN PREDICTION DISPLAY
# =====================================================

def display_sex(result):
    if not isinstance(result, dict) or len(result) != 2:
        st.error("Sex model did not return two class scores.")
        return

    scores = list(result.values())
    predicted_index = int(np.argmax(scores))

    if predicted_index >= len(SEX_CLASS_LABELS):
        st.error("Sex class mapping is unavailable.")
        return

    st.metric("Predicted Sex", SEX_CLASS_LABELS[predicted_index])
    st.caption(
        "Confirm Female/Male class order against the training labels."
    )


def display_disease(result):
    scores = np.asarray(result).reshape(-1)

    if len(scores) != len(DISEASE_LABELS):
        st.error(
            f"Expected {len(DISEASE_LABELS)} disease outputs; "
            f"received {len(scores)}."
        )
        st.write("Raw outputs:", scores.tolist())
        return

    diabetes_score = float(np.clip(scores[1], 0, 1))

    status = (
        "Positive"
        if diabetes_score >= DIABETES_THRESHOLD
        else "Negative"
    )

    st.metric("Diabetes Model Result", status)
    st.write(f"Model score: {diabetes_score * 100:.1f}%")
    st.progress(diabetes_score)

    st.caption(
        "Experimental output only. Verify class order and threshold "
        "against training data."
    )

    with st.expander("All retinal disease scores"):
        for label, score in zip(DISEASE_LABELS, scores):
            score = float(np.clip(score, 0, 1))
            st.write(f"**{label}:** {score * 100:.1f}%")
            st.progress(score)


def display_model_result(name, result):
    st.subheader(name)

    if name == "Retinal Age":
        st.metric("Estimated Age", f"{float(result):.1f} years")

    elif name == "Retinal Sex":
        display_sex(result)

    elif name == "Retinal Disease":
        display_disease(result)

    elif name == "CIMT Estimate":
        st.metric("Raw CIMT Model Output", f"{float(result):.4f}")
        st.caption(
            "Units have not been verified. Classification is disabled "
            "until units and a validated threshold are confirmed."
        )


# =====================================================
# ADDITIONAL PREDICTIONS
# =====================================================

def display_additional_predictions(prediction_results):
    st.divider()
    st.header("Additional Predictions")

    cad_col, risk_col = st.columns(2)

    # CAD / HEART DISEASE
    with cad_col:
        st.subheader("❤️ CAD / Heart Disease")

        cad_path = MODEL_DIR / ADDITIONAL_MODELS["CAD / Heart Disease"]

        if cad_path.exists():
            st.warning(
                f"Model file found: {cad_path.name}. "
                "Its architecture, input format, class labels and "
                "calibration must be configured before prediction."
            )
        else:
            st.info(
                "CAD prediction is not available yet. "
                "A trained and validated CAD model is required."
            )

        st.caption(
            "No CAD probability or diagnosis is generated by this section."
        )

    # RISK PREDICTION
    with risk_col:
        st.subheader("📊 Risk Prediction")

        risk_path = MODEL_DIR / ADDITIONAL_MODELS["Risk Prediction"]

        if risk_path.exists():
            st.warning(
                f"Model file found: {risk_path.name}. "
                "Required input variables, calibration and risk categories "
                "must be configured before prediction."
            )
        else:
            st.info(
                "Risk prediction is not available yet. "
                "A validated risk model and its required inputs are needed."
            )

        st.caption(
            "No Low/Moderate/High risk category is invented."
        )

    st.divider()
    cimt_col, cap_col = st.columns(2)

    # CIMT CLASSIFICATION
    with cimt_col:
        st.subheader("🩺 CIMT Classification")

        cimt_value = prediction_results.get("CIMT Estimate")

        if cimt_value is not None:
            st.metric(
                "Raw CIMT Model Output",
                f"{float(cimt_value):.4f}",
            )

            st.warning(
                "Units and a validated cutoff are unknown. "
                "True/False classification is disabled."
            )
        else:
            st.info(
                "CIMT output is unavailable. Check the model loading result."
            )

    # CAP GRADE
    with cap_col:
        st.subheader("🔬 CAP Grade")

        cap_path = MODEL_DIR / ADDITIONAL_MODELS["CAP Grade"]

        if cap_path.exists():
            st.warning(
                f"Model file found: {cap_path.name}. "
                "Its architecture, grade labels and output mapping "
                "must be configured before prediction."
            )
        else:
            st.info(
                "CAP Grade prediction is not available yet. "
                "A trained CAP model and verified grade mapping are required."
            )

        st.caption(
            "No CAP grade is generated without a compatible trained model."
        )

    st.divider()
    st.caption(
        "Research prototype only. These outputs are not a medical diagnosis."
    )


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("👁️ Retina AI")
st.sidebar.write("Research Prediction Dashboard")
st.sidebar.caption(f"Device: {DEVICE}")

st.sidebar.subheader("Configured Models")

for name, (filename, _) in MODEL_FILES.items():
    if (MODEL_DIR / filename).exists():
        st.sidebar.success(f"{name}: Found")
    else:
        st.sidebar.error(f"{name}: Missing")

st.sidebar.subheader("Additional Model Files")

for name, filename in ADDITIONAL_MODELS.items():
    if (MODEL_DIR / filename).exists():
        st.sidebar.success(f"{name}: File found")
    else:
        st.sidebar.info(f"{name}: Not found")


# =====================================================
# MAIN APP
# =====================================================

st.title("👁️ Retina AI Research Dashboard")
st.write("Take a photo or upload a retinal image.")

st.warning(
    "For research use only. This app does not provide a confirmed "
    "medical diagnosis or replace professional clinical assessment."
)

source = st.radio(
    "Choose image source",
    ["📷 Take Photo", "🖼️ Upload Image"],
    horizontal=True,
)

image = None

if source == "📷 Take Photo":
    camera_photo = st.camera_input(
        "Take a retinal photo",
        key="retina_camera",
    )

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")

else:
    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Could not open image. Upload a valid image.")


# =====================================================
# RUN ALL CONFIGURED PREDICTIONS
# =====================================================

if image is not None:
    image_col, info_col = st.columns(2)

    with image_col:
        st.image(
            image,
            caption="Selected retinal image",
            use_container_width=True,
        )

    with info_col:
        st.subheader("Image Information")
        st.write(f"Width: {image.width} px")
        st.write(f"Height: {image.height} px")
        st.write(f"Device: {DEVICE}")

    if st.button(
        "🔍 Run Predictions",
        type="primary",
        use_container_width=True,
    ):
        prediction_results = {}

        st.divider()
        st.header("Prediction Results")

        # Dict insertion order determines the display order.
        for model_name, (filename, task) in MODEL_FILES.items():
            with st.expander(model_name, expanded=True):
                try:
                    with st.spinner(f"Running {model_name}..."):
                        model = load_model(
                            str(MODEL_DIR / filename),
                            task,
                        )
                        result = run_prediction(model, image, task)

                    prediction_results[model_name] = result
                    display_model_result(model_name, result)

                except Exception as error:
                    st.error(f"{model_name} failed: {error}")

        # Additional cards use the results from this same run.
        display_additional_predictions(prediction_results)

else:
    st.info("Choose Take Photo or Upload Image to get started.")

st.sidebar.caption(
    "Camera access requires browser permission and HTTPS when deployed."
)
