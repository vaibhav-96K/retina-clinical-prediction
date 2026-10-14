
import os
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# ==================== PAGE SETTINGS ====================

st.set_page_config(
    page_title="Retina AI Dashboard",
    page_icon="👁️",
    layout="wide",
)

st.markdown("""
<style>
.main {background-color: #f5f8fc;}
h1, h2, h3 {color: #17365d;}
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #dce5f0;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==================== CONFIGURATION ====================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Display order requested:
# Retinal Age -> Retinal Sex -> Retinal Disease -> CIMT Estimate
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

# Confirm class order from the original training labels.
SEX_CLASS_LABELS = ["Female", "Male"]

# Temporary threshold; must be validated against the training data.
DIABETES_THRESHOLD = 0.50


# ==================== IMAGE TRANSFORM ====================

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==================== MODEL LOADING ====================

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

        # Stop if this architecture does not match the checkpoint.
        model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()
    return model


# ==================== PREDICTION ====================

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


# ==================== DISPLAY HELPERS ====================

def display_sex(result):
    if not isinstance(result, dict) or len(result) != 2:
        st.error("Sex model did not return exactly two class scores.")
        return

    scores = list(result.values())
    predicted_index = int(np.argmax(scores))

    if predicted_index >= len(SEX_CLASS_LABELS):
        st.error("Sex class mapping is unavailable.")
        return

    st.metric(
        "Predicted Sex",
        SEX_CLASS_LABELS[predicted_index],
    )

    st.caption(
        "Verify Male/Female order against the original training labels."
    )


def display_disease(result):
    scores = np.asarray(result).reshape(-1)

    if len(scores) != len(DISEASE_LABELS):
        st.error(
            f"Expected {len(DISEASE_LABELS)} disease outputs, "
            f"but model returned {len(scores)}."
        )
        st.write("Raw model outputs:", scores.tolist())
        return

    # This index is valid only if the training label order matches.
    diabetes_score = float(np.clip(scores[1], 0, 1))

    diabetes_status = (
        "Positive"
        if diabetes_score >= DIABETES_THRESHOLD
        else "Negative"
    )

    st.metric("Diabetes Model Result", diabetes_status)
    st.write(f"Diabetes model score: {diabetes_score * 100:.1f}%")
    st.progress(diabetes_score)

    st.caption(
        "Experimental result only. Confirm the training label order "
        "and validate the threshold before interpreting the output."
    )

    with st.expander("View all retinal disease scores"):
        for label, score in zip(DISEASE_LABELS, scores):
            score = float(np.clip(score, 0, 1))
            st.write(f"**{label}:** {score * 100:.1f}%")
            st.progress(score)


def display_model_result(title, result):
    st.subheader(title)

    if title == "Retinal Age":
        st.metric(
            "Estimated Age",
            f"{float(result):.1f} years",
        )

    elif title == "Retinal Sex":
        display_sex(result)

    elif title == "Retinal Disease":
        display_disease(result)

    elif title == "CIMT Estimate":
        st.metric(
            "CIMT Model Output",
            f"{float(result):.4f}",
        )
        st.caption(
            "Numeric regression output. Confirm the units and a validated "
            "threshold before converting this to True/False."
        )


# ==================== SIDEBAR ====================

st.sidebar.title("👁️ Retina AI")
st.sidebar.write("Research Prediction Dashboard")
st.sidebar.caption(f"Device: {DEVICE}")

st.sidebar.subheader("Model Files")

for name, (filename, _) in MODEL_FILES.items():
    if (MODEL_DIR / filename).exists():
        st.sidebar.success(f"{name}: Found")
    else:
        st.sidebar.error(f"{name}: Missing")


# ==================== MAIN APP ====================

st.title("👁️ Retina AI Research Dashboard")
st.write("Take a photo or upload a retinal image.")

st.warning(
    "Research use only. This app does not provide a confirmed medical "
    "diagnosis and should not be used to make treatment decisions."
)


# ==================== CAMERA / UPLOAD ====================

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


# ==================== RUN PREDICTIONS ====================

if image is not None:
    left, right = st.columns(2)

    with left:
        st.image(
            image,
            caption="Selected retinal image",
            use_container_width=True,
        )

    with right:
        st.subheader("Image Information")
        st.write(f"Width: {image.width} px")
        st.write(f"Height: {image.height} px")

    if st.button(
        "🔍 Run Predictions",
        type="primary",
        use_container_width=True,
    ):
        st.divider()
        st.header("Prediction Results")

        # MODEL_FILES insertion order controls displayed result order.
        for model_name, (filename, task) in MODEL_FILES.items():
            with st.expander(model_name, expanded=True):
                try:
                    with st.spinner(f"Running {model_name}..."):
                        model = load_model(
                            str(MODEL_DIR / filename),
                            task,
                        )
                        result = run_prediction(model, image, task)

                    display_model_result(model_name, result)

                except Exception as error:
                    st.error(f"{model_name} failed: {error}")

        st.divider()
        st.header("Additional Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### CAD / Heart Disease")
            st.info(
                "Not configured: a trained CAD/heart disease model "
                "and verified output mapping are not configured."
            )

            st.markdown("### Risk Prediction")
            st.info(
                "Not configured: a validated risk model or calibrated "
                "probability calculation is not available."
            )

        with col2:
            st.markdown("### CIMT Classification")
            st.info(
                "The current CIMT model returns a numeric value. "
                "True/False requires verified units and a validated threshold."
            )

            st.markdown("### CAP Grade")
            st.info(
                "Not configured: a trained CAP grade model and verified "
                "grade mapping are not configured."
            )

else:
    st.info("Choose Take Photo or Upload Image to get started.")


# ==================== FOOTER ====================

st.sidebar.caption(
    "Camera access depends on browser permissions. "
    "Use HTTPS and allow camera access when prompted."
)
