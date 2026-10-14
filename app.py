import os
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# ---------------- PAGE SETTINGS ----------------

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


# ---------------- MODEL SETTINGS ----------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILES = {
    "Retinal Age": ("retina_age_model.pth", "regression"),
    "Retinal Disease": ("retina_disease_model.pth", "multilabel"),
    "Retinal Sex": ("retina_sex_model_v2.pth", "classification"),
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

# IMPORTANT:
# Change this only after checking the original training labels.
# This example assumes class 0 = Female and class 1 = Male.
SEX_CLASS_LABELS = ["Female", "Male"]


# ---------------- IMAGE TRANSFORM ----------------

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- MODEL HELPERS ----------------

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
                checkpoint = checkpoint[key]
                break

        if isinstance(checkpoint, dict):
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
    candidates = []

    for key, value in state.items():
        if (
            value.ndim == 2
            and key.endswith((
                "classifier.1.weight",
                "classifier.weight",
                "fc.weight",
            ))
        ):
            candidates.append((len(key), int(value.shape[0])))

    return sorted(candidates)[0][1] if candidates else default


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

        result = model.load_state_dict(state, strict=False)

        if len(state) - len(result.unexpected_keys) <= 0:
            raise RuntimeError(
                "Checkpoint does not match the configured architecture."
            )

    model.to(DEVICE)
    model.eval()

    return model


# ---------------- PREDICTION ----------------

def run_prediction(model, image, task):
    tensor = TRANSFORM(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        output = output.get(
            "logits",
            output.get("output", next(iter(output.values()))),
        )

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
        p = float(
            1.0 / (
                1.0 + np.exp(-np.clip(values[0], -50, 50))
            )
        )
        return {"Class 0": 1.0 - p, "Class 1": p}

    values = values - np.max(values)
    exp_values = np.exp(np.clip(values, -50, 50))
    probabilities = exp_values / exp_values.sum()

    return {
        f"Class {i}": float(p)
        for i, p in enumerate(probabilities)
    }


# ---------------- DISPLAY RESULTS ----------------

def display_result(title, result, task):
    st.subheader(title)

    if title == "Retinal Age":
        st.metric(
            "Estimated Age",
            f"{float(result):.1f} years",
        )

    elif title == "Retinal Sex":
        if not isinstance(result, dict) or len(result) < 2:
            st.error("Sex model did not return two class scores.")
            return

        scores = list(result.values())
        predicted_index = int(np.argmax(scores))

        if predicted_index < len(SEX_CLASS_LABELS):
            predicted_label = SEX_CLASS_LABELS[predicted_index]
        else:
            predicted_label = f"Class {predicted_index}"

        st.metric("Predicted Sex", predicted_label)

        for index, score in enumerate(scores):
            label = (
                SEX_CLASS_LABELS[index]
                if index < len(SEX_CLASS_LABELS)
                else f"Class {index}"
            )
            st.write(f"{label}: {score * 100:.1f}%")
            st.progress(float(np.clip(score, 0, 1)))

        st.caption(
            "Male/Female labels are valid only if they match the "
            "model's original training class order."
        )

    elif title == "CIMT Estimate":
        st.metric(
            "CIMT Model Output",
            f"{float(result):.4f}",
        )
        st.caption(
            "Confirm the output units and validated threshold before "
            "interpreting the value clinically."
        )

    elif title == "Retinal Disease":
        scores = np.asarray(result).reshape(-1)

        labels = (
            DISEASE_LABELS
            if len(scores) == len(DISEASE_LABELS)
            else [f"Output {i + 1}" for i in range(len(scores))]
        )

        for label, score in zip(labels, scores):
            score = float(np.clip(score, 0, 1))
            st.write(f"**{label}:** {score * 100:.1f}%")
            st.progress(score)

        st.caption(
            "These are model scores, not confirmed diagnoses. "
            "Validated thresholds are needed for Positive/Negative labels."
        )


# ---------------- SIDEBAR ----------------

st.sidebar.title("👁️ Retina AI")
st.sidebar.write("Research Prediction Dashboard")
st.sidebar.caption(f"Device: {DEVICE}")

st.sidebar.subheader("Model Files")

for name, (filename, _) in MODEL_FILES.items():
    if (MODEL_DIR / filename).exists():
        st.sidebar.success(f"{name}: Found")
    else:
        st.sidebar.error(f"{name}: Missing")


# ---------------- MAIN APP ----------------

st.title("👁️ Retina AI Research Dashboard")
st.write("Take a photo using your camera or upload a retinal image.")

st.warning(
    "Research use only. Predictions are not a medical diagnosis "
    "and should not be used to make treatment decisions."
)


# ---------------- CAMERA / UPLOAD ----------------

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
        image = Image.open(uploaded_file).convert("RGB")


# ---------------- RUN PREDICTIONS ----------------

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

        for model_name, (filename, task) in MODEL_FILES.items():
            with st.expander(model_name, expanded=True):
                try:
                    with st.spinner(f"Running {model_name}..."):
                        model = load_model(
                            str(MODEL_DIR / filename),
                            task,
                        )
                        result = run_prediction(model, image, task)

                    display_result(model_name, result, task)

                except Exception as error:
                    st.error(f"{model_name} failed: {error}")

        st.divider()
        st.subheader("Additional Predictions")

        st.info(
            "CAD/Heart Disease and CAP grade are not displayed because "
            "separate trained models and validated output mappings have "
            "not been configured."
        )

else:
    st.info("Choose Take Photo or Upload Image to get started.")

st.sidebar.caption(
    "Camera access depends on browser permissions. "
    "Use HTTPS and allow camera access when prompted."
)
