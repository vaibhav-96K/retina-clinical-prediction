import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
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
.stApp {background-color:#f5f8fc;}
h1,h2,h3 {color:#17365d;}
div[data-testid="stMetric"] {
    background:white;
    border:1px solid #dce5f0;
    padding:14px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

MODEL_FILES = {
    "Retinal Age": ("retina_age_model.pth", "regression"),
    "Retinal Sex": ("retina_sex_model_v2.pth", "classification"),
    "Retinal Disease": ("retina_disease_model.pth", "multilabel"),
    "CIMT Estimate": ("cimt_thickness_model.pth", "regression"),
}

DISEASE_LABELS = [
    "Normal", "Diabetes", "Glaucoma", "Cataract",
    "AMD", "Hypertension", "Myopia", "Other"
]

SEX_CLASS_LABELS = ["Female", "Male"]
DIABETES_THRESHOLD = 0.50

CAD_MODEL_PATH = MODEL_DIR / "cad_heart_disease_model.joblib"
CAP_MODEL_PATH = MODEL_DIR / "cap_grade_model.pth"

# Keras retinal + clinical feature-fusion model and its scaler.
# PyTorch retinal + clinical fusion model
FUSION_MODEL_PATH = MODEL_DIR / "retinal_feature_fusion_model.pth"
CLINICAL_SCALER_PATH = MODEL_DIR / "clinical_scaler.pkl"

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# =====================================================
# CHECKPOINT HELPERS
# =====================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in (
            "state_dict", "model_state_dict",
            "model", "net", "weights"
        ):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]

        if checkpoint and all(
            torch.is_tensor(v) for v in checkpoint.values()
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


# =====================================================
# RETINAL MODEL LOADER
# =====================================================

@st.cache_resource(show_spinner=False)
def load_model(path, task, modified_time):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file missing: {path}")

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = clean_state_dict(
            extract_state_dict(checkpoint)
        )

        output_size = (
            8 if task == "multilabel" else 1
        )

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            output_size,
        )

        model.load_state_dict(state, strict=True)

    return model.to(DEVICE).eval()


def run_prediction(model, image, task):
    x = TRANSFORM(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(x)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        output = output.get(
            "logits", output.get("output")
        )

    values = output.detach().float().cpu().reshape(-1).numpy()

    if not np.all(np.isfinite(values)):
        raise ValueError("Model returned invalid values.")

    if task == "regression":
        return float(values[0])

    if task == "multilabel":
        if np.all((values >= 0) & (values <= 1)):
            return values
        return 1 / (1 + np.exp(-np.clip(values, -50, 50)))

    if len(values) == 1:
        p = float(
            1 / (1 + np.exp(-np.clip(values[0], -50, 50)))
        )
        return {"Class 0": 1 - p, "Class 1": p}

    exp_values = np.exp(
        np.clip(values - np.max(values), -50, 50)
    )
    probabilities = exp_values / exp_values.sum()

    return {
        f"Class {i}": float(p)
        for i, p in enumerate(probabilities)
    }


# =====================================================
# RETINAL RESULT DISPLAY
# =====================================================

def display_sex(result):
    if not isinstance(result, dict) or len(result) != 2:
        st.error("Sex model did not return two class scores.")
        return

    idx = int(np.argmax(list(result.values())))

    st.metric(
        "Predicted Sex (verify mapping)",
        SEX_CLASS_LABELS[idx],
    )
    st.caption(
        "Confirm Female/Male class order against training labels."
    )


def display_disease(result):
    scores = np.asarray(result).reshape(-1)

    if len(scores) != 8:
        st.error(f"Expected 8 outputs, received {len(scores)}.")
        return

    diabetes_score = float(np.clip(scores[1], 0, 1))

    st.metric(
        "Diabetes Model Result",
        "Positive" if diabetes_score >= DIABETES_THRESHOLD
        else "Negative",
    )
    st.write(f"Model score: {diabetes_score:.1%}")
    st.progress(diabetes_score)

    st.caption(
        "Experimental result. Verify label order and threshold."
    )

    with st.expander("All Retinal Disease Scores"):
        for label, score in zip(DISEASE_LABELS, scores):
            score = float(np.clip(score, 0, 1))
            st.write(f"**{label}:** {score:.1%}")
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
        st.metric("Raw CIMT Output", f"{float(result):.4f}")
        st.caption(
            "Units and clinical cutoff are not verified."
        )


# =====================================================
# RETINAL + CLINICAL FUSION RISK MODEL
# =====================================================

def display_risk_prediction(image):
    st.subheader("📊 Fusion Risk Model")
    st.caption(
        "This model is intended to combine a retinal image and clinical "
        "features. It is listed separately because it uses both data types."
    )

    if not FUSION_MODEL_PATH.is_file():
        st.info(
            "Fusion model not found. Expected file: "
            "models/retinal_feature_fusion_model.pth"
        )
        return

    st.success("Fusion Risk Model: Found")
    st.write(f"Model file: `{FUSION_MODEL_PATH.relative_to(PROJECT_DIR)}`")

    if not CLINICAL_SCALER_PATH.is_file():
        st.warning(
            "Clinical Scaler: Not found (`models/clinical_scaler.pkl`). "
            "If the fusion model was trained with scaled clinical features, "
            "the matching scaler file is also required."
        )

    st.info(
        "The .pth file is detected, but prediction is not enabled yet because "
        "the exact PyTorch fusion architecture and its input format must match "
        "the architecture used during training. Loading a state_dict with a "
        "different architecture can produce incorrect results or errors. "
        "Once the training model class / save code is added, this section can "
        "run the retinal-image + clinical-feature prediction."
    )

    with st.expander("Fusion model diagnostic"):
        try:
            checkpoint = torch.load(
                FUSION_MODEL_PATH,
                map_location="cpu",
                weights_only=False
            )
            if isinstance(checkpoint, nn.Module):
                st.success(
                    "Checkpoint contains a complete PyTorch model object. "
                    "Its input signature still needs to be verified."
                )
            elif isinstance(checkpoint, dict):
                st.write("Checkpoint keys:", list(checkpoint.keys())[:30])
                architecture = checkpoint.get("architecture")
                if architecture:
                    st.write("Architecture metadata:", architecture)
                st.caption(
                    "Checkpoint is a dictionary/state_dict. Its exact model "
                    "class must be recreated from the training code."
                )
            else:
                st.write("Checkpoint type:", type(checkpoint).__name__)
        except Exception as error:
            st.error(
                f"Could not inspect fusion checkpoint: "
                f"{type(error).__name__}: {error}"
            )


# =====================================================
# CAD MODEL
# =====================================================

CAD_CATEGORICAL_OPTIONS = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}

CAD_NUMERIC_RANGES = {
    "age": (1, 120, 50.0),
    "trestbps": (50, 300, 120.0),
    "chol": (50, 800, 200.0),
    "thalach": (30, 250, 150.0),
    "oldpeak": (0.0, 10.0, 1.0),
}


@st.cache_resource(show_spinner=False)
def load_cad_model(path, modified_time):
    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError("CAD artifact must be a dictionary.")

    missing = {"model", "feature_columns", "classes"} - set(artifact)

    if missing:
        raise ValueError(
            f"CAD artifact missing: {sorted(missing)}"
        )

    if not artifact["feature_columns"]:
        raise ValueError("CAD feature_columns is empty.")

    return artifact


def display_cad_prediction():
    st.subheader("❤️ CAD / Heart Disease")

    if not CAD_MODEL_PATH.exists():
        st.info(
            "CAD model not installed. Add "
            "models/cad_heart_disease_model.joblib."
        )
        return

    try:
        artifact = load_cad_model(
            str(CAD_MODEL_PATH),
            CAD_MODEL_PATH.stat().st_mtime,
        )
    except Exception as error:
        st.error(f"Could not load CAD model: {error}")
        return

    features = artifact["feature_columns"]
    numeric = artifact.get("numeric_columns", [])
    categorical = artifact.get("categorical_columns", [])

    values = {}

    with st.form("cad_clinical_form"):
        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):
            with (col1 if i % 2 == 0 else col2):
                if feature in CAD_CATEGORICAL_OPTIONS:
                    value = st.selectbox(
                        f"{feature} (dataset code)",
                        CAD_CATEGORICAL_OPTIONS[feature],
                        key=f"cad_{feature}",
                    )

                elif feature in CAD_NUMERIC_RANGES:
                    lo, hi, default = CAD_NUMERIC_RANGES[feature]

                    value = st.number_input(
                        feature,
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(default),
                        step=0.1 if feature == "oldpeak" else 1.0,
                        key=f"cad_{feature}",
                    )

                elif feature in numeric:
                    value = st.number_input(
                        feature,
                        value=0.0,
                        key=f"cad_{feature}"
                    )

                else:
                    value = st.text_input(
                        f"{feature} (category/code)",
                        key=f"cad_{feature}",
                    ).strip()

                values[feature] = value

        submitted = st.form_submit_button(
            "Run CAD Prediction",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    try:
        row = pd.DataFrame(
            [[values[col] for col in features]],
            columns=features,
        )

        for col in numeric:
            row[col] = pd.to_numeric(
                row[col],
                errors="raise"
            )

        model = artifact["model"]
        prediction = int(model.predict(row)[0])
        classes = artifact["classes"]

        if prediction < 0 or prediction >= len(classes):
            raise ValueError(
                "Prediction does not match saved class mapping."
            )

        st.write(
            f"**Predicted class:** {classes[prediction]}"
        )

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(row)[0]

            for label, score in zip(classes, scores):
                st.write(f"- {label}: {float(score):.1%}")

        st.warning(
            "Research prototype only; not a diagnosis."
        )

    except Exception as error:
        st.error(f"CAD prediction failed: {error}")


# =====================================================
# SESSION STATE
# =====================================================

if "retinal_results" not in st.session_state:
    st.session_state.retinal_results = {}

if "last_retinal_image_key" not in st.session_state:
    st.session_state.last_retinal_image_key = None

if "current_risk_image_key" not in st.session_state:
    st.session_state.current_risk_image_key = None

if "risk_result_image_key" not in st.session_state:
    st.session_state.risk_result_image_key = None

if "risk_result_level" not in st.session_state:
    st.session_state.risk_result_level = None


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("👁️ Retina AI")
st.sidebar.write("Research Prediction Dashboard")
st.sidebar.caption(f"Device: {DEVICE}")
st.sidebar.subheader("Retinal Models")

for name, (filename, _) in MODEL_FILES.items():
    if (MODEL_DIR / filename).exists():
        st.sidebar.success(f"{name}: Found")
    else:
        st.sidebar.error(f"{name}: Missing")

st.sidebar.subheader("Clinical Models")

if CAD_MODEL_PATH.exists():
    st.sidebar.success("CAD model: Found")
else:
    st.sidebar.info("CAD model: Not found")

if FUSION_MODEL_PATH.is_file():
    st.sidebar.success("Fusion Risk Model: Found")
else:
    st.sidebar.info("Fusion Risk Model: Not found")

if CLINICAL_SCALER_PATH.is_file():
    st.sidebar.success("Clinical Scaler: Found")
else:
    st.sidebar.info("Clinical Scaler: Not found")

if CAP_MODEL_PATH.exists():
    st.sidebar.success("CAP Grade: Found")
else:
    st.sidebar.info("CAP Grade: Not found")


# =====================================================
# MAIN DASHBOARD AND IMAGE INPUT
# =====================================================

st.title("👁️ Retina AI Research Dashboard")
st.caption(
    "Upload a retinal image, capture a retinal image, "
    "or capture an external eye photo."
)
st.warning(
    "For research use only. Not a confirmed diagnosis."
)

upload_col, retina_col, external_col = st.columns(3)

with upload_col:
    st.subheader("🖼️ Upload Image")

    uploaded_file = st.file_uploader(
        "Choose retinal image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="retina_upload",
    )

with retina_col:
    st.subheader("📷 Retina Camera")

    camera_photo = st.camera_input(
        "Capture retinal/fundus image",
        key="retina_camera",
    )

with external_col:
    st.subheader("👁️ External Eye Photo")

    external_eye_photo = st.camera_input(
        "Capture external eye",
        key="external_eye_camera",
    )

image = None
image_key = None

try:
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        image = Image.open(uploaded_file).convert("RGB")
        image_key = str(hash(raw))

    elif camera_photo is not None:
        raw = camera_photo.getvalue()
        image = Image.open(camera_photo).convert("RGB")
        image_key = str(hash(raw))

except Exception as error:
    st.error(f"Could not open retinal image: {error}")


# =====================================================
# UPDATE RISK STATE WHEN IMAGE CHANGES
# =====================================================

if image_key != st.session_state.current_risk_image_key:
    st.session_state.current_risk_image_key = image_key
    st.session_state.risk_result_image_key = None
    st.session_state.risk_result_level = None


# =====================================================
# RETINAL PREDICTIONS
# =====================================================

if image is not None:
    st.divider()
    image_col, info_col = st.columns([1, 1])

    with image_col:
        st.subheader("Selected Retinal Image")

        st.image(
            image,
            caption="Retinal / Fundus Image",
            use_container_width=True,
        )

    with info_col:
        st.subheader("Image Information")
        st.write(f"Width: {image.width} px")
        st.write(f"Height: {image.height} px")
        st.write(f"Runtime device: {DEVICE}")

        if image_key != st.session_state.last_retinal_image_key:
            st.caption("New image selected. Run predictions.")

        run_button = st.button(
            "🔍 Run Retinal Predictions",
            type="primary",
            use_container_width=True,
            key="run_retinal_predictions",
        )

    if run_button:
        results = {}
        st.session_state.retinal_results = {}
        st.session_state.last_retinal_image_key = image_key

        st.divider()
        st.header("Retinal Prediction Results")

        for name, (filename, task) in MODEL_FILES.items():
            with st.expander(name, expanded=True):
                try:
                    path = MODEL_DIR / filename

                    model = load_model(
                        str(path),
                        task,
                        path.stat().st_mtime if path.exists() else 0,
                    )

                    result = run_prediction(
                        model,
                        image,
                        task
                    )

                    results[name] = result
                    display_model_result(name, result)

                except Exception as error:
                    st.error(f"{name} failed: {error}")

        st.session_state.retinal_results = results

elif st.session_state.retinal_results:
    st.header("Previous Retinal Prediction Results")

    for name, result in st.session_state.retinal_results.items():
        with st.expander(name):
            display_model_result(name, result)


# =====================================================
# RISK PREDICTION
# =====================================================

st.divider()
display_risk_prediction(image)


# =====================================================
# EXTERNAL EYE PHOTO
# =====================================================

if external_eye_photo is not None:
    st.divider()
    st.header("👁️ External Eye Photo")

    try:
        external_image = Image.open(
            external_eye_photo
        ).convert("RGB")

        st.image(
            external_image,
            caption="Captured External Eye Photo",
            use_container_width=True,
        )

        st.success(
            "External eye photo captured successfully."
        )
        st.write(f"Width: {external_image.width} px")
        st.write(f"Height: {external_image.height} px")

        st.info(
            "External-eye prediction is not enabled. A separately "
            "trained model is required."
        )

    except Exception as error:
        st.error(
            f"Could not open external eye photo: {error}"
        )


# =====================================================
# CIMT CLASSIFICATION
# =====================================================

st.divider()
st.subheader("🩺 CIMT Classification")

cimt_value = st.session_state.retinal_results.get(
    "CIMT Estimate"
)

if cimt_value is not None:
    st.metric(
        "Raw CIMT Model Output",
        f"{float(cimt_value):.4f}"
    )

    st.warning(
        "Units and validated cutoff are unknown. "
        "Clinical classification is disabled."
    )

else:
    st.info(
        "Run retinal predictions to display CIMT output."
    )


# =====================================================
# CAD AND CAP
# =====================================================

st.divider()
display_cad_prediction()

st.divider()
st.subheader("🔬 CAP Grade")

if CAP_MODEL_PATH.exists():
    st.warning(
        "CAP model file exists, but its architecture and grade "
        "mapping are not configured."
    )

else:
    st.info(
        "A compatible trained CAP model and verified grade labels "
        "are required."
    )

st.divider()
st.caption(
    "Research prototype only. Not validated for clinical diagnosis."
)
st.sidebar.caption(
    "Camera access may require browser permission and HTTPS."
)
