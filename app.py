
# ============================================================
# 👁️ RETINA AI — RESEARCH PREDICTION DASHBOARD
# ============================================================

import io
import json
import hashlib
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import tensorflow as tf

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
FUSION_MODEL_PATH = MODEL_DIR / "retinal_feature_fusion_model_FINAL.keras"
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
    """Show one concise fusion-risk model status and its diagnostics."""
    st.subheader("📊 Fusion Risk Prediction")
    st.caption(
        "Uses the retinal image and clinical features together. "
        "Research prototype only; not a medical diagnosis."
    )

    if not FUSION_MODEL_PATH.is_file():
        st.warning(
            "Fusion risk model not found. Expected file: "
            "models/retinal_feature_fusion_model_FINAL.keras"
        )
        return

    st.success("Fusion risk model found and ready for inspection.")
    st.caption(f"Model file: `{FUSION_MODEL_PATH.relative_to(PROJECT_DIR)}`")

    if not CLINICAL_SCALER_PATH.is_file():
        st.warning(
            "Clinical scaler not found (`models/clinical_scaler.pkl`). "
            "If the model was trained with scaled clinical features, its "
            "matching scaler is required for valid predictions."
        )

    with st.expander("Fusion risk model diagnostics", expanded=False):
        try:
            keras_model = tf.keras.models.load_model(
                FUSION_MODEL_PATH,
                compile=False,
            )
            st.write("Model status: Loaded successfully")
            st.write("Input shape(s):", keras_model.input_shape)
            st.write("Output shape(s):", keras_model.output_shape)
            st.caption(
                "A risk prediction is not run here because the exact training-time "
                "input order, clinical feature schema, image preprocessing, and "
                "scaler must match the saved model. Showing a score without those "
                "details could produce an invalid result."
            )
        except Exception as error:
            st.error(
                f"Could not load fusion risk model: "
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
        # Decode from bytes so the file pointer state cannot affect preview.
        image = Image.open(io.BytesIO(raw)).convert("RGB").copy()
        image_key = str(hash(raw))

    elif camera_photo is not None:
        raw = camera_photo.getvalue()
        image = Image.open(io.BytesIO(raw)).convert("RGB").copy()
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

        # Encode explicitly as PNG bytes. This avoids Streamlit/browser
        # rendering issues with array-backed images while keeping the layout.
        preview_buffer = io.BytesIO()
        image.convert("RGB").save(preview_buffer, format="PNG")
        preview_bytes = preview_buffer.getvalue()
        if not preview_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Could not create a valid PNG preview.")
        st.image(
            preview_bytes,
            caption="Retinal / Fundus Image",
            output_format="PNG",
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

        external_buffer = io.BytesIO()
        external_image.convert("RGB").save(external_buffer, format="PNG")
        external_bytes = external_buffer.getvalue()
        st.image(
            external_bytes,
            caption="Captured External Eye Photo",
            output_format="PNG",
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retina AI Dashboard",
    page_icon="👁️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background-color: #f5f8fc;
}
h1, h2, h3 {
    color: #17365d;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce5f0;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

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
DIABETES_THRESHOLD = 0.50

CAD_MODEL_PATH = MODEL_DIR / "cad_heart_disease_model.joblib"
CAP_MODEL_PATH = MODEL_DIR / "cap_grade_model.pth"

FUSION_MODEL_PATH = (
    MODEL_DIR / "retinal_feature_fusion_model_FINAL.keras"
)
CLINICAL_SCALER_PATH = MODEL_DIR / "clinical_scaler.pkl"
FUSION_METADATA_PATH = MODEL_DIR / "fusion_metadata.json"

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "retinal_results": {},
    "last_retinal_image_key": None,
    "fusion_result": None,
    "fusion_result_key": None,
    "cad_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def image_hash(raw_bytes):
    return hashlib.sha256(raw_bytes).hexdigest()


def safe_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    exp_values = np.exp(np.clip(values, -80, 80))
    return exp_values / exp_values.sum()


def safe_sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -80, 80)
    return 1.0 / (1.0 + np.exp(-values))


def validate_finite(values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Model returned NaN or infinite values.")
    return values


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
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate

        if checkpoint and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise ValueError(
        "Unsupported PyTorch checkpoint. "
        "Expected a model or state dictionary."
    )


def clean_state_dict(state):
    cleaned = {}

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        while key.startswith(("module.", "model.", "net.")):
            key = key.split(".", 1)[1]

        cleaned[key] = value

    return cleaned


def get_model_output(output):
    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        elif "output" in output:
            output = output["output"]
        else:
            raise ValueError(
                "Model output dictionary has no logits/output key."
            )

    return output


# ============================================================
# RETINAL MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_retinal_model(path_string, task, modified_time):
    path = Path(path_string)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model file: {path.name}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    # Full saved PyTorch model
    if isinstance(checkpoint, nn.Module):
        model = checkpoint

    else:
        state = clean_state_dict(
            extract_state_dict(checkpoint)
        )

        # Assumes the checkpoint was trained using EfficientNet-B0.
        # Output dimensions must match the original training architecture.
        if task == "multilabel":
            output_size = len(DISEASE_LABELS)
        elif task == "classification":
            # A binary classifier may use either one logit or two logits.
            # Try the checkpoint's classifier dimensions.
            classifier_weight = state.get(
                "classifier.1.weight"
            )
            if classifier_weight is None:
                raise ValueError(
                    "Cannot identify classifier dimensions. "
                    "Check the saved checkpoint architecture."
                )
            output_size = int(classifier_weight.shape[0])
        else:
            output_size = 1

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            output_size,
        )

        model.load_state_dict(state, strict=True)

    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# RETINAL PREDICTION
# ============================================================

def run_retinal_prediction(model, image, task):
    tensor = TRANSFORM(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = get_model_output(model(tensor))

    values = output.detach().float().cpu().numpy().reshape(-1)
    values = validate_finite(values)

    if task == "regression":
        return float(values[0])

    if task == "multilabel":
        # Use sigmoid for logits. If outputs already look like
        # probabilities, preserve them.
        if np.all((values >= 0) & (values <= 1)):
            return values
        return safe_sigmoid(values)

    # Binary classifier with one output logit
    if len(values) == 1:
        probability = float(safe_sigmoid(values)[0])
        return {
            "Class 0": 1.0 - probability,
            "Class 1": probability,
        }

    probabilities = safe_softmax(values)

    return {
        f"Class {i}": float(probability)
        for i, probability in enumerate(probabilities)
    }


# ============================================================
# RETINAL RESULTS DISPLAY
# ============================================================

def display_sex(result):
    if not isinstance(result, dict) or len(result) != 2:
        st.error("Sex model must return exactly two class scores.")
        return

    probabilities = list(result.values())
    index = int(np.argmax(probabilities))

    # Confirm this mapping against the training dataset!
    st.metric(
        "Predicted Sex",
        SEX_CLASS_LABELS[index],
    )
    st.write(f"Model score: {probabilities[index]:.1%}")
    st.caption(
        "Experimental output. Confirm class order against training labels."
    )


def display_disease(result):
    scores = np.asarray(result).reshape(-1)

    if len(scores) != len(DISEASE_LABELS):
        st.error(
            f"Expected {len(DISEASE_LABELS)} disease scores; "
            f"received {len(scores)}."
        )
        return

    scores = np.clip(scores, 0, 1)

    diabetes_score = float(scores[1])

    st.metric(
        "Diabetes Model Result",
        "Positive"
        if diabetes_score >= DIABETES_THRESHOLD
        else "Negative",
    )
    st.write(f"Model score: {diabetes_score:.1%}")
    st.progress(diabetes_score)

    with st.expander("All Retinal Disease Scores"):
        for label, score in zip(DISEASE_LABELS, scores):
            st.write(f"**{label}:** {score:.1%}")
            st.progress(float(score))

    st.caption(
        "Thresholds and label order must match the training configuration."
    )


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
            "Units and validated clinical cutoff are not configured."
        )


# ============================================================
# FUSION MODEL — ACTUAL INFERENCE
# ============================================================

def load_fusion_metadata():
    if not FUSION_METADATA_PATH.is_file():
        raise FileNotFoundError(
            "fusion_metadata.json is missing. It must specify "
            "the original training-time input schema and output labels."
        )

    with open(FUSION_METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    required = {
        "clinical_features",
        "output_labels",
        "image_input_name",
        "clinical_input_name",
        "image_size",
    }

    missing = required - set(metadata)

    if missing:
        raise ValueError(
            f"Fusion metadata missing keys: {sorted(missing)}"
        )

    if not metadata["clinical_features"]:
        raise ValueError("clinical_features cannot be empty.")

    return metadata


@st.cache_resource(show_spinner=False)
def load_fusion_model(path_string, modified_time):
    return tf.keras.models.load_model(
        path_string,
        compile=False,
    )


def display_fusion_prediction(image):
    st.subheader("📊 Retinal + Clinical Fusion Risk Prediction")
    st.caption(
        "Combines retinal image features with clinical information. "
        "Research prototype only; not a medical diagnosis."
    )

    if not FUSION_MODEL_PATH.is_file():
        st.warning(
            f"Fusion model missing: {FUSION_MODEL_PATH.name}"
        )
        return

    if not FUSION_METADATA_PATH.is_file():
        st.warning(
            "Actual fusion inference is disabled until the matching "
            "fusion_metadata.json is provided."
        )
        return

    try:
        metadata = load_fusion_metadata()
        model = load_fusion_model(
            str(FUSION_MODEL_PATH),
            FUSION_MODEL_PATH.stat().st_mtime,
        )
    except Exception as error:
        st.error(f"Fusion model loading failed: {error}")
        return

    clinical_features = metadata["clinical_features"]

    # Input form is generated from the saved feature schema.
    values = {}

    with st.form("fusion_clinical_form"):
        st.markdown("**Clinical information**")
        columns = st.columns(2)

        for index, feature in enumerate(clinical_features):
            name = feature["name"]
            label = feature.get("label", name)
            default = float(feature.get("default", 0.0))
            minimum = float(feature.get("min", -1e6))
            maximum = float(feature.get("max", 1e6))
            step = float(feature.get("step", 1.0))

            with columns[index % 2]:
                values[name] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                    step=step,
                    key=f"fusion_{name}",
                )

        submitted = st.form_submit_button(
            "Run Fusion Prediction",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    try:
        # Image preprocessing must match the original Keras training.
        size = int(metadata["image_size"])
        image_array = np.asarray(
            image.convert("RGB").resize((size, size)),
            dtype=np.float32,
        )

        preprocessing = metadata.get(
            "image_preprocessing", "rescale_0_1"
        )

        if preprocessing == "rescale_0_1":
            image_array /= 255.0
        elif preprocessing == "none":
            pass
        elif preprocessing == "efficientnet":
            image_array = tf.keras.applications.efficientnet.preprocess_input(
                image_array
            )
        else:
            raise ValueError(
                f"Unsupported image preprocessing: {preprocessing}"
            )

        clinical_row = pd.DataFrame(
            [[values[item["name"]] for item in clinical_features]],
            columns=[item["name"] for item in clinical_features],
        )

        # A scaler is used only if metadata explicitly says it is required.
        if metadata.get("use_clinical_scaler", False):
            if not CLINICAL_SCALER_PATH.is_file():
                raise FileNotFoundError(
                    "Clinical scaler is required but missing."
                )

            scaler = joblib.load(CLINICAL_SCALER_PATH)
            clinical_array = scaler.transform(clinical_row)
        else:
            clinical_array = clinical_row.to_numpy(dtype=np.float32)

        image_name = metadata["image_input_name"]
        clinical_name = metadata["clinical_input_name"]

        model_input_names = [
            tensor.name.split(":")[0].split("/")[-1]
            for tensor in model.inputs
        ]

        if image_name not in model_input_names:
            raise ValueError(
                f"Image input '{image_name}' not found. "
                f"Model inputs: {model_input_names}"
            )

        if clinical_name not in model_input_names:
            raise ValueError(
                f"Clinical input '{clinical_name}' not found. "
                f"Model inputs: {model_input_names}"
            )

        input_map = {
            image_name: image_array[np.newaxis, ...],
            clinical_name: clinical_array.astype(np.float32),
        }

        prediction = model.predict(input_map, verbose=0)
        prediction = np.asarray(prediction)

        if prediction.ndim == 2 and prediction.shape[0] == 1:
            prediction = prediction[0]
        else:
            prediction = prediction.reshape(-1)

        prediction = validate_finite(prediction)
        labels = metadata["output_labels"]

        if len(labels) != len(prediction):
            raise ValueError(
                "Output labels do not match the model output dimensions."
            )

        output_type = metadata.get("output_type", "probabilities")

        if output_type == "logits":
            if len(prediction) == 1:
                prediction = safe_sigmoid(prediction)
            else:
                prediction = safe_softmax(prediction)
        elif output_type == "probabilities":
            if np.any(prediction < 0) or np.any(prediction > 1):
                raise ValueError(
                    "Expected probabilities between 0 and 1. "
                    "Check output_type in fusion_metadata.json."
                )
        elif output_type != "regression":
            raise ValueError(f"Unknown output_type: {output_type}")

        st.success("Fusion prediction completed.")

        if output_type == "regression":
            st.metric(
                "Fusion Model Output",
                f"{float(prediction[0]):.4f}",
            )
        else:
            for label, score in zip(labels, prediction):
                st.write(f"**{label}:** {float(score):.1%}")
                st.progress(float(np.clip(score, 0, 1)))

        st.warning(
            "Research output only. The model is not clinically validated."
        )

    except Exception as error:
        st.error(
            f"Fusion prediction failed: "
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# CAD / HEART DISEASE MODEL
# ============================================================

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
    "age": (1.0, 120.0, 50.0),
    "trestbps": (50.0, 300.0, 120.0),
    "chol": (50.0, 800.0, 200.0),
    "thalach": (30.0, 250.0, 150.0),
    "oldpeak": (0.0, 10.0, 1.0),
}


@st.cache_resource(show_spinner=False)
def load_cad_model(path_string, modified_time):
    artifact = joblib.load(path_string)

    if not isinstance(artifact, dict):
        raise ValueError("CAD model artifact must be a dictionary.")

    required = {"model", "feature_columns", "classes"}
    missing = required - set(artifact)

    if missing:
        raise ValueError(
            f"CAD artifact missing keys: {sorted(missing)}"
        )

    if not artifact["feature_columns"]:
        raise ValueError("CAD feature_columns is empty.")

    return artifact


def display_cad_prediction():
    st.subheader("❤️ CAD / Heart Disease")

    if not CAD_MODEL_PATH.is_file():
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

    model = artifact["model"]
    features = artifact["feature_columns"]
    numeric = artifact.get("numeric_columns", [])
    categorical = artifact.get("categorical_columns", [])

    values = {}

    with st.form("cad_clinical_form"):
        col1, col2 = st.columns(2)

        for index, feature in enumerate(features):
            with col1 if index % 2 == 0 else col2:
                if feature in CAD_CATEGORICAL_OPTIONS:
                    value = st.selectbox(
                        f"{feature} (dataset code)",
                        CAD_CATEGORICAL_OPTIONS[feature],
                        key=f"cad_{feature}",
                    )

                elif feature in CAD_NUMERIC_RANGES:
                    low, high, default = CAD_NUMERIC_RANGES[feature]
                    value = st.number_input(
                        feature,
                        min_value=low,
                        max_value=high,
                        value=default,
                        step=0.1 if feature == "oldpeak" else 1.0,
                        key=f"cad_{feature}",
                    )

                elif feature in numeric:
                    value = st.number_input(
                        feature,
                        value=0.0,
                        key=f"cad_{feature}",
                    )

                elif feature in categorical:
                    value = st.text_input(
                        f"{feature} (training category)",
                        key=f"cad_{feature}",
                    )

                else:
                    value = st.text_input(
                        f"{feature} (verify feature type)",
                        key=f"cad_{feature}",
                    )

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
            [[values[feature] for feature in features]],
            columns=features,
        )

        for feature in numeric:
            row[feature] = pd.to_numeric(
                row[feature],
                errors="raise",
            )

        for feature in categorical:
            if feature in row.columns:
                row[feature] = row[feature].astype(str)

        prediction = model.predict(row)[0]

        # Prefer the model's actual class label, not its integer index.
        st.metric("Predicted CAD Class", str(prediction))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            model_classes = getattr(
                model, "classes", getattr(model, "classes_", None)
            )

            if model_classes is None:
                model_classes = artifact["classes"]

            st.subheader("Class Probabilities")

            for label, probability in zip(
                model_classes, probabilities
            ):
                st.write(f"**{label}:** {float(probability):.1%}")
                st.progress(float(np.clip(probability, 0, 1)))

        st.warning(
            "Research prototype only. Not a confirmed diagnosis."
        )

    except Exception as error:
        st.error(f"CAD prediction failed: {error}")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👁️ Retina AI")
st.sidebar.write("Research Prediction Dashboard")
st.sidebar.caption(f"Device: {DEVICE}")

st.sidebar.subheader("Retinal Models")

for name, (filename, _) in MODEL_FILES.items():
    if (MODEL_DIR / filename).is_file():
        st.sidebar.success(f"{name}: Found")
    else:
        st.sidebar.error(f"{name}: Missing")

st.sidebar.subheader("Clinical Models")

for label, path in [
    ("CAD Model", CAD_MODEL_PATH),
    ("Fusion Risk Model", FUSION_MODEL_PATH),
    ("Clinical Scaler", CLINICAL_SCALER_PATH),
    ("Fusion Metadata", FUSION_METADATA_PATH),
    ("CAP Grade Model", CAP_MODEL_PATH),
]:
    if path.is_file():
        st.sidebar.success(f"{label}: Found")
    else:
        st.sidebar.info(f"{label}: Missing")


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("👁️ Retina AI Research Dashboard")
st.caption(
    "Upload a retinal image, capture a fundus image, "
    "or capture an external eye photo."
)
st.warning(
    "Research use only. This dashboard does not provide a "
    "confirmed medical diagnosis."
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


# ============================================================
# IMAGE DECODING
# ============================================================

image = None
image_key = None

try:
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(raw)).convert("RGB").copy()
        image_key = image_hash(raw)

    elif camera_photo is not None:
        raw = camera_photo.getvalue()
        image = Image.open(io.BytesIO(raw)).convert("RGB").copy()
        image_key = image_hash(raw)

except Exception as error:
    st.error(f"Could not open retinal image: {error}")


# Reset image-specific outputs when a new image is selected.
if image_key != st.session_state.last_retinal_image_key:
    st.session_state.retinal_results = {}
    st.session_state.fusion_result = None
    st.session_state.fusion_result_key = None
    st.session_state.last_retinal_image_key = image_key


# ============================================================
# RETINAL IMAGE PREVIEW AND PREDICTIONS
# ============================================================

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

        run_button = st.button(
            "🔍 Run Retinal Predictions",
            type="primary",
            use_container_width=True,
            key="run_retinal_predictions",
        )

    if run_button:
        results = {}

        st.divider()
        st.header("Retinal Prediction Results")

        for name, (filename, task) in MODEL_FILES.items():
            with st.expander(name, expanded=True):
                path = MODEL_DIR / filename

                try:
                    model = load_retinal_model(
                        str(path),
                        task,
                        path.stat().st_mtime,
                    )

                    result = run_retinal_prediction(
                        model,
                        image,
                        task,
                    )

                    results[name] = result
                    display_model_result(name, result)

                except Exception as error:
                    st.error(
                        f"{name} prediction failed: "
                        f"{type(error).__name__}: {error}"
                    )

        st.session_state.retinal_results = results

    elif st.session_state.retinal_results:
        st.header("Previous Retinal Prediction Results")

        for name, result in st.session_state.retinal_results.items():
            with st.expander(name):
                display_model_result(name, result)


# ============================================================
# RETINAL + CLINICAL FUSION
# ============================================================

st.divider()

if image is not None:
    display_fusion_prediction(image)
else:
    st.subheader("📊 Retinal + Clinical Fusion Risk Prediction")
    st.info("Upload or capture a retinal image to run fusion prediction.")


# ============================================================
# EXTERNAL EYE PHOTO
# ============================================================

if external_eye_photo is not None:
    st.divider()
    st.header("👁️ External Eye Photo")

    try:
        external_image = Image.open(
            io.BytesIO(external_eye_photo.getvalue())
        ).convert("RGB")

        st.image(
            external_image,
            caption="Captured External Eye Photo",
            use_container_width=True,
        )

        st.success("External eye photo captured successfully.")
        st.write(f"Width: {external_image.width} px")
        st.write(f"Height: {external_image.height} px")

        st.info(
            "External-eye prediction requires a separately trained "
            "model and is not enabled by the supplied model files."
        )

    except Exception as error:
        st.error(f"Could not open external eye photo: {error}")


# ============================================================
# CIMT CLASSIFICATION
# ============================================================

st.divider()
st.subheader("🩺 CIMT Classification")

cimt_value = st.session_state.retinal_results.get("CIMT Estimate")

if cimt_value is not None:
    st.metric(
        "Raw CIMT Model Output",
        f"{float(cimt_value):.4f}",
    )
    st.warning(
        "Clinical classification is disabled until the output units "
        "and validated cutoff are confirmed."
    )
else:
    st.info("Run retinal predictions to display the CIMT output.")


# ============================================================
# CAD / HEART DISEASE
# ============================================================

st.divider()
display_cad_prediction()


# ============================================================
# CAP GRADE
# ============================================================

st.divider()
st.subheader("🔬 CAP Grade")

if CAP_MODEL_PATH.is_file():
    st.warning(
        "CAP model file found, but its architecture, input preprocessing, "
        "and grade mapping are not configured. Prediction is disabled "
        "until these training details are supplied."
    )
else:
    st.info(
        "A compatible trained CAP model and verified grade labels "
        "are required."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Retina AI Research Dashboard | Research prototype only | "
    "Not validated for clinical diagnosis."
)

st.sidebar.caption(
    "Camera access may require browser permission and HTTPS."
)
