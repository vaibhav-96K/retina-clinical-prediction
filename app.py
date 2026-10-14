
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
div[data-testid="stFileUploader"],
div[data-testid="stCameraInput"] {
    background: white;
    padding: 10px;
    border-radius: 12px;
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

CAD_MODEL_FILE = "cad_heart_disease_model.joblib"

ADDITIONAL_MODELS = {
    "Risk Prediction": "risk_prediction_model.pth",
    "CAP Grade": "cap_grade_model.pth",
}

CAD_MODEL_PATH = MODEL_DIR / CAD_MODEL_FILE


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
# RETINAL MODEL LOADING
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
        raise FileNotFoundError(
            f"Model file missing: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = clean_state_dict(
            extract_state_dict(checkpoint)
        )

        default_outputs = (
            8 if task == "multilabel" else 1
        )

        output_size = infer_output_size(
            state,
            default_outputs,
        )

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
# CAD MODEL LOADING
# =====================================================

@st.cache_resource(show_spinner=False)
def load_cad_model(path, modified_time):
    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            "CAD artifact must be a dictionary."
        )

    required_keys = {
        "model",
        "feature_columns",
        "classes",
    }

    missing = required_keys - set(artifact.keys())

    if missing:
        raise ValueError(
            f"CAD artifact is missing keys: {sorted(missing)}"
        )

    if not artifact["feature_columns"]:
        raise ValueError(
            "CAD feature_columns is empty."
        )

    return artifact


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


# =====================================================
# CAD CLINICAL INPUT FORM
# =====================================================

def render_cad_input(
    feature_columns,
    numeric_columns,
    categorical_columns,
):
    values = {}

    st.write(
        "Enter clinical measurements using the units "
        "and category codes expected by the training dataset."
    )

    st.caption(
        "Form defaults are examples only. Enter verified "
        "measurements and the correct category codes."
    )

    with st.form("cad_clinical_form"):
        col1, col2 = st.columns(2)

        for index, feature in enumerate(feature_columns):
            container = col1 if index % 2 == 0 else col2

            with container:
                if feature in CAD_CATEGORICAL_OPTIONS:
                    value = st.selectbox(
                        f"{feature} (dataset code)",
                        options=CAD_CATEGORICAL_OPTIONS[feature],
                        key=f"cad_{feature}",
                    )

                elif feature in CAD_NUMERIC_RANGES:
                    minimum, maximum, default = (
                        CAD_NUMERIC_RANGES[feature]
                    )

                    value = st.number_input(
                        feature,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(default),
                        step=(
                            0.1 if feature == "oldpeak"
                            else 1.0
                        ),
                        key=f"cad_{feature}",
                    )

                elif feature in numeric_columns:
                    value = st.number_input(
                        feature,
                        value=0.0,
                        key=f"cad_{feature}",
                    )

                elif feature in categorical_columns:
                    value = st.text_input(
                        f"{feature} (category)",
                        key=f"cad_{feature}",
                    ).strip()

                    if value == "":
                        st.warning(
                            f"Enter a category value for {feature}."
                        )

                else:
                    value = st.text_input(
                        feature,
                        key=f"cad_{feature}",
                    ).strip()

            values[feature] = value

        submitted = st.form_submit_button(
            "Run CAD Prediction",
            type="primary",
            use_container_width=True,
        )

    return submitted, values


def display_cad_prediction():
    st.subheader("❤️ CAD / Heart Disease")

    if not CAD_MODEL_PATH.exists():
        st.info(
            f"CAD model is not installed. Add "
            f"models/{CAD_MODEL_FILE} to the repository."
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

    feature_columns = artifact["feature_columns"]
    numeric_columns = artifact.get("numeric_columns", [])
    categorical_columns = artifact.get(
        "categorical_columns", []
    )

    st.caption(
        f"Loaded clinical model with "
        f"{len(feature_columns)} input features."
    )

    submitted, values = render_cad_input(
        feature_columns,
        numeric_columns,
        categorical_columns,
    )

    if not submitted:
        st.caption(
            "Complete the form and select Run CAD Prediction."
        )
        return

    try:
        row = pd.DataFrame(
            [[values[col] for col in feature_columns]],
            columns=feature_columns,
        )

        for col in numeric_columns:
            row[col] = pd.to_numeric(
                row[col],
                errors="raise",
            )

        model = artifact["model"]
        prediction = int(model.predict(row)[0])
        classes = artifact["classes"]

        if prediction < 0 or prediction >= len(classes):
            raise ValueError(
                "Prediction does not match saved class mapping."
            )

        st.markdown("### Model Output")
        st.write(
            f"**Predicted class:** {classes[prediction]}"
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]

            if len(probabilities) == len(classes):
                st.write("Class scores:")

                for label, score in zip(
                    classes,
                    probabilities,
                ):
                    st.write(
                        f"- {label}: {float(score):.1%}"
                    )

        st.warning(
            "Research prototype only. These scores are not "
            "validated clinical probabilities or a diagnosis."
        )

    except Exception as error:
        st.error(f"CAD prediction failed: {error}")


# =====================================================
# RETINAL PREDICTION HELPERS
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

    values = (
        output.detach()
        .float()
        .cpu()
        .reshape(-1)
        .numpy()
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Model returned invalid numeric values."
        )

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
                1.0 + np.exp(
                    -np.clip(values[0], -50, 50)
                )
            )
        )

        return {
            "Class 0": 1.0 - probability,
            "Class 1": probability,
        }

    shifted = values - np.max(values)
    exp_values = np.exp(
        np.clip(shifted, -50, 50)
    )
    probabilities = exp_values / exp_values.sum()

    return {
        f"Class {index}": float(probability)
        for index, probability in enumerate(probabilities)
    }


def display_sex(result):
    if not isinstance(result, dict) or len(result) != 2:
        st.error(
            "Sex model did not return two class scores."
        )
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
        "Confirm Female/Male class order against training labels."
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

    diabetes_score = float(
        np.clip(scores[1], 0, 1)
    )

    status = (
        "Positive"
        if diabetes_score >= DIABETES_THRESHOLD
        else "Negative"
    )

    st.metric("Diabetes Model Result", status)
    st.write(
        f"Model score: {diabetes_score * 100:.1f}%"
    )
    st.progress(diabetes_score)

    st.caption(
        "Experimental output. Verify label order and threshold "
        "against the model's training configuration."
    )

    with st.expander("All Retinal Disease Scores"):
        for label, score in zip(DISEASE_LABELS, scores):
            score = float(np.clip(score, 0, 1))
            st.write(
                f"**{label}:** {score * 100:.1f}%"
            )
            st.progress(score)


def display_model_result(name, result):
    st.subheader(name)

    if name == "Retinal Age":
        st.metric(
            "Estimated Age",
            f"{float(result):.1f} years",
        )

    elif name == "Retinal Sex":
        display_sex(result)

    elif name == "Retinal Disease":
        display_disease(result)

    elif name == "CIMT Estimate":
        st.metric(
            "Raw CIMT Model Output",
            f"{float(result):.4f}",
        )

        st.caption(
            "Units have not been verified. Classification "
            "is disabled until units and a validated cutoff "
            "are confirmed."
        )


# =====================================================
# ADDITIONAL PREDICTIONS
# =====================================================

def display_other_additional_predictions():
    risk_col, cap_col = st.columns(2)

    with risk_col:
        st.subheader("📊 Risk Prediction")
        risk_path = (
            MODEL_DIR / ADDITIONAL_MODELS["Risk Prediction"]
        )

        if risk_path.exists():
            st.warning(
                f"{risk_path.name} exists, but its model format, "
                "inputs and output mapping are not configured."
            )
        else:
            st.info(
                "A compatible trained risk model and its required "
                "inputs are needed."
            )

        st.caption("No risk category is invented.")

    with cap_col:
        st.subheader("🔬 CAP Grade")
        cap_path = (
            MODEL_DIR / ADDITIONAL_MODELS["CAP Grade"]
        )

        if cap_path.exists():
            st.warning(
                f"{cap_path.name} exists, but its model format "
                "and grade mapping are not configured."
            )
        else:
            st.info(
                "A compatible trained CAP model and verified "
                "grade labels are needed."
            )

        st.caption(
            "No CAP grade is generated without a compatible model."
        )


# =====================================================
# SESSION STATE
# =====================================================

if "retinal_results" not in st.session_state:
    st.session_state.retinal_results = {}

if "last_retinal_image_key" not in st.session_state:
    st.session_state.last_retinal_image_key = None


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

for name, filename in ADDITIONAL_MODELS.items():
    if (MODEL_DIR / filename).exists():
        st.sidebar.success(f"{name}: File found")
    else:
        st.sidebar.info(f"{name}: Not found")


# =====================================================
# MAIN DASHBOARD
# =====================================================

st.title("👁️ Retina AI Research Dashboard")

st.caption(
    "Upload a retinal image, capture a retinal image, "
    "or capture an external eye photo."
)

st.warning(
    "For research use only. Outputs do not provide a "
    "confirmed medical diagnosis."
)


# =====================================================
# THREE SIDE-BY-SIDE IMAGE OPTIONS
# =====================================================

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


# =====================================================
# LOAD SELECTED RETINAL IMAGE
# Upload takes priority when both are present.
# =====================================================

image = None
image_key = None

try:
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(
            uploaded_file
        ).convert("RGB")
        image_key = str(hash(image_bytes))

    elif camera_photo is not None:
        camera_bytes = camera_photo.getvalue()
        image = Image.open(
            camera_photo
        ).convert("RGB")
        image_key = str(hash(camera_bytes))

except Exception as error:
    st.error(
        f"Could not open retinal image: {error}"
    )


# =====================================================
# RETINAL IMAGE PREVIEW AND PREDICTIONS
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
            st.caption(
                "A new image is selected. Run predictions to "
                "calculate results for this image."
            )

        run_button = st.button(
            "🔍 Run Retinal Predictions",
            type="primary",
            use_container_width=True,
            key="run_retinal_predictions",
        )

    if run_button:
        prediction_results = {}

        st.session_state.retinal_results = {}
        st.session_state.last_retinal_image_key = image_key

        st.divider()
        st.header("Retinal Prediction Results")

        for model_name, (filename, task) in MODEL_FILES.items():
            with st.expander(
                model_name,
                expanded=True,
            ):
                try:
                    with st.spinner(
                        f"Running {model_name}..."
                    ):
                        model = load_model(
                            str(MODEL_DIR / filename),
                            task,
                        )

                        result = run_prediction(
                            model,
                            image,
                            task,
                        )

                    prediction_results[model_name] = result
                    display_model_result(
                        model_name,
                        result,
                    )

                except Exception as error:
                    st.error(
                        f"{model_name} failed: {error}"
                    )

        st.session_state.retinal_results = prediction_results

elif st.session_state.retinal_results:
    st.divider()
    st.header("Previous Retinal Prediction Results")

    st.caption(
        "These are the results from the previous prediction run."
    )

    for name, result in st.session_state.retinal_results.items():
        with st.expander(name):
            display_model_result(name, result)


# =====================================================
# EXTERNAL EYE PHOTO PREVIEW
# This is not a retinal/fundus image.
# =====================================================

if external_eye_photo is not None:
    st.divider()
    st.header("👁️ External Eye Photo")

    try:
        external_eye_image = Image.open(
            external_eye_photo
        ).convert("RGB")

        st.image(
            external_eye_image,
            caption="Captured External Eye Photo",
            use_container_width=True,
        )

        st.success(
            "External eye photo captured successfully."
        )

        with st.expander("Image Information"):
            st.write(
                f"Width: {external_eye_image.width} px"
            )
            st.write(
                f"Height: {external_eye_image.height} px"
            )

        st.info(
            "External-eye AI prediction is not enabled. "
            "A separately trained and validated model for "
            "external-eye photographs is required. The existing "
            "retinal models are not assumed to work on these images."
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
        f"{float(cimt_value):.4f}",
    )

    st.warning(
        "Units and a validated cutoff are unknown. True/False "
        "classification is disabled until the model units and "
        "clinical threshold are verified."
    )
else:
    st.info(
        "Run retinal predictions to display the raw CIMT output."
    )


# =====================================================
# CAD PREDICTION
# =====================================================

st.divider()
display_cad_prediction()


# =====================================================
# OTHER ADDITIONAL PREDICTIONS
# =====================================================

st.divider()
st.header("Additional Predictions")

display_other_additional_predictions()


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Research prototype only. Not validated for clinical diagnosis."
)

st.sidebar.caption(
    "Camera access requires browser permission and HTTPS "
    "when deployed."
)
