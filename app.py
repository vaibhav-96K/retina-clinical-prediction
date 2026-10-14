import io
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# =====================================================
# OPTIONAL SHAP AND PDF SUPPORT
# =====================================================

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    PDF_OK = True
except Exception:
    PDF_OK = False


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Retina AI Research Dashboard",
    page_icon="👁️",
    layout="wide",
)

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


# =====================================================
# CONFIGURATION AND PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

# Risk model was saved in this Drive folder.
# Change this path if you move the model file.
RISK_MODEL_PATH = Path(
    "/content/drive/MyDrive/retinal and clinical project 1/"
    "risk_prediction_model.pth"
)

CAD_MODEL_FILE = "cad_heart_disease_model.joblib"
CAD_MODEL_PATH = MODEL_DIR / CAD_MODEL_FILE

CAP_MODEL_FILE = "cap_grade_model.pth"

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
# IMAGE TRANSFORMATION
# =====================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def image_to_tensor(image):
    return (
        TRANSFORM(image.convert("RGB"))
        .unsqueeze(0)
        .to(DEVICE)
    )


# =====================================================
# GENERIC CHECKPOINT HELPERS
# =====================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
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


def safe_torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


# =====================================================
# EXISTING RETINAL MODELS
# =====================================================

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
def load_retinal_model(path, task, modified_time):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file missing: {path}"
        )

    checkpoint = safe_torch_load(path)

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


def run_retinal_prediction(model, image, task):
    tensor = image_to_tensor(image)

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
        # Avoid applying sigmoid twice if output is already probabilities.
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
    exp_values = np.exp(np.clip(shifted, -50, 50))
    probabilities = exp_values / exp_values.sum()

    return {
        f"Class {index}": float(probability)
        for index, probability in enumerate(probabilities)
    }


# =====================================================
# RISK PREDICTION MODEL
# Architecture matches the shown checkpoint keys:
# image_encoder.features...
# clinical_encoder.0...
# head.0...
# head.3...
# =====================================================

class RiskPredictionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = models.efficientnet_b0(
            weights=None
        )

        # The training checkpoint has a 1280-feature image encoder.
        self.image_encoder.classifier = nn.Identity()

        self.clinical_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(1296, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, image, clinical):
        image_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical)

        combined = torch.cat(
            [image_features, clinical_features],
            dim=1,
        )

        return self.head(combined)


@st.cache_resource(show_spinner=False)
def load_risk_model(path, modified_time):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Risk model file missing: {path}"
        )

    checkpoint = safe_torch_load(path)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = extract_state_dict(checkpoint)

        # Some checkpoints store the best validation weights separately.
        if isinstance(checkpoint, dict):
            best_state = checkpoint.get("best_state")
            if isinstance(best_state, dict):
                if all(
                    torch.is_tensor(value)
                    for value in best_state.values()
                ):
                    state = best_state

        state = clean_state_dict(state)

        model = RiskPredictionNet()
        model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


def display_risk_prediction():
    st.subheader("📊 Retinal Risk Prediction")

    st.warning(
        "Research prototype only. This model predicts the dataset's "
        "binary label; it is not a validated heart-disease risk score "
        "or a medical diagnosis."
    )

    if not RISK_MODEL_PATH.exists():
        st.error(
            "Risk model file not found. Expected path:\n\n"
            f"{RISK_MODEL_PATH}"
        )
        st.info(
            "Copy risk_prediction_model.pth into the path above, "
            "or update RISK_MODEL_PATH in this code."
        )
        return

    st.caption(
        "Enter age and gender using the same encoding as the training dataset."
    )

    age = st.number_input(
        "Age in years",
        min_value=1,
        max_value=120,
        value=50,
        step=1,
        key="risk_age",
    )

    gender = st.selectbox(
        "Gender code used during training",
        options=[0, 1],
        key="risk_gender",
        help=(
            "The meanings of codes 0 and 1 must match the training dataset. "
            "Do not assume which code means female or male."
        ),
    )

    uploaded = st.file_uploader(
        "Upload fundus / retinal image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="risk_image",
    )

    if uploaded is None:
        st.info("Upload a retinal image to run the model.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(
            image,
            caption="Image used for risk-model inference",
            use_container_width=True,
        )
    except Exception as error:
        st.error(f"Could not open image: {error}")
        return

    if st.button(
        "Run Risk Prediction",
        type="primary",
        key="run_risk_prediction",
    ):
        try:
            model = load_risk_model(
                str(RISK_MODEL_PATH),
                RISK_MODEL_PATH.stat().st_mtime,
            )

            image_tensor = image_to_tensor(image)

            # This assumes training used raw age_years and gender values.
            # If training normalized them, this must be changed accordingly.
            clinical_tensor = torch.tensor(
                [[float(age), float(gender)]],
                dtype=torch.float32,
                device=DEVICE,
            )

            with torch.inference_mode():
                logits = model(
                    image_tensor,
                    clinical_tensor,
                )

                scores = torch.softmax(
                    logits,
                    dim=1,
                )[0].detach().cpu().numpy()

            if len(scores) != 2 or not np.all(np.isfinite(scores)):
                raise ValueError(
                    "Risk model returned unexpected output."
                )

            predicted_class = int(np.argmax(scores))

            st.markdown("### Model output")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted dataset class",
                str(predicted_class),
            )
            col2.metric(
                "Class 0 score",
                f"{scores[0] * 100:.1f}%",
            )
            col3.metric(
                "Class 1 score",
                f"{scores[1] * 100:.1f}%",
            )

            st.progress(float(scores[predicted_class]))

            st.caption(
                "These are model scores, not calibrated clinical probabilities. "
                "The meanings of labels 0 and 1 must be verified from the dataset "
                "before interpreting the result."
            )

        except Exception as error:
            st.error(
                f"Risk prediction failed: {error}"
            )
            st.caption(
                "If loading failed, verify the model architecture and checkpoint "
                "format against the training notebook."
            )


# =====================================================
# CAD / CLINICAL MODEL
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


def render_cad_input(
    feature_columns,
    numeric_columns,
    categorical_columns,
):
    values = {}

    st.write(
        "Enter clinical measurements using the units and category codes "
        "expected by the model's training dataset."
    )

    st.caption(
        "Default values are examples only. Enter verified measurements."
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
            "CAD clinical model not installed. Add "
            f"{CAD_MODEL_FILE} to the app's models folder."
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
        f"Loaded clinical model with {len(feature_columns)} input features."
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

        st.markdown("### Model output")
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
            "Research prototype only. These scores are not validated "
            "clinical probabilities or a diagnosis."
        )

    except Exception as error:
        st.error(f"CAD prediction failed: {error}")


# =====================================================
# RETINAL RESULT DISPLAY
# =====================================================

def display_sex(result):
    if not isinstance(result, dict) or len(result) != 2:
        st.error(
            "Sex model did not return two class scores."
        )
        return

    scores = list(result.values())
    predicted_index = int(np.argmax(scores))

    st.metric(
        "Predicted Sex Class",
        str(predicted_index),
    )

    st.caption(
        "Confirm the Female/Male class order against the training labels."
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

    st.metric(
        "Diabetes Model Result",
        status,
    )
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


def display_retinal_model_result(name, result):
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
            "Units have not been verified. A clinical category is disabled "
            "until the model units and a validated cutoff are confirmed."
        )


# =====================================================
# PDF REPORT
# =====================================================

def create_pdf_report(results, filename):
    if not PDF_OK:
        raise RuntimeError(
            "Install reportlab to generate PDF reports."
        )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "Retina AI Research Report",
            styles["Title"],
        ),
        Spacer(1, 8),
        Paragraph(
            f"Image: {filename}",
            styles["Normal"],
        ),
        Paragraph(
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            styles["Normal"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Research use only. Outputs may be uncalibrated and must not "
            "be used as a diagnosis or for clinical decisions.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    rows = [["Model", "Output", "Value"]]

    for model_name, values in results.items():
        if isinstance(values, dict):
            for label, value in values.items():
                rows.append([
                    model_name,
                    label,
                    (
                        f"{value:.4f}"
                        if isinstance(value, (float, np.floating))
                        else str(value)
                    ),
                ])
        elif isinstance(values, (list, tuple, np.ndarray)):
            for index, value in enumerate(values):
                rows.append([
                    model_name,
                    f"Output {index}",
                    f"{float(value):.4f}",
                ])
        else:
            rows.append([
                model_name,
                "Prediction",
                str(values),
            ])

    table = Table(
        rows,
        repeatRows=1,
        colWidths=[100, 280, 130],
    )

    table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#263746"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white,
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.4,
            colors.grey,
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP",
        ),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#f2f5f8")],
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8,
        ),
    ]))

    story.append(table)
    document.build(story)

    return buffer.getvalue()


# =====================================================
# SESSION STATE
# =====================================================

if "retinal_results" not in st.session_state:
    st.session_state.retinal_results = {}

if "retinal_image_key" not in st.session_state:
    st.session_state.retinal_image_key = None


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

st.sidebar.subheader("Risk Model")

if RISK_MODEL_PATH.exists():
    st.sidebar.success("Risk Prediction: Found")
else:
    st.sidebar.error("Risk Prediction: Missing")

st.sidebar.subheader("CAD Model")

if CAD_MODEL_PATH.exists():
    st.sidebar.success("CAD model: Found")
else:
    st.sidebar.info("CAD model: Not found")


# =====================================================
# MAIN DASHBOARD
# =====================================================

st.title("👁️ Retina AI Research Dashboard")

st.caption(
    "Retinal image analysis, experimental risk prediction, "
    "and clinical-model research tools."
)

st.warning(
    "For research use only. Outputs are not confirmed diagnoses "
    "and should not be used alone for medical decisions."
)


# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Retinal Predictions",
    "📊 Risk Prediction",
    "❤️ CAD / Heart Disease",
    "ℹ️ Model Info",
])


# =====================================================
# TAB 1: RETINAL PREDICTIONS
# =====================================================

with tab1:
    upload_col, camera_col = st.columns(2)

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload fundus / retinal image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp",
                "tif",
                "tiff",
            ],
            key="main_retinal_upload",
        )

    with camera_col:
        camera_photo = st.camera_input(
            "Capture image",
            key="main_retinal_camera",
        )

    image = None
    image_key = None
    filename = "retinal_image"

    try:
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")
            image_key = str(hash(image_bytes))
            filename = uploaded_file.name

        elif camera_photo is not None:
            image_bytes = camera_photo.getvalue()
            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")
            image_key = str(hash(image_bytes))
            filename = "camera_capture.jpg"

    except Exception as error:
        st.error(f"Could not open image: {error}")

    if image is not None:
        st.image(
            image,
            caption="Selected retinal image",
            use_container_width=True,
        )

        st.write(
            f"Image dimensions: {image.width} × {image.height} pixels"
        )

        if st.button(
            "🔍 Run Retinal Predictions",
            type="primary",
            key="run_all_retinal",
        ):
            results = {}

            st.session_state.retinal_results = {}
            st.session_state.retinal_image_key = image_key

            progress = st.progress(0)

            for index, (name, (filename_model, task)) in enumerate(
                MODEL_FILES.items()
            ):
                model_path = MODEL_DIR / filename_model

                try:
                    with st.spinner(f"Running {name}..."):
                        model = load_retinal_model(
                            str(model_path),
                            task,
                            (
                                model_path.stat().st_mtime
                                if model_path.exists()
                                else 0
                            ),
                        )

                        result = run_retinal_prediction(
                            model,
                            image,
                            task,
                        )

                        results[name] = result

                    st.success(f"{name} completed.")

                except Exception as error:
                    st.error(f"{name} failed: {error}")

                progress.progress(
                    (index + 1) / len(MODEL_FILES)
                )

            st.session_state.retinal_results = results

            if results:
                st.subheader("Retinal Results")

                for name, result in results.items():
                    with st.expander(name, expanded=True):
                        display_retinal_model_result(
                            name,
                            result,
                        )

                rows = []

                for model_name, result in results.items():
                    if isinstance(result, dict):
                        for label, value in result.items():
                            rows.append({
                                "model": model_name,
                                "output": label,
                                "value": value,
                            })
                    elif isinstance(
                        result,
                        (list, tuple, np.ndarray),
                    ):
                        for index, value in enumerate(result):
                            rows.append({
                                "model": model_name,
                                "output": f"Output {index}",
                                "value": float(value),
                            })
                    else:
                        rows.append({
                            "model": model_name,
                            "output": "Prediction",
                            "value": result,
                        })

                st.download_button(
                    "⬇️ Download CSV report",
                    data=pd.DataFrame(rows).to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="retina_ai_report.csv",
                    mime="text/csv",
                )

                if PDF_OK:
                    try:
                        pdf_data = create_pdf_report(
                            results,
                            filename,
                        )

                        st.download_button(
                            "⬇️ Download PDF report",
                            data=pdf_data,
                            file_name="retina_ai_report.pdf",
                            mime="application/pdf",
                        )
                    except Exception as error:
                        st.warning(
                            f"PDF export issue: {error}"
                        )
                else:
                    st.info(
                        "Install reportlab to enable PDF export."
                    )

    elif st.session_state.retinal_results:
        st.info(
            "Previous retinal results are available for this session."
        )

        for name, result in st.session_state.retinal_results.items():
            with st.expander(name):
                display_retinal_model_result(
                    name,
                    result,
                )


# =====================================================
# TAB 2: RISK PREDICTION
# =====================================================

with tab2:
    display_risk_prediction()


# =====================================================
# TAB 3: CAD / HEART DISEASE
# =====================================================

with tab3:
    display_cad_prediction()


# =====================================================
# TAB 4: MODEL INFORMATION
# =====================================================

with tab4:
    st.subheader("Model Files")

    for name, (filename, task) in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        st.write(f"**{name}**")
        st.write(f"Task: `{task}`")
        st.write(f"Path: `{model_path}`")
        st.write(
            "Status: "
            + ("Found" if model_path.exists() else "Missing")
        )
        st.divider()

    st.write("**Risk Prediction**")
    st.write(f"Path: `{RISK_MODEL_PATH}`")
    st.write(
        "Status: "
        + (
            "Found"
            if RISK_MODEL_PATH.exists()
            else "Missing"
        )
    )

    st.warning(
        "The risk model was trained on a binary dataset label. "
        "Verify what label 0 and label 1 mean, check patient-level split "
        "separation, and validate preprocessing and performance before "
        "interpreting results clinically."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Retina AI Research Prototype — not validated for clinical diagnosis."
)

st.caption(
    f"Runtime device: {DEVICE}"
)
