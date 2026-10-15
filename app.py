
# app.py
# Retina AI - Research Prediction Dashboard

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Retina AI | Research Prediction Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

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

CAD_MODEL_PATH = MODEL_DIR / "cad_heart_disease_model.joblib"
FUSION_MODEL_PATH = MODEL_DIR / "retinal_feature_fusion_model_FINAL.keras"
CLINICAL_SCALER_PATH = MODEL_DIR / "clinical_scaler.pkl"
FUSION_METADATA_PATH = MODEL_DIR / "fusion_metadata.json"


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 750;
        color: #172554;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        color: #1e293b;
        font-size: 1.4rem;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

DEFAULT_STATE = {
    "retinal_results": None,
    "retinal_result_key": None,
    "cad_prediction_result": None,
    "last_image_key": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">👁️ Retina AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Research Prediction Dashboard</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Research prototype for retinal image analysis and clinical "
    "heart-disease classification."
)

st.divider()


# ---------------------------------------------------------
# DEVICE INFORMATION
# ---------------------------------------------------------

try:
    import torch

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
except Exception:
    torch = None
    DEVICE = "cpu"


# ---------------------------------------------------------
# LOAD CLINICAL HEART DISEASE MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_cad_artifact(model_path, modified_time):
    artifact = joblib.load(model_path)

    # Support the saved dictionary format and a plain estimator.
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {
        "model": artifact,
        "feature_columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal",
        ],
        "classes": ["No disease", "Disease present"],
        "test_accuracy": None,
        "test_roc_auc": None,
        "dataset": "Unknown",
        "warning": "Research prototype; not clinically validated.",
    }


# ---------------------------------------------------------
# HEART DISEASE RISK SECTION
# ---------------------------------------------------------

def heart_disease_section():
    st.header("❤️ Heart Disease Risk")
    st.write(
        "Enter the patient's clinical values to run the saved "
        "clinical model. The retinal image is not used by this model."
    )

    if not CAD_MODEL_PATH.exists():
        st.error(
            "Clinical model file was not found.\n\n"
            f"Expected path: `{CAD_MODEL_PATH}`\n\n"
            "Place cad_heart_disease_model.joblib inside the models folder."
        )
        return

    try:
        artifact = load_cad_artifact(
            str(CAD_MODEL_PATH),
            CAD_MODEL_PATH.stat().st_mtime,
        )
        model = artifact["model"]
        feature_columns = artifact.get(
            "feature_columns",
            [
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak",
                "slope", "ca", "thal",
            ],
        )
    except Exception as error:
        st.error(
            f"Could not load the clinical model: "
            f"{type(error).__name__}: {error}"
        )
        return

    with st.form("cad_prediction_form"):
        st.subheader("Patient Clinical Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age (years)", min_value=1, max_value=120, value=45
            )
            sex = st.selectbox(
                "Sex", options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
            )
            cp = st.selectbox(
                "Chest pain type",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Type 1",
                    2: "Type 2",
                    3: "Type 3",
                    4: "Type 4",
                }[x],
            )
            trestbps = st.number_input(
                "Resting blood pressure",
                min_value=50, max_value=250, value=120,
            )
            chol = st.number_input(
                "Cholesterol",
                min_value=50, max_value=700, value=200,
            )

        with c2:
            fbs = st.selectbox(
                "Fasting blood sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
            )
            restecg = st.selectbox(
                "Resting ECG", options=[0, 1, 2]
            )
            thalach = st.number_input(
                "Maximum heart rate achieved",
                min_value=40, max_value=250, value=150,
            )
            exang = st.selectbox(
                "Exercise-induced angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
            )

        with c3:
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0, max_value=10.0,
                value=1.0, step=0.1,
            )
            slope = st.selectbox(
                "Slope of peak exercise ST segment",
                options=[1, 2, 3],
            )
            ca = st.selectbox(
                "Number of major vessels (ca)",
                options=[0, 1, 2, 3],
            )
            thal = st.selectbox(
                "Thal", options=[3, 6, 7],
                format_func=lambda x: {
                    3: "3",
                    6: "6",
                    7: "7",
                }[x],
            )

        submitted = st.form_submit_button(
            "❤️ Run Heart Disease Risk Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        patient = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        try:
            patient_df = pd.DataFrame([patient])

            # Keep the exact feature order used during training.
            patient_df = patient_df.reindex(
                columns=feature_columns
            )

            prediction = int(model.predict(patient_df)[0])

            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(patient_df)[0]
                classes = list(getattr(model, "classes_", [0, 1]))

                positive_indices = [
                    i for i, cls in enumerate(classes)
                    if str(cls) in ("1", "1.0", "True")
                ]

                if positive_indices:
                    probability = float(
                        probabilities[positive_indices[0]]
                    )
                elif len(probabilities) > 1:
                    probability = float(probabilities[-1])

            st.session_state["cad_prediction_result"] = {
                "prediction": prediction,
                "probability": probability,
                "patient": patient,
            }

        except Exception as error:
            st.error(
                f"Prediction failed: {type(error).__name__}: {error}"
            )

    result = st.session_state.get("cad_prediction_result")

    if result is not None:
        st.divider()
        st.subheader("Clinical Model Prediction")

        prediction = result["prediction"]
        probability = result["probability"]

        if prediction == 1:
            st.warning(
                "Model output: Disease present"
            )
        else:
            st.success(
                "Model output: No disease detected by the model"
            )

        if probability is not None:
            st.metric(
                "Model-estimated positive-class probability",
                f"{probability * 100:.2f}%",
            )
            st.progress(
                max(0.0, min(1.0, probability))
            )

        metric1, metric2 = st.columns(2)

        accuracy = artifact.get("test_accuracy")
        auc = artifact.get("test_roc_auc")

        with metric1:
            if accuracy is not None:
                st.metric(
                    "Saved test accuracy",
                    f"{float(accuracy) * 100:.2f}%",
                )
            else:
                st.metric("Saved test accuracy", "Not available")

        with metric2:
            if auc is not None:
                st.metric(
                    "Saved test ROC-AUC",
                    f"{float(auc):.4f}",
                )
            else:
                st.metric("Saved test ROC-AUC", "Not available")

        st.caption(
            f"Training dataset: {artifact.get('dataset', 'Not specified')}"
        )
        st.warning(
            "This is a research model trained on a public dataset. "
            "Its output is not a clinically validated personal risk "
            "estimate and must not be used for diagnosis or treatment."
        )


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

def prepare_image(uploaded_file):
    """Open an uploaded image safely and return an RGB PIL image."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def get_image_key(uploaded_file):
    """Create a stable identifier for the currently uploaded image."""
    import hashlib

    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)

    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------
# LOAD RETINAL MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_retinal_model(model_path, task, modified_time):
    """
    Load a complete PyTorch model or TorchScript model.

    A state_dict-only checkpoint needs its original model architecture.
    This function intentionally raises a clear error rather than
    guessing the architecture and producing misleading predictions.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # First try TorchScript.
    try:
        model = torch.jit.load(
            str(path), map_location=DEVICE
        )
        model.eval()
        return model
    except Exception:
        pass

    # Then try a saved complete nn.Module.
    try:
        checkpoint = torch.load(
            str(path),
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            str(path), map_location=DEVICE
        )

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.to(DEVICE)
        checkpoint.eval()
        return checkpoint

    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), torch.nn.Module):
            model = checkpoint["model"]
            model.to(DEVICE)
            model.eval()
            return model

        raise RuntimeError(
            "This .pth file appears to contain model weights only "
            "(state_dict), not a complete model. The original model "
            "architecture used during training is required to load it. "
            "Use the model architecture from the training notebook."
        )

    raise RuntimeError(
        f"Unsupported checkpoint format: {type(checkpoint).__name__}"
    )


# ---------------------------------------------------------
# RETINAL MODEL PREDICTION
# ---------------------------------------------------------

def retinal_predict(model, image, task):
    """
    Run inference for a complete PyTorch model.

    Assumes a 224x224 RGB input and ImageNet normalization.
    Verify these preprocessing settings against the original training
    notebook before relying on the predictions.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
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

    if not torch.is_tensor(output):
        output = torch.as_tensor(output)

    output = output.detach().cpu().float().numpy()
    output = np.asarray(output).squeeze()

    if task == "regression":
        return {
            "task": task,
            "value": float(np.asarray(output).reshape(-1)[0]),
        }

    if task == "classification":
        values = np.asarray(output).reshape(-1)

        if len(values) == 1:
            probability = float(1 / (1 + np.exp(-values[0])))
            index = int(probability >= 0.5)
        else:
            # If the output is already probabilities, preserve them;
            # otherwise apply softmax.
            if (
                np.all(values >= 0)
                and np.all(values <= 1)
                and np.isclose(values.sum(), 1.0, atol=0.02)
            ):
                probabilities = values
            else:
                shifted = values - np.max(values)
                exp_values = np.exp(shifted)
                probabilities = exp_values / exp_values.sum()

            index = int(np.argmax(probabilities))
            probability = float(probabilities[index])

        if len(values) == 2:
            label = SEX_CLASS_LABELS[index]
        else:
            label = f"Class {index}"

        return {
            "task": task,
            "class_index": index,
            "label": label,
            "confidence": probability,
        }

    if task == "multilabel":
        values = np.asarray(output).reshape(-1)

        # Apply sigmoid if outputs do not look like probabilities.
        if not (
            np.all(values >= 0)
            and np.all(values <= 1)
        ):
            values = 1 / (1 + np.exp(-values))

        labels = []
        for i, score in enumerate(values):
            if score >= 0.5:
                label = (
                    DISEASE_LABELS[i]
                    if i < len(DISEASE_LABELS)
                    else f"Label {i}"
                )
                labels.append({
                    "label": label,
                    "score": float(score),
                })

        return {
            "task": task,
            "scores": values.tolist(),
            "labels": labels,
        }

    raise ValueError(f"Unknown retinal task: {task}")


# ---------------------------------------------------------
# DISPLAY RETINAL RESULTS
# ---------------------------------------------------------

def show_retinal_results(results):
    st.subheader("👁️ Retinal Prediction Results")

    if not results:
        st.info("No retinal prediction results are available.")
        return

    for name, result in results.items():
        with st.container(border=True):
            st.markdown(f"### {name}")

            if isinstance(result, dict):
                task = result.get("task")

                if task == "regression":
                    st.metric(
                        "Predicted value",
                        f"{result['value']:.4f}",
                    )

                elif task == "classification":
                    st.write(
                        f"**Predicted class:** {result['label']}"
                    )
                    st.metric(
                        "Model confidence",
                        f"{result['confidence'] * 100:.2f}%",
                    )

                elif task == "multilabel":
                    labels = result.get("labels", [])

                    if labels:
                        for item in labels:
                            st.write(
                                f"**{item['label']}** — "
                                f"{item['score'] * 100:.2f}%"
                            )
                    else:
                        st.info(
                            "No labels crossed the 50% threshold. "
                            "This does not prove that the image is normal."
                        )

                    scores = result.get("scores", [])
                    if scores:
                        labels = [
                            DISEASE_LABELS[i]
                            if i < len(DISEASE_LABELS)
                            else f"Label {i}"
                            for i in range(len(scores))
                        ]

                        st.dataframe(
                            pd.DataFrame({
                                "Label": labels,
                                "Score": [
                                    round(float(x), 4)
                                    for x in scores
                                ],
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

                else:
                    st.json(result)
            else:
                st.write(result)


# ---------------------------------------------------------
# GRAD-CAM PLACEHOLDER
# ---------------------------------------------------------

def show_gradcam(image, image_key):
    """
    Grad-CAM depends on the exact CNN architecture and target layer.
    It is not fabricated here if those details are unavailable.
    """
    st.subheader("🔥 Grad-CAM Explainability")

    st.info(
        "Grad-CAM requires the original CNN architecture and its target "
        "convolutional layer. Configure those from the retinal model's "
        "training notebook before generating a valid heatmap."
    )


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

st.header("📤 Upload Retinal / Fundus Image")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="Upload a retinal fundus image in a supported image format.",
)

image = None
image_key = None

if uploaded_file is not None:
    try:
        image = prepare_image(uploaded_file)
        image_key = get_image_key(uploaded_file)

        # Clear results when the uploaded image changes.
        if st.session_state["last_image_key"] != image_key:
            st.session_state["retinal_results"] = None
            st.session_state["retinal_result_key"] = None
            st.session_state["cad_prediction_result"] = None
            st.session_state["last_image_key"] = image_key

    except Exception as error:
        st.error(
            f"Could not open the uploaded image: "
            f"{type(error).__name__}: {error}"
        )


# ---------------------------------------------------------
# SHOW IMAGE CORRECTLY
# ---------------------------------------------------------

if image is not None:
    st.divider()
    st.header("📷 Selected Retinal Image")

    image_col, info_col = st.columns([1, 1])

    with image_col:
        # IMPORTANT: Display the PIL image directly.
        # Do not construct an HTML <img> tag or a local file URL.
        st.image(
            image,
            caption="Uploaded Retinal / Fundus Image",
            use_container_width=True,
        )

    with info_col:
        st.subheader("Image Information")
        st.write(f"**Width:** {image.width} px")
        st.write(f"**Height:** {image.height} px")
        st.write(f"**Format:** {image.format or 'RGB image'}")
        st.write(f"**Device:** {DEVICE}")

    st.divider()

    # -----------------------------------------------------
    # TWO PREDICTION OPTIONS
    # -----------------------------------------------------

    st.header("Choose Prediction")

    option = st.radio(
        "Select which prediction to run:",
        [
            "❤️ Heart Disease Risk",
            "👁️ Retinal Prediction Results",
        ],
        horizontal=True,
        key="prediction_option",
    )

    if option == "❤️ Heart Disease Risk":
        heart_disease_section()

    else:
        st.subheader("👁️ Retinal Prediction Results")

        if st.button(
            "🔍 Run Retinal Prediction",
            type="primary",
            use_container_width=True,
            key="run_retinal",
        ):
            results = {}

            with st.spinner("Running retinal models..."):
                for name, (filename, task) in MODEL_FILES.items():
                    model_path = MODEL_DIR / filename

                    try:
                        model = load_retinal_model(
                            str(model_path),
                            task,
                            (
                                model_path.stat().st_mtime
                                if model_path.exists()
                                else 0
                            ),
                        )

                        results[name] = retinal_predict(
                            model,
                            image,
                            task,
                        )

                    except Exception as error:
                        st.error(
                            f"**{name}** — "
                            f"{type(error).__name__}: {error}"
                        )

            st.session_state["retinal_results"] = results
            st.session_state["retinal_result_key"] = image_key

        if (
            st.session_state["retinal_results"] is not None
            and st.session_state["retinal_result_key"] == image_key
        ):
            show_retinal_results(
                st.session_state["retinal_results"]
            )

            show_gradcam(image, image_key)

else:
    st.info(
        "Upload a retinal image above to display the image and select "
        "one of the two prediction options."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Retina AI Research Prediction Dashboard | Academic research use only. "
    "Predictions are not a substitute for professional medical evaluation."
)
```
