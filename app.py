# ============================================================
# RETINA AI — RESEARCH PREDICTION DASHBOARD
# ============================================================
# Keeps retinal outputs separate from the optional fusion model.
# Place trained model files in ./models beside this file.
# IMPORTANT: checkpoint architecture/preprocessing must match training.
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
import io
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import tensorflow as tf

# --------------------------- PAGE CONFIG ---------------------------

st.set_page_config(
    page_title="Retina AI — Research Prediction Dashboard",
    page_icon="👁️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f8fc; }
    h1, h2, h3 { color: #17365d; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce5f0;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

MODEL_FILES = {
    "Retinal Age": ("retina_age_model.pth", "regression"),
    "Retinal Sex": ("retina_sex_model_v2.pth", "classification"),
    "Retinal Disease": ("retina_disease_model.pth", "multilabel"),
    "CIMT Estimate": ("cimt_any_eye_resnet18_v1.pth", "cimt_classification"),
}
DISEASE_LABELS = [
    "Normal", "Diabetes", "Glaucoma", "Cataract",
    "AMD", "Hypertension", "Myopia", "Other",
]
SEX_CLASS_LABELS = ["Female", "Male"]
DISEASE_THRESHOLD = 0.50
DIABETES_THRESHOLD = 0.50
# Leave None until the target model's units and research cutoff are verified.
CIMT_POSITIVE_THRESHOLD = None

CAD_MODEL_PATH = MODEL_DIR / "cad_heart_disease_model.joblib"

RISK_LOW_MAX = 0.33
RISK_MEDIUM_MAX = 0.66

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

# --------------------------- SESSION STATE ---------------------------

DEFAULTS = {
    "selected_image_key": None,
    "retinal_results": None,
    "retinal_result_key": None,
    "cad_result": None,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------- HELPERS ---------------------------

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def finite_array(values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Model returned invalid numeric values.")
    return values


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -80, 80)
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    exps = np.exp(np.clip(values, -80, 80))
    return exps / exps.sum()


def get_output(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        elif "output" in output:
            output = output["output"]
        else:
            raise ValueError("Unknown model output dictionary.")
    return output


def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith(("module.", "model.", "net.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "weights", "model", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format.")


# --------------------------- RETINAL MODEL LOADER ---------------------------

@st.cache_resource(show_spinner=False)
def load_retinal_model(path_string, task, modified_time):
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model: {path.name}")

    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = clean_state_dict(extract_state_dict(checkpoint))

        # CIMT model: trained as torchvision ResNet-18 with a Sequential
        # classifier whose Linear layer is fc.1 (checkpoint confirms this).
        if task == "cimt_classification":
            architecture = checkpoint.get("architecture") if isinstance(checkpoint, dict) else None
            if architecture != "ResNet18_single_fundus_image_any_eye":
                raise ValueError(
                    "Unexpected CIMT architecture metadata: "
                    f"{architecture!r}. Expected ResNet18_single_fundus_image_any_eye."
                )
            weight = state.get("fc.1.weight")
            bias = state.get("fc.1.bias")
            if weight is None or bias is None or tuple(weight.shape) != (2, 512):
                raise ValueError(
                    "CIMT checkpoint must contain fc.1.weight (2, 512) and fc.1.bias."
                )
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc.in_features, int(weight.shape[0])),
            )
            model.load_state_dict(state, strict=True)
            class_map = checkpoint.get("class_names", {"0": "normal_CIMT", "1": "thickened_CIMT"})
            model._cimt_class_names = [
                str(class_map.get("0", "normal_CIMT")),
                str(class_map.get("1", "thickened_CIMT")),
            ]
            model._cimt_threshold = float(checkpoint.get("threshold", 0.5))
        else:
            # Existing retinal checkpoints use the app's EfficientNet-B0 loader.
            if task == "multilabel":
                output_size = len(DISEASE_LABELS)
            elif task == "classification":
                weight = state.get("classifier.1.weight")
                if weight is None:
                    raise ValueError(
                        "Cannot infer classifier size from this checkpoint. "
                        "Use the original model architecture from training."
                    )
                output_size = int(weight.shape[0])
            else:
                output_size = 1

            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, output_size
            )
            model.load_state_dict(state, strict=True)

    model = model.to(DEVICE).eval()
    return model


def retinal_predict(model, image, task):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output = get_output(model(tensor))

    values = output.detach().float().cpu().numpy().reshape(-1)
    values = finite_array(values)

    if task == "regression":
        return float(values[0])

    if task == "cimt_classification":
        probabilities = softmax(values)
        labels = getattr(model, "_cimt_class_names", ["normal_CIMT", "thickened_CIMT"])
        return {
            "probabilities": {labels[i]: float(probabilities[i]) for i in range(min(2, len(probabilities)))},
            "predicted_class": labels[int(np.argmax(probabilities))],
            "confidence": float(np.max(probabilities)),
            "threshold": float(getattr(model, "_cimt_threshold", 0.5)),
        }

    if task == "multilabel":
        # Verify whether the training code used sigmoid logits or probabilities.
        if np.all((values >= 0) & (values <= 1)):
            return values
        return sigmoid(values)

    if len(values) == 1:
        p = float(sigmoid(values)[0])
        return {"Class 0": 1.0 - p, "Class 1": p}

    probabilities = softmax(values)
    return {f"Class {i}": float(p) for i, p in enumerate(probabilities)}


# --------------------------- RETINAL RESULTS ---------------------------

def show_disease_results(scores):
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(DISEASE_LABELS):
        st.error(
            f"Expected {len(DISEASE_LABELS)} disease outputs; received "
            f"{len(scores)}. Check the trained model and label mapping."
        )
        return

    scores = np.clip(scores, 0, 1)
    diabetes = float(scores[1])
    max_non_normal = float(np.max(scores[1:]))

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Diabetes Retinal Model",
            "Positive" if diabetes >= DIABETES_THRESHOLD else "Negative",
            f"Score: {diabetes:.1%}",
        )
    with col2:
        st.metric(
            "Overall Retinal Disease",
            "Positive" if max_non_normal >= DISEASE_THRESHOLD else "Negative",
            f"Highest score: {max_non_normal:.1%}",
        )

    with st.expander("View all disease scores", expanded=True):
        for label, score in zip(DISEASE_LABELS, scores):
            st.write(f"**{label}:** {float(score):.1%}")
            st.progress(float(score))

    st.caption(
        "Research predictions only. Confirm label order, preprocessing, "
        "and thresholds against the training configuration."
    )


def show_cimt_result(result):
    st.markdown("### 🩺 CIMT Classification")
    if not isinstance(result, dict) or "probabilities" not in result:
        st.error("CIMT model returned an unexpected result format.")
        return

    predicted = result.get("predicted_class", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    probabilities = result.get("probabilities", {})
    col1, col2 = st.columns(2)
    with col1:
        display_name = {
            "normal_CIMT": "Normal CIMT",
            "thickened_CIMT": "Thickened CIMT",
        }.get(predicted, predicted)
        st.metric("Predicted CIMT Class", display_name)
    with col2:
        st.metric("Prediction confidence", f"{confidence:.1%}")

    for label, probability in probabilities.items():
        display_label = {
            "normal_CIMT": "Normal CIMT",
            "thickened_CIMT": "Thickened CIMT",
        }.get(label, label)
        st.write(f"**{display_label}:** {float(probability):.1%}")
        st.progress(float(np.clip(probability, 0, 1)))

    st.caption(
        "Research classification from one fundus image. This is not a direct "
        "measurement of CIMT thickness and is not a clinical diagnosis."
    )


def show_retinal_results(results):
    st.header("🔍 Retinal Prediction Results")
    if not results:
        st.warning("No retinal model produced a prediction.")
        return

    cols = st.columns(2)
    for index, (name, result) in enumerate(results.items()):
        with cols[index % 2]:
            with st.container(border=True):
                st.subheader(name)
                if name == "Retinal Age":
                    st.metric("Estimated Age", f"{float(result):.1f} years")
                elif name == "Retinal Sex":
                    if not isinstance(result, dict) or len(result) != 2:
                        st.error("Sex model must return two class scores.")
                    else:
                        probabilities = list(result.values())
                        idx = int(np.argmax(probabilities))
                        st.metric("Predicted Sex", SEX_CLASS_LABELS[idx])
                        st.write(f"Model score: {probabilities[idx]:.1%}")
                        st.caption("Verify class order using training labels.")
                elif name == "Retinal Disease":
                    show_disease_results(result)
                elif name == "CIMT Estimate":
                    show_cimt_result(result)


# --------------------------- IMAGE-ONLY CIMT MODEL ---------------------------

IMAGE_ONLY_CIMT_MODEL_PATH = MODEL_DIR / "image_only_cimt_final.keras"
IMAGE_ONLY_CIMT_EVALUATION_PATH = MODEL_DIR / "evaluation_results.json"

@st.cache_resource(show_spinner=False)
def load_image_only_cimt_model(path_string, modified_time):
    return tf.keras.models.load_model(path_string, compile=False)


def image_only_cimt_prediction_section(image):
    st.divider()
    st.header("🩺 Retinal Image-only CIMT Classification")
    st.caption(
        "Uses the newly trained retinal image model. This predicts CIMT class, "
        "not heart disease directly and not a measured CIMT thickness."
    )

    if not IMAGE_ONLY_CIMT_MODEL_PATH.is_file():
        st.warning(
            f"Model file not found: `{IMAGE_ONLY_CIMT_MODEL_PATH}`. "
            "Upload best_image_only_cimt.keras into the app's models folder."
        )
        return

    saved_results = {}
    if IMAGE_ONLY_CIMT_RESULTS_PATH.is_file():
        try:
            with open(IMAGE_ONLY_CIMT_RESULTS_PATH, "r", encoding="utf-8") as f:
                saved_results = json.load(f)
        except Exception as error:
            st.warning(f"Could not read evaluation_results.json: {error}")

    if saved_results:
        st.markdown("#### 📊 Saved test-set evaluation")
        metric_cols = st.columns(3)
        accuracy = saved_results.get("test_accuracy")
        auc = saved_results.get("test_roc_auc")
        test_images = saved_results.get("test_images")
        metric_cols[0].metric(
            "Test Accuracy",
            f"{float(accuracy):.2%}" if accuracy is not None else "Not available",
        )
        metric_cols[1].metric(
            "Test ROC-AUC",
            f"{float(auc):.4f}" if auc is not None else "Not available",
        )
        metric_cols[2].metric(
            "Test Images",
            str(test_images) if test_images is not None else "Not available",
        )

    if st.button(
        "🔍 Run Image-only CIMT Prediction",
        type="primary",
        use_container_width=True,
        key="run_image_only_cimt",
    ):
        try:
            model = load_image_only_cimt_model(
                str(IMAGE_ONLY_CIMT_MODEL_PATH),
                IMAGE_ONLY_CIMT_MODEL_PATH.stat().st_mtime,
            )
            image_size = 224
            model_shape = getattr(model, "input_shape", None)
            if isinstance(model_shape, list):
                model_shape = model_shape[0]
            if model_shape and len(model_shape) >= 3:
                image_size = int(model_shape[1] or 224)

            image_array = np.asarray(
                image.convert("RGB").resize((image_size, image_size)),
                dtype=np.float32,
            ) / 255.0

            output = np.asarray(
                model.predict(image_array[np.newaxis, ...], verbose=0)
            ).reshape(-1)
            output = finite_array(output)
            if len(output) != 1:
                raise ValueError(
                    f"Expected one sigmoid output for binary CIMT classification; got {len(output)}."
                )

            p_class1 = float(np.clip(output[0], 0.0, 1.0))
            p_class0 = 1.0 - p_class1
            predicted_class = 1 if p_class1 >= 0.5 else 0
            class_name = "Thickened CIMT (Class 1)" if predicted_class == 1 else "Normal CIMT (Class 0)"
            confidence = max(p_class0, p_class1)

            st.markdown("#### 🧾 Prediction result")
            result_cols = st.columns(2)
            result_cols[0].metric("Predicted CIMT Class", class_name)
            result_cols[1].metric("Prediction confidence", f"{confidence:.1%}")
            st.write(f"**Class 0 probability:** {p_class0:.1%}")
            st.progress(p_class0)
            st.write(f"**Class 1 probability:** {p_class1:.1%}")
            st.progress(p_class1)
            st.caption(
                "Research output only. It is not a direct CIMT measurement, "
                "a heart-disease prediction, or a clinical diagnosis."
            )
        except Exception as error:
            st.error(
                f"Image-only CIMT prediction failed: {type(error).__name__}: {error}"
            )


# --------------------------- CLINICAL-ONLY MODEL ---------------------------

@st.cache_resource(show_spinner=False)
def load_cad_artifact(path_string, modified_time):
    artifact = joblib.load(path_string)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {
        "model": artifact,
        "feature_columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ],
        "test_accuracy": None,
        "test_roc_auc": None,
        "dataset": "Unknown",
    }



def clinical_only_prediction_section(image=None, image_key=None):
    st.divider()
    st.header("❤️ Heart Disease Prediction — Clinical Model")
    st.caption(
        "Clinical-only model. Retinal image predictions are shown separately; "
        "clinical values must come from verified patient information."
    )

    if not CAD_MODEL_PATH.is_file():
        st.info("Clinical-only model not found in the models folder.")
        return

    try:
        artifact = load_cad_artifact(
            str(CAD_MODEL_PATH), CAD_MODEL_PATH.stat().st_mtime
        )
        model = artifact["model"]
        features = artifact.get("feature_columns", [])
    except Exception as error:
        st.error(f"Could not load clinical model: {error}")
        return

    # Show retinal model estimates when already available.
    retinal_results = st.session_state.get("retinal_results")
    if (
        image_key is not None
        and st.session_state.get("retinal_result_key") == image_key
        and retinal_results
    ):
        st.subheader("👁️ Image-based estimates")
        if "Retinal Age" in retinal_results:
            st.metric(
                "Estimated Age",
                f"{float(retinal_results['Retinal Age']):.1f} years",
            )

        if "Retinal Sex" in retinal_results:
            sex_result = retinal_results["Retinal Sex"]
            if isinstance(sex_result, dict) and len(sex_result) == 2:
                labels = list(sex_result.keys())
                probabilities = list(sex_result.values())
                st.metric(
                    "Estimated Sex",
                    SEX_CLASS_LABELS[int(np.argmax(probabilities))],
                )
                st.caption(
                    "Check the retinal sex model's training class order."
                )
    else:
        st.info(
            "Run Retinal Prediction first to display available image-based "
            "age and sex estimates."
        )

    st.subheader("Patient Clinical Information")
    st.info(
        "Enter verified clinical information below. The retinal image alone "
        "cannot reliably determine these clinical measurements."
    )

    with st.form("cad_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age (years)", 1, 120, 51)
            sex = st.selectbox(
                "Sex",
                [0, 1],
                index=0,
                format_func=lambda x: "Female" if x == 0 else "Male",
            )
            cp = st.selectbox("Chest pain type (cp)", [1, 2, 3, 4], index=0)
            trestbps = st.number_input(
                "Resting blood pressure", 50, 250, 120
            )
            chol = st.number_input("Cholesterol", 50, 700, 200)

        with c2:
            fbs = st.selectbox(
                "Fasting blood sugar > 120",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
            )
            restecg = st.selectbox("Resting ECG", [0, 1, 2], index=0)
            thalach = st.number_input("Maximum heart rate", 40, 250, 150)
            exang = st.selectbox(
                "Exercise-induced angina",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
            )

        with c3:
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            slope = st.selectbox("Slope", [1, 2, 3], index=0)
            ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3], index=0)
            thal = st.selectbox("Thal", [3, 6, 7], index=0)

        submitted = st.form_submit_button(
            "Run Clinical Heart Disease Prediction",
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
            frame = pd.DataFrame([patient]).reindex(columns=features)
            prediction = int(model.predict(frame)[0])

            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(frame)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                positive = [
                    i for i, cls in enumerate(classes)
                    if str(cls) in ("1", "1.0", "True")
                ]
                if positive:
                    probability = float(probabilities[positive[0]])
                elif len(probabilities) > 1:
                    probability = float(probabilities[-1])

            st.session_state.cad_result = {
                "prediction": prediction,
                "probability": probability,
                "image_key": image_key,
            }

        except Exception as error:
            st.error(
                f"Clinical prediction failed: "
                f"{type(error).__name__}: {error}"
            )

    result = st.session_state.cad_result
    if (
        result is not None
        and result.get("image_key") == image_key
    ):
        if result["prediction"] == 1:
            st.warning("Model output: Disease present")
        else:
            st.success("Model output: No disease detected by the model")

        if result["probability"] is not None:
            st.metric(
                "Positive-class probability",
                f"{result['probability']:.2%}",
            )

        accuracy = artifact.get("test_accuracy")
        auc = artifact.get("test_roc_auc")
        a, b = st.columns(2)

        with a:
            st.metric(
                "Saved test accuracy",
                f"{float(accuracy) * 100:.2f}%"
                if accuracy is not None else "Not available",
            )

        with b:
            st.metric(
                "Saved test ROC-AUC",
                f"{float(auc):.4f}"
                if auc is not None else "Not available",
            )

        st.caption(f"Dataset: {artifact.get('dataset', 'Not specified')}")
        st.warning(
            "Research prototype only; not a diagnosis or validated "
            "personal risk estimate."
        )



# --------------------------- GRAD-CAM ---------------------------

def generate_gradcam(model, image, target_index=None):
    if not hasattr(model, "features"):
        raise ValueError("Grad-CAM here is configured for EfficientNet-style models.")

    activations, gradients = [], []
    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        model.zero_grad(set_to_none=True)
        output = get_output(model(tensor))
        if output.ndim != 2 or output.shape[1] < 2:
            raise ValueError("Grad-CAM requires a multi-class classification output.")
        if target_index is None:
            target_index = int(output[0].argmax().item())
        if not 0 <= target_index < output.shape[1]:
            raise ValueError("Invalid Grad-CAM target class.")
        output[0, target_index].backward()

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False,
        )[0, 0].detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, target_index
    finally:
        handle1.remove()
        handle2.remove()


def show_gradcam(image):
    st.divider()
    st.header("🔥 Grad-CAM — Model Attention")
    st.caption(
        "Heatmap is an explanation visualization, not disease segmentation "
        "or proof of clinical reasoning."
    )

    available = [
        name for name, (filename, task) in MODEL_FILES.items()
        if task in ("classification", "multilabel")
        and (MODEL_DIR / filename).is_file()
    ]
    if not available:
        st.info("No compatible saved classification model found for Grad-CAM.")
        return

    chosen = st.selectbox("Choose model for Grad-CAM", available, key="gradcam_model")
    if st.button("🔥 Generate Grad-CAM", type="primary", key="generate_gradcam"):
        filename, task = MODEL_FILES[chosen]
        path = MODEL_DIR / filename
        try:
            model = load_retinal_model(str(path), task, path.stat().st_mtime)
            cam, class_index = generate_gradcam(model, image)
            base = np.asarray(
                image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)),
                dtype=np.float32,
            ) / 255.0
            heat = np.zeros_like(base)
            heat[..., 0] = cam
            heat[..., 1] = np.maximum(0, 1 - np.abs(cam - 0.5) * 2)
            heat[..., 2] = 1 - cam
            overlay = np.clip(0.55 * base + 0.45 * heat, 0, 1)
            overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))
       
            c1, c2 = st.columns(2)

            with c1:
                st.image(
                    image.convert("RGB"),
                    caption="Original retinal image",
                    width=450,
                )

            with c2:
                st.image(
                    overlay_image.convert("RGB"),
                    caption=f"Grad-CAM — class index {class_index}",
                    width=450,
                )
        except Exception as error:
            st.error(f"Grad-CAM failed: {type(error).__name__}: {error}")


# --------------------------- SIDEBAR STATUS ---------------------------

st.sidebar.title("👁️ Retina AI")
st.sidebar.write("Research Prediction Dashboard")
st.sidebar.caption(f"Device: {DEVICE}")
st.sidebar.subheader("Retinal Models")

for name, (filename, task) in MODEL_FILES.items():
    path = MODEL_DIR / filename
    if path.is_file():
        st.sidebar.success(f"{name}: Found")
    else:
        st.sidebar.error(f"{name}: Missing")

st.sidebar.subheader("Prediction Models")
for name, path in [
    ("Image-only CIMT Model", IMAGE_ONLY_CIMT_MODEL_PATH),
    ("CIMT Evaluation Results", IMAGE_ONLY_CIMT_RESULTS_PATH),,
    ("Clinical CAD Model", CAD_MODEL_PATH),
]:
    if path.is_file():
        st.sidebar.success(f"{name}: Found")
    else:
        st.sidebar.info(f"{name}: Missing")


# --------------------------- MAIN DASHBOARD ---------------------------

import io
import hashlib
from PIL import Image, ImageOps

st.title("👁️ Retina AI Research Dashboard")
st.caption(
    "Upload a retinal image once, then run retinal analysis and "
    "heart-disease prediction separately."
)
st.warning(
    "Research prototype only. Not validated for clinical diagnosis."
)

# --------------------------- IMAGE UPLOAD ---------------------------

upload_col, camera_col = st.columns(2)

with upload_col:
    st.subheader("🖼️ Upload Retinal Image")
    uploaded_file = st.file_uploader(
        "Choose retinal image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        key="retina_upload",
    )

with camera_col:
    st.subheader("📷 Capture Retinal Image")
    camera_photo = st.camera_input(
        "Capture fundus image",
        key="retina_camera",
    )

# --------------------------- LOAD IMAGE ---------------------------

image = None
image_key = None

selected = uploaded_file if uploaded_file is not None else camera_photo

if selected is not None:
    try:
        raw_image_bytes = selected.getvalue()
        image_key = hashlib.sha256(raw_image_bytes).hexdigest()

        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(raw_image_bytes))
        ).convert("RGB")

    except Exception as error:
        st.error(
            f"Could not open image: {type(error).__name__}: {error}"
        )

# --------------------------- SESSION STATE ---------------------------

default_states = {
    "selected_image_key": None,
    "retinal_results": None,
    "retinal_result_key": None,
    "cad_result": None,
}

for state_name, default_value in default_states.items():
    if state_name not in st.session_state:
        st.session_state[state_name] = default_value

if image_key != st.session_state.selected_image_key:
    st.session_state.selected_image_key = image_key
    st.session_state.retinal_results = None
    st.session_state.retinal_result_key = None
    st.session_state.cad_result = None

# --------------------------- IMAGE PREVIEW ---------------------------

if image is not None:
    st.divider()
    st.subheader("📷 Selected Retinal Image")

    preview_col, details_col = st.columns([1, 1])

    with preview_col:
        st.markdown("### Uploaded Retinal Image")
        st.image(
            image,
            caption="Uploaded Retinal / Fundus Image",
            use_container_width=True,
        )

    with details_col:
        st.markdown("### Image Information")
        st.write(f"**Width:** {image.width} px")
        st.write(f"**Height:** {image.height} px")
        st.write("**Format:** RGB")
        st.write(f"**Image mode:** {image.mode}")


# --------------------------- REST OF YOUR DASHBOARD ---------------------------
# Keep retinal analysis, clinical prediction, Grad-CAM and other dashboard code.

# --------------------------------------------------
# DASHBOARD CONTINUES BELOW
# --------------------------------------------------


    st.divider()
    st.header("Choose Prediction")
    retinal_col, heart_col = st.columns(2)

    with retinal_col:
        st.markdown("### 👁️ Retinal Analysis")
        run_retinal = st.button(
            "🔍 Run Retinal Prediction",
            type="primary",
            use_container_width=True,
            key="run_retinal",
        )

    with heart_col:
        st.markdown("### ❤️ Heart Disease Risk")
        st.caption("Clinical heart-disease model remains a separate section below.")

    if run_retinal:
        results = {}
        with st.spinner("Running retinal models..."):
            for name, (filename, task) in MODEL_FILES.items():
                path = MODEL_DIR / filename
                try:
                    model = load_retinal_model(
                        str(path), task,
                        path.stat().st_mtime if path.exists() else 0,
                    )
                    results[name] = retinal_predict(model, image, task)
                except Exception as error:
                    st.error(f"{name}: {type(error).__name__}: {error}")

        st.session_state.retinal_results = results
        st.session_state.retinal_result_key = image_key

    if (
        st.session_state.retinal_results is not None
        and st.session_state.retinal_result_key == image_key
    ):
        show_retinal_results(st.session_state.retinal_results)

    # Newly trained Keras image-only CIMT model is shown separately.
    image_only_cimt_prediction_section(image)

    show_gradcam(image)

else:
    st.info(
        "Upload or capture a retinal image to run retinal analysis and the new image-only CIMT model. "
        "The clinical-only Heart Disease Model below works without an image."
    )

# Clinical-only model is available whether or not a retinal image was uploaded.
clinical_only_prediction_section()

st.divider()
st.caption(
    "Retina AI Research Dashboard | Experimental research only | "
    "Not validated for clinical diagnosis."
)
st.sidebar.caption("Camera access may require browser permission and HTTPS.")
