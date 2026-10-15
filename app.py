
import io
import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Retina AI Research Prediction Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Retina Age Model": ("retina_age_model.pth", "regression"),
    "Retina Sex Model": ("retina_sex_model_v2.pth", "classification"),
    "Retina Disease Model": ("retina_disease_model.pth", "classification"),
    "CIMT Estimate": ("cimt_thickness_model.pth", "regression"),
}

FUSION_MODEL_PATH = (
    MODEL_DIR / "retinal_feature_fusion_model_FINAL.keras"
)
CLINICAL_SCALER_PATH = MODEL_DIR / "clinical_scaler.pkl"
CAD_MODEL_PATH = MODEL_DIR / "cad_heart_disease_model.joblib"
CAP_MODEL_PATH = MODEL_DIR / "cap_grade_model.pth"


# =========================================================
# 2. SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "retinal_results": {},
    "last_retinal_image_key": None,
    "current_risk_image_key": None,
    "risk_result_image_key": None,
    "risk_result_level": None,
    "risk_result_score": None,
    "cad_result": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# 3. GENERAL HELPERS
# =========================================================

def file_status(path):
    return "Found" if Path(path).is_file() else "Missing"


def show_file_status(label, path):
    if Path(path).is_file():
        st.sidebar.success(f"{label}: Found")
    else:
        st.sidebar.error(f"{label}: Missing")


def safe_float(value):
    try:
        arr = np.asarray(value).astype(float).ravel()
        if len(arr) and np.isfinite(arr[0]):
            return float(arr[0])
    except (TypeError, ValueError):
        pass
    return None


def image_to_array(image, size=(224, 224)):
    image = image.convert("RGB").resize(size)
    return np.asarray(image, dtype=np.float32) / 255.0


# =========================================================
# 4. RETINAL MODEL LOADER
# =========================================================

@st.cache_resource(show_spinner=False)
def load_retinal_model(model_path, modified_time):
    """
    Supports TorchScript models and checkpoints containing
    a complete serialized torch.nn.Module.

    A state_dict alone is not enough to reconstruct a custom
    architecture; that architecture must be supplied separately.
    """
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path.name}")

    try:
        model = torch.jit.load(str(path), map_location=DEVICE)
        model.eval()
        return model
    except Exception:
        pass

    try:
        checkpoint = torch.load(
            str(path),
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(str(path), map_location=DEVICE)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(DEVICE)
        model.eval()
        return model

    if isinstance(checkpoint, dict):
        for key in ("model", "module", "network"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, nn.Module):
                candidate = candidate.to(DEVICE)
                candidate.eval()
                return candidate

    raise RuntimeError(
        f"{path.name} is a checkpoint/state_dict, not a complete model. "
        "Its original architecture is required to run predictions."
    )


def run_retinal_prediction(model, image, task):
    """
    Generic inference for models accepting a 224x224 RGB tensor.
    Verify the image size, normalization, and output mapping against
    the training notebook before interpreting results clinically.
    """
    img = image.convert("RGB").resize((224, 224))

    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

    if isinstance(output, (list, tuple)):
        output = output[0]

    if isinstance(output, dict):
        if len(output) == 1:
            output = next(iter(output.values()))
        else:
            raise RuntimeError(
                "Model returned multiple named outputs. "
                "Output mapping needs model-specific configuration."
            )

    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)

    values = output.detach().float().cpu().numpy().ravel()

    if len(values) == 0:
        raise RuntimeError("Model returned an empty prediction.")

    if task == "regression":
        return {
            "type": "regression",
            "value": float(values[0]),
        }

    if len(values) == 1:
        raw_score = float(values[0])
        probability = (
            raw_score if 0 <= raw_score <= 1
            else 1 / (1 + np.exp(-np.clip(raw_score, -50, 50)))
        )
        return {
            "type": "binary",
            "score": probability,
            "note": "Class meaning must be verified from training labels.",
        }

    probabilities = np.exp(values - np.max(values))
    probabilities = probabilities / probabilities.sum()
    predicted_class = int(np.argmax(probabilities))

    return {
        "type": "multiclass",
        "class_index": predicted_class,
        "score": float(probabilities[predicted_class]),
        "probabilities": probabilities.tolist(),
        "note": "Class names must be verified from training labels.",
    }


def display_model_result(name, result):
    if result["type"] == "regression":
        st.metric("Raw model output", f'{result["value"]:.4f}')
        st.caption(
            "Raw output only. Units and interpretation depend on "
            "the model's training configuration."
        )

    elif result["type"] == "binary":
        st.metric("Raw predicted score", f'{result["score"]:.2%}')
        st.info(result.get("note", "Verify model class mapping."))

    elif result["type"] == "multiclass":
        st.metric("Predicted class index", result["class_index"])
        st.metric("Top-class score", f'{result["score"]:.2%}')
        st.info(result.get("note", "Verify model class mapping."))


# =========================================================
# 5. FUSION MODEL LOADER AND PREDICTION
# =========================================================

@st.cache_resource(show_spinner=False)
def load_fusion_model(model_path, modified_time):
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            "retinal_feature_fusion_model_FINAL.keras was not found."
        )

    return tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )


@st.cache_resource(show_spinner=False)
def load_clinical_scaler(scaler_path, modified_time):
    if not Path(scaler_path).is_file():
        raise FileNotFoundError("Clinical scaler was not found.")
    return joblib.load(str(scaler_path))


def get_fusion_image_shape(model):
    for tensor in model.inputs:
        shape = tuple(tensor.shape)
        if len(shape) == 4:
            height = int(shape[1]) if shape[1] is not None else 224
            width = int(shape[2]) if shape[2] is not None else 224
            channels = int(shape[3]) if shape[3] is not None else 3
            return height, width, channels

    return 224, 224, 3


def predict_fusion_risk(image, gender, thickness, age):
    """
    Uses the fusion model and clinical scaler.

    Input order and preprocessing must match the original training
    notebook. This function supports image input plus clinical input
    when the saved model exposes compatible input tensors.
    """
    model = load_fusion_model(
        str(FUSION_MODEL_PATH),
        FUSION_MODEL_PATH.stat().st_mtime,
    )

    scaler = load_clinical_scaler(
        str(CLINICAL_SCALER_PATH),
        CLINICAL_SCALER_PATH.stat().st_mtime,
    )

    # Expected clinical feature order from the project configuration.
    clinical = pd.DataFrame(
        [[gender, thickness, age]],
        columns=["gender", "thickness", "age"],
    )
    clinical_scaled = scaler.transform(clinical).astype(np.float32)

    height, width, channels = get_fusion_image_shape(model)
    img = image.convert("RGB").resize((width, height))
    img_array = np.asarray(img, dtype=np.float32) / 255.0

    if channels == 1:
        img_array = np.mean(img_array, axis=-1, keepdims=True)
    elif channels != 3:
        raise ValueError(
            f"Unsupported fusion image channel count: {channels}"
        )

    image_input = np.expand_dims(img_array, axis=0)
    clinical_input = np.asarray(clinical_scaled, dtype=np.float32)

    input_tensors = list(model.inputs)

    if len(input_tensors) == 1:
        input_shape = tuple(input_tensors[0].shape)

        if len(input_shape) == 4:
            prediction = model.predict(image_input, verbose=0)
        elif len(input_shape) == 2:
            if clinical_input.shape[-1] != input_shape[-1]:
                raise ValueError(
                    "Clinical feature count does not match the fusion "
                    "model's single input. Check the training notebook."
                )
            prediction = model.predict(clinical_input, verbose=0)
        else:
            raise ValueError(
                "Unsupported single-input fusion model shape. "
                "Check the training notebook."
            )
    else:
        feed = []

        for tensor in input_tensors:
            shape = tuple(tensor.shape)
            name = tensor.name.lower()

            if len(shape) == 4:
                feed.append(image_input)
            elif len(shape) == 2:
                if (
                    shape[-1] is not None
                    and int(shape[-1]) != clinical_input.shape[-1]
                ):
                    raise ValueError(
                        f"Clinical input has {clinical_input.shape[-1]} "
                        f"features but model input {tensor.name} expects "
                        f"{shape[-1]}. Verify feature columns and order."
                    )
                feed.append(clinical_input)
            else:
                raise ValueError(
                    f"Unsupported fusion input: {tensor.name}, shape={shape}"
                )

        prediction = model.predict(feed, verbose=0)

    output = np.asarray(prediction, dtype=np.float64).ravel()

    if output.size == 0 or not np.all(np.isfinite(output)):
        raise ValueError("Fusion model returned an invalid output.")

    if output.size == 1:
        score = float(output[0])

        # Do not silently turn an arbitrary regression value into a risk
        # probability. A scalar is usable as a score only if in [0, 1].
        if not 0 <= score <= 1:
            raise ValueError(
                f"Model returned {score:.4f}, outside [0, 1]. "
                "Confirm whether the model output is a probability, logit, "
                "or regression value in the training notebook."
            )

        probability = score

    else:
        # For multi-class outputs, use the last class only when the
        # training label mapping confirms that it represents high risk.
        probabilities = np.exp(output - np.max(output))
        probabilities = probabilities / probabilities.sum()
        probability = float(probabilities[-1])

    return probability


def display_risk_prediction(image):
    st.header("❤️ Heart Disease Risk Prediction")
    st.caption(
        "Uses retinal image and clinical inputs when compatible model "
        "and scaler files are available."
    )

    if image is None:
        st.info("Upload or capture a retinal image to run risk prediction.")
        return

    if not FUSION_MODEL_PATH.is_file():
        st.error("Fusion model file is missing.")
        return

    if not CLINICAL_SCALER_PATH.is_file():
        st.error("Clinical scaler file is missing.")
        return

    left, right = st.columns(2)

    with left:
        gender_label = st.selectbox(
            "Gender value (use training dataset encoding)",
            ["0", "1"],
            key="fusion_gender",
        )
        thickness = st.number_input(
            "CIMT / thickness value",
            min_value=0.0,
            value=0.6,
            step=0.01,
            format="%.4f",
            key="fusion_thickness",
        )
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45,
            key="fusion_age",
        )

    with right:
        st.write("**Fusion inputs**")
        st.write("• Retinal image")
        st.write("• Gender")
        st.write("• Thickness")
        st.write("• Age")
        st.warning(
            "Use the same encoding, units, feature order, and "
            "preprocessing as the training notebook."
        )

    if st.button(
        "❤️ Predict Heart Disease Risk",
        type="primary",
        use_container_width=True,
        key="predict_fusion_risk",
    ):
        try:
            with st.spinner("Running fusion model..."):
                score = predict_fusion_risk(
                    image=image,
                    gender=float(gender_label),
                    thickness=float(thickness),
                    age=float(age),
                )

            st.session_state.risk_result_image_key = (
                st.session_state.current_risk_image_key
            )
            st.session_state.risk_result_score = score

            st.subheader("Heart Disease Risk Prediction")
            st.metric("Model output score", f"{score:.2%}")
            st.caption(
                "This is a model score, not a confirmed diagnosis. "
                "Its interpretation depends on verified model labels "
                "and clinical validation."
            )

        except Exception as error:
            st.error(f"Fusion prediction failed: {error}")


# =========================================================
# 6. CAD MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_cad_model(model_path, modified_time):
    if not Path(model_path).is_file():
        raise FileNotFoundError("CAD model was not found.")
    return joblib.load(str(model_path))


def display_cad_prediction():
    st.header("🫀 CAD Clinical Prediction")

    if not CAD_MODEL_PATH.is_file():
        st.info("CAD model file is not available.")
        return

    st.caption(
        "Clinical input columns must match the CAD model's training "
        "features. The fields below are generic examples."
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45,
        key="cad_age",
    )
    gender = st.selectbox(
        "Gender encoding",
        [0, 1],
        key="cad_gender",
    )
    thickness = st.number_input(
        "Thickness value",
        min_value=0.0,
        value=0.6,
        step=0.01,
        key="cad_thickness",
    )

    if st.button(
        "Run CAD Prediction",
        key="run_cad_prediction",
        use_container_width=True,
    ):
        try:
            model = load_cad_model(
                str(CAD_MODEL_PATH),
                CAD_MODEL_PATH.stat().st_mtime,
            )

            features = pd.DataFrame(
                [[age, gender, thickness]],
                columns=["age", "gender", "thickness"],
            )

            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                if not set(expected).issubset(features.columns):
                    st.error(
                        "CAD model expects different clinical features: "
                        + ", ".join(expected)
                    )
                    return
                features = features[expected]

            prediction = model.predict(features)[0]
            st.session_state.cad_result = str(prediction)

            st.metric("CAD model output", str(prediction))

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                st.write("Class probabilities:")
                st.dataframe(
                    pd.DataFrame({
                        "Class": [
                            str(x) for x in model.classes_
                        ],
                        "Probability": probabilities,
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "Verify feature names, feature order and class mapping "
                "against the CAD training notebook."
            )

        except Exception as error:
            st.error(f"CAD prediction failed: {error}")


# =========================================================
# 7. SIDEBAR
# =========================================================

st.sidebar.title("👁️ Retina AI")
st.sidebar.write("Research Prediction Dashboard")
st.sidebar.caption(f"Device: {DEVICE}")

st.sidebar.subheader("Retinal Models")

for name, (filename, task) in MODEL_FILES.items():
    show_file_status(name, MODEL_DIR / filename)

st.sidebar.subheader("Clinical Models")
show_file_status("CAD Model", CAD_MODEL_PATH)
show_file_status("Fusion Risk Model", FUSION_MODEL_PATH)
show_file_status("Clinical Scaler", CLINICAL_SCALER_PATH)
show_file_status("CAP Grade Model", CAP_MODEL_PATH)

st.sidebar.caption(
    "Camera access may require browser permission and HTTPS."
)


# =========================================================
# 8. MAIN DASHBOARD
# =========================================================

st.title("👁️ Retina AI Research Dashboard")
st.caption(
    "Retinal image analysis and clinical prediction research prototype."
)
st.warning(
    "For research use only. This application does not provide "
    "a confirmed medical diagnosis."
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


# =========================================================
# 9. LOAD SELECTED IMAGE
# =========================================================

image = None
image_key = None

try:
    selected_file = uploaded_file or camera_photo

    if selected_file is not None:
        raw = selected_file.getvalue()
        image = Image.open(io.BytesIO(raw)).convert("RGB").copy()
        image_key = str(hash(raw))

except Exception as error:
    st.error(f"Could not open retinal image: {error}")


if image_key != st.session_state.current_risk_image_key:
    st.session_state.current_risk_image_key = image_key
    st.session_state.risk_result_image_key = None
    st.session_state.risk_result_level = None
    st.session_state.risk_result_score = None


# =========================================================
# 10. RETINAL PREDICTIONS
# =========================================================

if image is not None:
    st.divider()
    image_col, info_col = st.columns(2)

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
        st.header("Retinal Prediction Results")

        for name, (filename, task) in MODEL_FILES.items():
            path = MODEL_DIR / filename

            with st.expander(name, expanded=True):
                if not path.is_file():
                    st.error(f"Model file missing: {filename}")
                    continue

                try:
                    model = load_retinal_model(
                        str(path),
                        path.stat().st_mtime,
                    )
                    result = run_retinal_prediction(model, image, task)
                    results[name] = result
                    display_model_result(name, result)

                except Exception as error:
                    st.error(f"{name} failed: {error}")

        st.session_state.retinal_results = results
        st.session_state.last_retinal_image_key = image_key

elif st.session_state.retinal_results:
    st.header("Previous Retinal Prediction Results")

    for name, result in st.session_state.retinal_results.items():
        with st.expander(name):
            display_model_result(name, result)


# =========================================================
# 11. HEART DISEASE RISK PREDICTION
# =========================================================

st.divider()
display_risk_prediction(image)


# =========================================================
# 12. EXTERNAL EYE PHOTO
# =========================================================

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
        st.success("External eye photo captured successfully.")
        st.write(f"Width: {external_image.width} px")
        st.write(f"Height: {external_image.height} px")

        st.info(
            "External-eye prediction is not enabled. A separately "
            "trained and configured model is required."
        )

    except Exception as error:
        st.error(f"Could not open external eye photo: {error}")


# =========================================================
# 13. CIMT CLASSIFICATION
# =========================================================

st.divider()
st.subheader("🩺 CIMT Classification")

cimt_value = st.session_state.retinal_results.get("CIMT Estimate")

if cimt_value is not None:
    if cimt_value.get("type") == "regression":
        st.metric(
            "Raw CIMT Model Output",
            f'{cimt_value["value"]:.4f}',
        )
        st.warning(
            "Units and validated clinical cutoff are unknown. "
            "Clinical classification is disabled."
        )
    else:
        display_model_result("CIMT Estimate", cimt_value)
else:
    st.info("Run retinal predictions to display CIMT output.")


# =========================================================
# 14. CAD PREDICTION
# =========================================================

st.divider()
display_cad_prediction()


# =========================================================
# 15. CAP GRADE
# =========================================================

st.divider()
st.subheader("🔬 CAP Grade")

if CAP_MODEL_PATH.is_file():
    st.success("CAP model file found.")
    st.warning(
        "CAP inference is not enabled because the model architecture "
        "and verified grade-label mapping must be configured."
    )
else:
    st.info(
        "A compatible trained CAP model and verified grade labels "
        "are required."
    )


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()
st.caption(
    "Retina AI Research Dashboard | Research prototype only. "
    "Not validated for clinical diagnosis."
)
