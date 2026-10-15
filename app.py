import os
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Retina AI Research Dashboard", page_icon="👁️", layout="wide")

st.markdown("""
<style>
.main {background-color:#f7f9fc;}
h1 {color:#17365d;}
div.stButton > button {background:#ef4444;color:white;border-radius:10px;height:3em;width:100%;font-weight:600;}
[data-testid="stMetric"] {background:white;border:1px solid #d8dee9;padding:18px;border-radius:12px;}
[data-testid="stFileUploader"] {background:white;border:1px dashed #b8c4d6;padding:12px;border-radius:12px;}
.block-container {padding-top:1.5rem;}
</style>
""", unsafe_allow_html=True)

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
DISEASE_LABELS = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
        if isinstance(checkpoint, dict):
            return checkpoint
    raise ValueError("Unsupported checkpoint format")

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
        if value.ndim == 2 and key.endswith(("classifier.1.weight", "classifier.weight", "fc.weight")):
            candidates.append((len(key), int(value.shape[0])))
    return sorted(candidates)[0][1] if candidates else default

@st.cache_resource(show_spinner=False)
def load_model(path, task):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = clean_state_dict(extract_state_dict(checkpoint))
        default_outputs = 8 if task == "multilabel" else 1
        output_size = infer_output_size(state, default_outputs)
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_size)
        result = model.load_state_dict(state, strict=False)
        if len(state) - len(result.unexpected_keys) <= 0:
            raise RuntimeError("Checkpoint keys do not match EfficientNet-B0; original training architecture is needed.")
    model.to(DEVICE).eval()
    return model

def run_prediction(model, image, task):
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("logits", output.get("output", next(iter(output.values()))))
    values = output.detach().float().cpu().reshape(-1).numpy()
    if task == "regression":
        return float(values[0])
    if task == "multilabel":
        return values if np.all((values >= 0) & (values <= 1)) else 1 / (1 + np.exp(-np.clip(values, -60, 60)))
    if len(values) == 1:
        p = float(1 / (1 + np.exp(-np.clip(values[0], -60, 60))))
        return {"Class 1": p, "Class 0": 1-p}
    values = values - values.max()
    probs = np.exp(values) / np.exp(values).sum()
    return {f"Class {i}": float(p) for i, p in enumerate(probs)}

def display_result(title, result, task):
    st.subheader(title)
    if task == "regression":
        st.metric("Raw model output", f"{result:.4f}")
        st.caption("Confirm units and clinical interpretation from training documentation.")
    elif task == "multilabel":
        labels = DISEASE_LABELS if len(result) == len(DISEASE_LABELS) else [f"Output {i+1}" for i in range(len(result))]
        cols = st.columns(2)
        for i, (label, score) in enumerate(zip(labels, result)):
            with cols[i % 2]:
                st.metric(label, f"{float(score)*100:.1f}%")
                st.progress(float(np.clip(score, 0, 1)))
        st.caption("Scores are not confirmed diagnoses or necessarily calibrated probabilities.")
    else:
        cols = st.columns(max(1, len(result)))
        for i, (label, score) in enumerate(result.items()):
            cols[i].metric(label, f"{score*100:.1f}%")
        st.caption("Class names are generic until the original label mapping is confirmed.")

st.title("👁️ Retina AI Research Dashboard")
st.write("Upload a retinal/fundus image to view outputs from the available research models.")

with st.sidebar:
    st.header("Dashboard status")
    st.write(f"**Device:** `{DEVICE}`")
    st.write(f"**Models folder:** `{MODEL_DIR}`")
    st.subheader("Checkpoint files")
    for name, (filename, _) in MODEL_FILES.items():
        exists = (MODEL_DIR / filename).is_file()
        st.write(("✅ " if exists else "⚠️ ") + name)
        st.caption(filename)
    st.caption("Research/education only; not a medical diagnosis.")

left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("1. Image source")
    uploaded = st.file_uploader("Upload retinal image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded retinal image", use_container_width=True)
            st.caption(f"Image size: {image.width} × {image.height}")
        except Exception as exc:
            st.error(f"Could not open image: {exc}")

with right:
    st.subheader("2. Run analysis")
    st.write("Each model is run independently, so one failed checkpoint will not hide other outputs.")
    run_all = st.button("Run Retina AI Analysis", disabled=image is None, use_container_width=True)
    st.info("Missing or incompatible model files will show an error in that model's result card.")

if run_all and image is not None:
    st.divider()
    st.header("All model outputs")
    st.caption("Raw research outputs only. Do not use alone for diagnosis or treatment.")
    successful = 0
    columns = st.columns(2, gap="large")
    for index, (name, (filename, task)) in enumerate(MODEL_FILES.items()):
        with columns[index % 2]:
            with st.container(border=True):
                try:
                    model = load_model(str(MODEL_DIR / filename), task)
                    result = run_prediction(model, image, task)
                    display_result(name, result, task)
                    successful += 1
                except Exception as exc:
                    st.error(f"{name} could not run.")
                    st.code(str(exc))
                    st.caption("Verify the checkpoint file and the model architecture used during training.")
    if successful:
        st.success(f"Outputs produced by {successful} model(s).")
    else:
        st.warning("No model produced an output. Review the errors shown above.")

st.divider()
st.subheader("Output interpretation")
st.markdown("""
- **Retinal Age / CIMT:** shown as raw numeric outputs; units and valid ranges must be verified.
- **Retinal Disease:** shows per-output scores; eight labels are used only if the model returns eight values.
- **Retinal Sex:** class labels remain generic until the training label mapping is verified.
- Model quality depends on image quality, training data, calibration, and external validation. This dashboard is not a substitute for a qualified clinician.
""")
