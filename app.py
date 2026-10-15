
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
        image = Image.open(
            io.BytesIO(raw)
        ).convert("RGB").copy()
        image_key = str(hash(raw))

    elif camera_photo is not None:
        raw = camera_photo.getvalue()
        image = Image.open(
            io.BytesIO(raw)
        ).convert("RGB").copy()
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

        preview_buffer = io.BytesIO()
        image.convert("RGB").save(
            preview_buffer,
            format="PNG",
        )
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
                        task,
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
# HEART DISEASE RISK PREDICTION
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
        external_image.convert("RGB").save(
            external_buffer,
            format="PNG",
        )
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
        f"{float(cimt_value):.4f}",
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
