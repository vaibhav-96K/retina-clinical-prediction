
# =====================================================
# CHOOSE IMAGE SOURCE — FIRST SECTION
# =====================================================

st.header("👁️ External Eye Photo")

st.write(
    "मोबाइलच्या camera ने डोळ्याचा बाहेरून फोटो काढा. "
    "हा retinal किंवा fundus photo पेक्षा वेगळा प्रकार आहे."
)

image_source = st.radio(
    "Choose image source",
    ["📷 Take Photo", "🖼️ Upload Image"],
    horizontal=True,
    key="image_source",
)

if image_source == "📷 Take Photo":
    selected_image = st.camera_input(
        "📷 Take Photo",
        key="external_eye_camera",
    )
else:
    selected_image = st.file_uploader(
        "🖼️ Upload Image",
        type=["jpg", "jpeg", "png"],
        key="external_eye_upload",
    )

if selected_image is not None:
    from PIL import Image

    eye_image = Image.open(selected_image).convert("RGB")
    st.image(
        eye_image,
        caption="Selected External Eye Photo",
        use_container_width=True,
    )
