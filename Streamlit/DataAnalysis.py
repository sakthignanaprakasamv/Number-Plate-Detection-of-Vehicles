import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import easyocr

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Number Plate Detection", layout="wide")
st.title("🖼 Image-based Number Plate Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("runs/exp/weights/best.pt")   # ✅ YOUR PATH

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

model = load_model()
ocr_reader = load_ocr()

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

st.markdown(f"**Selected Threshold:** `{confidence:.2f}`")

# ---------------- PREVIEW ----------------
if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    st.subheader("Image Preview")
    st.image(image, use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting number plates and running OCR..."):

            img_rgb = np.array(image)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            results = model.predict(
                source=img_bgr,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])

                # ---- Crop plate for OCR ----
                plate_crop = img_rgb[y1:y2, x1:x2]

                ocr_text = ""
                if plate_crop.size != 0:
                    ocr_result = ocr_reader.readtext(plate_crop)
                    if ocr_result:
                        ocr_text = ocr_result[0][1]

                # ---- Draw bounding box (THICK) ----
                cv2.rectangle(
                    img_bgr,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3   # 🔥 thick box
                )

                # ---- Text ABOVE box ----
                label = f"{ocr_text} ({conf_score:.2f})" if ocr_text else f"({conf_score:.2f})"

                cv2.putText(
                    img_bgr,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            final_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.subheader("Detection Result")
            st.image(final_img, use_container_width=True)
