import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import easyocr
import re
import pandas as pd
import os
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Image Detection", layout="wide")
st.title("🖼 Image Upload – Number Plate Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("runs/exp/weights/best.pt")   # ✅ YOUR MODEL PATH

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

model = load_model()
ocr_reader = load_ocr()

# ---------------- DIRECTORIES ----------------
BASE_DIR = "data"
ORIGINAL_DIR = f"{BASE_DIR}/originals"
PREDICTED_DIR = f"{BASE_DIR}/predictions"
CSV_PATH = f"{BASE_DIR}/detection_log.csv"

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# ---------------- INIT CSV ----------------
if not os.path.exists(CSV_PATH):
    df_init = pd.DataFrame(columns=[
        "entry_datetime",
        "threshold",
        "ocr_text",
        "original_image_path",
        "predicted_image_path"
    ])
    df_init.to_csv(CSV_PATH, index=False)

# ---------------- OCR PREPROCESS ----------------
def preprocess_for_ocr(plate_rgb):
    gray = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return thresh

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    0.10, 0.90, 0.40, 0.05
)

st.markdown(f"**Selected Threshold:** `{confidence:.2f}`")

# ---------------- IMAGE PIPELINE ----------------
if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    st.subheader("Original Image Preview")
    st.image(image, use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running detection and OCR..."):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = f"{ORIGINAL_DIR}/img_{timestamp}.jpg"
            predicted_path = f"{PREDICTED_DIR}/pred_{timestamp}.jpg"

            image.save(original_path)

            img_rgb = np.array(image)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            results = model.predict(
                source=img_bgr,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

            ocr_values = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])

                plate_crop = img_rgb[y1:y2, x1:x2]
                ocr_text = ""

                if plate_crop.size != 0:
                    plate_proc = preprocess_for_ocr(plate_crop)
                    ocr_result = ocr_reader.readtext(
                        plate_proc,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=0
                    )
                    if ocr_result:
                        ocr_text = re.sub(r'[^A-Z0-9]', '', ocr_result[0])
                        ocr_values.append(ocr_text)

                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

                label = f"{ocr_text} ({conf_score:.2f})" if ocr_text else f"({conf_score:.2f})"
                cv2.putText(
                    img_bgr,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            final_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(final_img).save(predicted_path)

            # ---------------- SAVE LOG ----------------
            log_row = {
                "entry_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "threshold": confidence,
                "ocr_text": ", ".join(ocr_values),
                "original_image_path": original_path,
                "predicted_image_path": predicted_path
            }

            df = pd.read_csv(CSV_PATH)
            df = pd.concat([df, pd.DataFrame([log_row])], ignore_index=True)
            df.to_csv(CSV_PATH, index=False)

            st.subheader("Detection Result")
            st.image(final_img, use_container_width=True)

            st.success("✅ Detection logged successfully")


# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Number Plate Detection • YOLO + OCR</p>",
    unsafe_allow_html=True
)
