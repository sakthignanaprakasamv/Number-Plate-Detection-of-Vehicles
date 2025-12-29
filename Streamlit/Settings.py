import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Live Camera Detection", layout="wide")
st.title("🎥 Live Camera Number Plate Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("runs/exp/weights/best.pt")   # ✅ YOUR MODEL PATH

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

model = load_model()
ocr_reader = load_ocr()

# ---------------- UI CONTROLS ----------------
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

st.markdown(f"**Selected Threshold:** `{confidence:.2f}`")

col_start, col_stop = st.columns(2)
start_cam = col_start.button("▶ Start Camera")
stop_cam = col_stop.button("⏹ Stop Camera")

frame_placeholder = st.empty()

# ---------------- SESSION STATE ----------------
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

if start_cam:
    st.session_state.run_camera = True

if stop_cam:
    st.session_state.run_camera = False

# ---------------- LIVE CAMERA LOOP ----------------
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Unable to access camera")
    else:
        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])

                # ---- Crop plate for OCR ----
                plate_crop = frame[y1:y2, x1:x2]
                ocr_text = ""

                if plate_crop.size != 0:
                    ocr_result = ocr_reader.readtext(plate_crop)
                    if ocr_result:
                        ocr_text = ocr_result[0][1]

                # ---- Draw thick bounding box ----
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # ---- Label (OCR + confidence only) ----
                label = f"{ocr_text} ({conf_score:.2f})" if ocr_text else f"({conf_score:.2f})"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)

            time.sleep(0.03)

        cap.release()

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Live Number Plate Detection • YOLO + OCR</p>",
    unsafe_allow_html=True
)

