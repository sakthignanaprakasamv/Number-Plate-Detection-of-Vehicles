import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import easyocr
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Video Detection", layout="wide")
st.title("🎞 Video-based Number Plate Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("runs/exp/weights/best.pt")  # ✅ YOUR PATH

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

model = load_model()
ocr_reader = load_ocr()

# ---------------- UI ----------------
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

st.markdown(f"**Selected Threshold:** `{confidence:.2f}`")

uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

# ---------------- VIDEO PROCESSING ----------------
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    video_placeholder = st.empty()

    if st.button("▶ Run Detection on Video"):
        with st.spinner("Processing video..."):

            while cap.isOpened():
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
                        ocr = ocr_reader.readtext(plate_crop)
                        if ocr:
                            ocr_text = ocr[0][1]

                    # ---- Draw thick bounding box ----
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    label = (
                        f"{ocr_text} ({conf_score:.2f})"
                        if ocr_text else f"({conf_score:.2f})"
                    )

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
                video_placeholder.image(frame_rgb, use_container_width=True)

            cap.release()
            os.remove(tfile.name)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Video Detection • YOLO + OCR</p>",
    unsafe_allow_html=True
)
