import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Live Camera Detection", layout="wide")
st.title("📷 Live Camera Detection")
st.markdown("Real-time number plate detection using live camera feed with OCR output.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("runs/exp/weights/best.pt")  # ✅ YOUR MODEL PATH

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

model = load_model()
ocr_reader = load_ocr()

# ---------------- USER CONTROLS ----------------
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

st.markdown(f"**Selected Threshold:** `{confidence:.2f}`")

# ---------------- SESSION STATE (CRITICAL FIX) ----------------
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False

# ---------------- VIDEO PROCESSOR ----------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=img,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])

            plate_crop = img[y1:y2, x1:x2]
            ocr_text = ""

            if plate_crop.size != 0:
                ocr_result = ocr_reader.readtext(
                    plate_crop,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=0
                )
                if ocr_result:
                    ocr_text = re.sub(r'[^A-Z0-9]', '', ocr_result[0])

            # ---- Draw thick bounding box ----
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # ---- Label: OCR + confidence only ----
            label = f"{ocr_text} ({conf_score:.2f})" if ocr_text else f"({conf_score:.2f})"

            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- UI CONTROLS ----------------
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.camera_started:
        if st.button("▶ Start Camera", use_container_width=True):
            st.session_state.camera_started = True
            st.rerun()

with col2:
    if st.session_state.camera_started:
        if st.button("⏹ Stop Camera", use_container_width=True):
            st.session_state.camera_started = False
            st.rerun()

# ---------------- WEBRTC STREAM (SAFE) ----------------
if st.session_state.camera_started:
    webrtc_streamer(
        key="live-camera-detection",  # MUST stay constant
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:openrelay.metered.ca:80"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        },
    )

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Live Camera Detection • YOLO + OCR</p>",
    unsafe_allow_html=True
)
