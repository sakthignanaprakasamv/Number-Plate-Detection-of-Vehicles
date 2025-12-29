import streamlit as st
import pandas as pd
import os
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Detection Reports", layout="wide")
st.title("📊 Detection Reports & Logs")

st.markdown(
    "This page displays all detection logs, OCR results, confidence thresholds, "
    "and allows downloading the report as CSV."
)

# ---------------- PATHS ----------------
DATA_DIR = "data"
CSV_PATH = f"{DATA_DIR}/detection_log.csv"

ORIGINAL_DIR = f"{DATA_DIR}/originals"
PREDICTED_DIR = f"{DATA_DIR}/predictions"

# ---------------- CHECK CSV ----------------
if not os.path.exists(CSV_PATH):
    st.warning("No detection logs found yet. Run Image Detection first.")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)

if df.empty:
    st.info("Detection log is empty.")
    st.stop()

# ---------------- FILTER SECTION ----------------
st.subheader("🔍 Filter Logs")

col1, col2 = st.columns(2)

with col1:
    min_threshold = st.slider(
        "Minimum Confidence Threshold",
        0.0, 1.0, 0.0, 0.05
    )

with col2:
    search_ocr = st.text_input(
        "Search OCR Text (optional)",
        placeholder="e.g. KL41"
    )

filtered_df = df[df["threshold"] >= min_threshold]

if search_ocr:
    filtered_df = filtered_df[
        filtered_df["ocr_text"].str.contains(search_ocr, case=False, na=False)
    ]

# ---------------- TABLE VIEW ----------------
st.subheader("📋 Detection Log Table")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# ---------------- IMAGE PREVIEW SECTION ----------------
st.subheader("🖼 Detection Image Preview")

for _, row in filtered_df.iterrows():
    with st.expander(
        f"{row['entry_datetime']} | OCR: {row['ocr_text']} | Threshold: {row['threshold']}"
    ):
        col_img1, col_img2 = st.columns(2)

        # Original image
        with col_img1:
            st.markdown("**Original Image**")
            if os.path.exists(row["original_image_path"]):
                st.image(
                    Image.open(row["original_image_path"]),
                    use_container_width=True
                )
            else:
                st.warning("Original image not found")

        # Predicted image
        with col_img2:
            st.markdown("**Predicted Image**")
            if os.path.exists(row["predicted_image_path"]):
                st.image(
                    Image.open(row["predicted_image_path"]),
                    use_container_width=True
                )
            else:
                st.warning("Predicted image not found")

# ---------------- DOWNLOAD SECTION ----------------
st.divider()
st.subheader("⬇ Download & Export")

st.download_button(
    label="Download Detection Log (CSV)",
    data=df.to_csv(index=False),
    file_name="number_plate_detection_log.csv",
    mime="text/csv"
)

# ---------------- SUMMARY METRICS ----------------
st.divider()
st.subheader("📈 Quick Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Detections", len(df))

with col2:
    st.metric(
        "Average Threshold",
        f"{df['threshold'].mean():.2f}"
    )

with col3:
    st.metric(
        "Unique Plates Detected",
        df["ocr_text"].nunique()
    )

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Detection Reports • Download & Export Enabled</p>",
    unsafe_allow_html=True
)
