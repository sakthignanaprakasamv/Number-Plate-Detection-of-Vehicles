import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Performance Dashboard", layout="wide")
st.title("📈 Performance Dashboard")

st.markdown(
    "This dashboard provides performance insights, confidence analysis, "
    "and detection trends for the Number Plate Detection system."
)

# ---------------- PATH ----------------
CSV_PATH = "data/detection_log.csv"

# ---------------- LOAD DATA ----------------
if not os.path.exists(CSV_PATH):
    st.warning("No detection data available. Run Image Detection first.")
    st.stop()

df = pd.read_csv(CSV_PATH)

if df.empty:
    st.info("Detection log is empty.")
    st.stop()

# ---------------- METRICS ----------------
st.subheader("🔢 Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Detections", len(df))

with col2:
    st.metric(
        "Average Confidence",
        f"{df['threshold'].mean():.2f}"
    )

with col3:
    st.metric(
        "Highest Confidence",
        f"{df['threshold'].max():.2f}"
    )

with col4:
    st.metric(
        "Unique Plates",
        df["ocr_text"].nunique()
    )

# ---------------- CONFIDENCE DISTRIBUTION ----------------
st.divider()
st.subheader("📊 Confidence Threshold Distribution")

fig, ax = plt.subplots()
ax.hist(df["threshold"], bins=10)
ax.set_xlabel("Confidence Threshold")
ax.set_ylabel("Detection Count")
ax.set_title("Distribution of Confidence Thresholds")

st.pyplot(fig)

# ---------------- DETECTION OVER TIME ----------------
st.divider()
st.subheader("🕒 Detection Trend Over Time")

# Convert datetime column
df["entry_datetime"] = pd.to_datetime(df["entry_datetime"])

# Group by date
daily_counts = df.groupby(df["entry_datetime"].dt.date).size()

fig2, ax2 = plt.subplots()
ax2.plot(daily_counts.index, daily_counts.values, marker="o")
ax2.set_xlabel("Date")
ax2.set_ylabel("Number of Detections")
ax2.set_title("Detections per Day")
ax2.grid(True)

st.pyplot(fig2)

# ---------------- OCR FREQUENCY ----------------
st.divider()
st.subheader("🔤 Most Detected Plate Numbers")

ocr_counts = (
    df["ocr_text"]
    .value_counts()
    .head(10)
)

st.bar_chart(ocr_counts)

# ---------------- TABLE VIEW ----------------
st.divider()
st.subheader("📋 Raw Detection Data (Preview)")

st.dataframe(
    df.sort_values("entry_datetime", ascending=False),
    use_container_width=True
)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Performance Dashboard • Metrics & Trends</p>",
    unsafe_allow_html=True
)
