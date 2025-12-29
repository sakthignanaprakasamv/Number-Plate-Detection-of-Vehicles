import streamlit as st

st.title("⚙️ Settings")
st.sidebar.write("---")

threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.5)
auto_refresh = st.checkbox("Auto Refresh", value=True)

st.write(f"✓ Settings saved - Threshold: {threshold}")
