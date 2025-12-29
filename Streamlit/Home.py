import streamlit as st

st.title("🏠 Welcome to Home")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Users", 2500, "+15%")
with col2:
    st.metric("Active Sessions", 342, "+8%")

st.write("This is your dashboard overview.")
