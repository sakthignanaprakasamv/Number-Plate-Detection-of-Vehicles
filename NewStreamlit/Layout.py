import streamlit as st

# Page configuration
st.set_page_config(page_title="Streamlit Demo", layout="wide")
st.markdown("""
<style>

/* Sidebar header container */
div[data-testid="stLogoSpacer"] {
    height: 40px;                 /* smaller height */
    display: flex;
    align-items: center;
    padding-left: 12px;
    position: relative;
}

/* Logo + text */
div[data-testid="stLogoSpacer"]::before {
    content: "Sakthi";
    display: flex;
    align-items: center;
    gap: 10px;

    font-size: 16px;              /* smaller text */
    font-weight: 600;
    color: #ffffff;
    font-family: system-ui, -apple-system, BlinkMacSystemFont;

    padding-left: 44px;
    height: 32px;
    line-height: 32px;
}

/* Logo box */
div[data-testid="stLogoSpacer"]::after {
    content: "S";
    position: absolute;
    left: 0px;

    width: 32px;                  /* smaller logo */
    height: 32px;
    background-color: #e6e6e6;
    border-radius: 8px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 16px;
    font-weight: 600;
    color: #7a7a7a;
}


</style>
""", unsafe_allow_html=True)
st.divider()

# Sidebar navigation

pg = st.navigation(
    pages=[
        st.Page("Dashboard.py", title="Dashboard", icon="📈"),
        st.Page("ImageDetection.py", title="Image Detection", icon="🖼️"),
        st.Page("LiveCameraDetection.py", title="Live Camera Detection", icon="📷"),
        st.Page("Results.py", title="Detection Reports", icon="📊"),
    ]
)

# Run the selected page
pg.run()


