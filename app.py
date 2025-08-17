# ===== top of file =====
import streamlit as st
st.set_page_config(page_title="🚦 Bangalore Traffic Dashboard", layout="wide")

# Safe import for option_menu (with graceful fallback)
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
df=pd.read_csv("Banglore_cleaned_data.csv")
# ---------- Custom CSS ----------
st.markdown("""
    <style>
        body { background-color: #f9f9f9; font-family: 'Segoe UI', sans-serif; }
        .main, .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stButton>button {
            background-color: #4CAF50; color: white; border: none;
            border-radius: 8px; font-size: 16px; padding: 8px 20px;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #1F2937; }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar Navigation ----------
with st.sidebar:
    if HAS_OPTION_MENU:
        selected = option_menu(
            "Main Menu",
            ["🏠 Home", "📊 EDA", "📈 Visualizations", "🤖 Predict Traffic"],
            icons=['house', 'bar-chart', 'graph-up', 'cpu'],
            default_index=0
        )
    else:
        # Fallback if streamlit-option-menu is not installed
        selected = st.radio(
            "Main Menu",
            ["🏠 Home", "📊 EDA", "📈 Visualizations", "🤖 Predict Traffic"],
            index=0
        )
        st.caption("Install `streamlit-option-menu` for nicer sidebar:\n`pip install streamlit-option-menu`")

# ---------- Pages ----------
if selected == "🏠 Home":
    st.title("✅ Streamlit running on the correct port")
    st.subheader("🚦 Bangalore Traffic Dashboard")
    if not df.empty:
        st.write("Sample rows:")
        st.dataframe(df.head())
    else:
        st.info("Upload or place your `bangalore_traffic.csv` in the app folder to see data here.")

elif selected == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    if df.empty:
        st.warning("No data loaded.")
    else:
        st.write(df.describe(include='all'))
        col1, col2 = st.columns(2)
        with col1:
            if 'DayOfWeek' in df.columns:
                st.bar_chart(df['DayOfWeek'].value_counts())
        with col2:
            if 'Hour' in df.columns:
                st.bar_chart(df['Hour'].value_counts())

elif selected == "📈 Visualizations":
    st.title("📈 Visualizations")
    if df.empty:
        st.warning("No data loaded.")
    else:
        # Example: boxplot Traffic Volume by DayOfWeek
        if {'DayOfWeek','Traffic Volume'}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(8,4))
            sns.boxplot(data=df, x="DayOfWeek", y="Traffic Volume", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Add columns 'DayOfWeek' and 'Traffic Volume' for this plot.")

elif selected == "🤖 Predict Traffic":
    st.title("🤖 Predict Traffic (placeholder)")
    st.write("Hook your saved model here and build the input form.")
    st.caption("We can wire this to `best_rf_model.joblib` once your preprocessing is finalized.")
