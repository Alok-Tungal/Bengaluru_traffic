import streamlit as st
# from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="🚦 Bangalore Traffic Dashboard", layout="wide")

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif; 
        }
        .main, .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 20px;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1F2937;
        }
    </style>
""", unsafe_allow_html=True)

# ============================
# SIDEBAR NAVIGATION
# ============================
with st.sidebar:
    selected = option_menu(
        "Main Menu", 
        ["🏠 Home", "📊 EDA", "📈 Visualizations", "🤖 Predict Traffic"],
        icons=['house', 'bar-chart', 'graph-up', 'cpu'],
        default_index=0
    )

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("bangalore_traffic.csv")   # <-- replace with your dataset path
    return df

data = load_data()

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    return joblib.load("best_rf_model.joblib")

model = load_model()

# ============================
# HOME TAB
# ============================
if selected == "🏠 Home":
    st.title("🚦 Bangalore Traffic Analysis & Prediction")
    st.markdown("""
        This dashboard provides:
        - Exploratory Data Analysis (EDA)  
        - Visualizations for traffic trends  
        - ML-powered Traffic Prediction  
    """)
    st.dataframe(data.head(10))

# ============================
# EDA TAB
# ============================
elif selected == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("Basic info about dataset:")
    st.write(data.describe())

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(data['DayOfWeek'].value_counts())
    with col2:
        st.bar_chart(data['Hour'].value_counts())

# ============================
# VISUALIZATION TAB
# ============================
elif selected == "📈 Visualizations":
    st.title("📈 Traffic Visualizations")

    fig = px.scatter(data, x="Traffic Volume", y="Average Speed",
                     color="Area Name", size="Congestion Level",
                     hover_data=["Road/Intersection Name"])
    st.plotly_chart(fig, use_container_width=True)

    fig2, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(x="DayOfWeek", y="Traffic Volume", data=data, ax=ax)
    st.pyplot(fig2)

# ============================
# PREDICTION TAB
# ============================
elif selected == "🤖 Predict Traffic":
    st.title("🚗 Predict Traffic Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        traffic_volume = st.number_input("Traffic Volume", min_value=1000, max_value=100000, step=1000)
        avg_speed = st.number_input("Average Speed (km/h)", min_value=0, max_value=120, step=1)

    with col2:
        tti = st.number_input("Travel Time Index", min_value=0.5, max_value=5.0, step=0.1)
        congestion = st.slider("Congestion Level (%)", 0, 100, 50)

    with col3:
        hour = st.slider("Hour of Day", 0, 23, 8)
        dayofweek = st.selectbox("Day of Week", [0,1,2,3,4,5,6])  # 0=Mon, 6=Sun

    # Predict button
    if st.button("🔮 Predict"):
        features = [[traffic_volume, avg_speed, tti, congestion, hour, dayofweek]]
        prediction = model.predict(features)
        st.success(f"Predicted Road Capacity Utilization: **{prediction[0]:.2f}%**")
