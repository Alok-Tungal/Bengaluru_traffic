import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_rf_model.joblib")

model = load_model()

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Bangalore Traffic Predictor 🚦", layout="wide")

st.title("🚗 Bangalore Traffic Congestion Prediction")
st.markdown("### AI-powered prediction of traffic congestion levels in Bangalore")

# Sidebar Info
st.sidebar.header("ℹ️ About the Project")
st.sidebar.write("""
This app uses a **Random Forest model** trained on Bangalore traffic data to predict congestion.
You can adjust parameters like traffic volume, speed, and road capacity to simulate scenarios.
""")

# -----------------------------
# User Input Form
# -----------------------------
st.subheader("📊 Enter Traffic Data")

with st.form("traffic_form"):
    col1, col2 = st.columns(2)

    with col1:
        traffic_volume = st.number_input("Traffic Volume (vehicles/hr)", 1000, 100000, 35000)
        avg_speed = st.number_input("Average Speed (km/h)", 5, 120, 30)
        tti = st.number_input("Travel Time Index", 0.5, 5.0, 1.5)
        congestion_level = st.slider("Congestion Level (%)", 0, 100, 70)
        road_capacity = st.slider("Road Capacity Utilization (%)", 0, 100, 75)

    with col2:
        incidents = st.number_input("Incident Reports", 0, 20, 1)
        env_impact = st.number_input("Environmental Impact Index", 0, 500, 120)
        public_transport = st.slider("Public Transport Usage (%)", 0, 100, 50)
        signal_compliance = st.slider("Traffic Signal Compliance (%)", 0, 100, 80)
        parking_usage = st.slider("Parking Usage (%)", 0, 100, 40)
        pedestrians = st.number_input("Pedestrian & Cyclist Count", 0, 1000, 120)

    submitted = st.form_submit_button("🚦 Predict Traffic")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Traffic Volume": traffic_volume,
        "Average Speed": avg_speed,
        "Travel Time Index": tti,
        "Congestion Level": congestion_level,
        "Road Capacity Utilization": road_capacity,
        "Incident Reports": incidents,
        "Environmental Impact": env_impact,
        "Public Transport Usage": public_transport,
        "Traffic Signal Compliance": signal_compliance,
        "Parking Usage": parking_usage,
        "Pedestrian and Cyclist Count": pedestrians
    }])

    prediction = model.predict(input_data)[0]
    
    st.success(f"### 🚥 Predicted Congestion Score: **{prediction:.2f}**")
    st.markdown("---")

    # -----------------------------
    # Visualization
    # -----------------------------
    st.subheader("📈 Input Data Overview")
    fig = px.bar(input_data.T, orientation="h", labels={"index": "Feature", "value": "Value"}, title="Traffic Input Profile")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("🔑 Feature Importance (from Random Forest)")
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Model Feature Importance")
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # SHAP Explanations
    # -----------------------------
    st.subheader("🧠 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(fig)

