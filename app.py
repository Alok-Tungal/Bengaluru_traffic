import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="🚦 Bangalore Traffic Analysis", layout="wide")

st.title("🚦 Bangalore Traffic Dataset - Step 1")

# --- Load dataset directly from GitHub ---
# Replace the below link with your raw GitHub CSV file link
github_csv_url = "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/bangalore_traffic.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(github_csv_url)
    return df

df = load_data()

st.success("✅ Data Loaded from GitHub Successfully!")

# Display preview
st.subheader("🔎 Dataset Preview")
st.dataframe(df.head())

# Show basic info
st.subheader("📊 Dataset Shape")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
