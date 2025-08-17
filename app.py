import streamlit as st
import pandas as pd

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🚦 Bangalore Traffic Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Bangalore Traffic Dataset - Step 1")

# ----------------------------
# Load Data from GitHub
# ----------------------------
github_csv_url = "https://github.com/Alok-Tungal/Bengaluru_traffic/edit/main/app.py"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(github_csv_url, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(github_csv_url, sep="\t", engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ----------------------------
# Dataset Overview Section
# ----------------------------
st.success("✅ Data Loaded from GitHub Successfully!")

# Metrics in columns
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", df.shape[0])
c2.metric("Total Columns", df.shape[1])
c3.metric("Missing Values", df.isnull().sum().sum())
c4.metric("Duplicate Rows", df.duplicated().sum())

# ----------------------------
# Preview + Schema
# ----------------------------
st.subheader("🔎 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("📑 Dataset Schema")
schema_df = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})
st.dataframe(schema_df, use_container_width=True)

# ----------------------------
# Sidebar Info
# ----------------------------
st.sidebar.header("ℹ️ About Step-1")
st.sidebar.write("""
This step loads the **Bangalore Traffic Dataset** directly from GitHub, 
cleans it, and provides a quick overview:
- Total rows & columns  
- Missing values count  
- Duplicate rows  
- Dataset schema  
""")
