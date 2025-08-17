import streamlit as st
import pandas as pd

# Your raw GitHub CSV link (RAW format, not the HTML page link!)
url = "https://raw.githubusercontent.com/<your-username>/<repo-name>/main/<filename>.csv"

@st.cache_data
def load_data():
    try:
        # Try reading with default CSV format
        df = pd.read_csv(url, engine="python", on_bad_lines="skip")
    except Exception:
        # Fallback: try tab separator if comma fails
        df = pd.read_csv(url, sep="\t", engine="python", on_bad_lines="skip")

    # --- Basic cleaning ---
    df.columns = df.columns.str.strip()  # remove spaces
    df = df.dropna(how="all")            # drop empty rows
    df = df.fillna("Unknown")            # replace NaN
    return df

df = load_data()

st.write("✅ Dataset loaded successfully from GitHub")
st.dataframe(df.head())
