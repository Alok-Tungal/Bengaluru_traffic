# app.py  -- Step 1: Basic Setup & Data Load
import streamlit as st
st.set_page_config(page_title="🚦 Bangalore Traffic — Step 1: Data Load", layout="wide")

# optional nice sidebar menu (fallback handled)
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

import pandas as pd
import numpy as np
import requests, io
from datetime import datetime

# ---------- Small default sample (used if GitHub URL not provided) ----------
SAMPLE_CSV = """Area Name,Road/Intersection Name,Traffic Volume,Average Speed,Travel Time Index,Congestion Level,Road Capacity Utilization,Incident Reports,Environmental Impact,Public Transport Usage,Traffic Signal Compliance,Parking Usage,Pedestrian and Cyclist Count,Weather Conditions,Roadwork and Construction Activity,Year,Month,Day,Hour,DayOfWeek,IsWeekend
Indiranagar,100 Feet Road,50590,50.230299,1.5,100,100,0,151.18,70.63233,84.0446,85.403629,111,Clear,No,2022,1,1,0,5,1
Indiranagar,CMH Road,30825,29.377125,1.5,100,100,1,111.65,41.924899,91.407038,59.983689,100,Clear,No,2022,1,1,0,5,1
"""

# ---------- CSS for a cleaner look ----------
st.markdown("""
    <style>
        body { background-color: #f8fafc; font-family: 'Segoe UI', Roboto, sans-serif; }
        .block-container { padding: 1rem 1.5rem; }
        .stButton>button { background:#0f766e; color:white; border-radius:8px; padding:6px 12px; }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Bangalore Traffic — Step 1")
    st.markdown("Load CSV from GitHub (raw URL). If empty, sample data will be used.")
    github_url = st.text_input("GitHub raw CSV URL", value="")
    if st.button("Force reload (clear cache)"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("Helpful:")
    st.caption("Use raw GitHub URL e.g. https://raw.githubusercontent.com/username/repo/main/bangalore_traffic.csv")
    st.markdown("---")
    if not HAS_OPTION_MENU:
        st.caption("Optional: install `streamlit-option-menu` for nicer sidebar UI (pip install streamlit-option-menu)")

# ---------- Data loader (cached) ----------
@st.cache_data
def load_from_github(url: str):
    """Return dataframe loaded from GitHub raw URL or empty df on failure."""
    if not url or url.strip() == "":
        return pd.DataFrame()
    try:
        # try direct pandas first (works with raw URLs)
        df = pd.read_csv(url)
        return df
    except Exception:
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception:
            return pd.DataFrame()

# Load df (either from github url or fallback sample)
df = load_from_github(github_url)
if df is None or df.empty:
    st.warning("No valid GitHub CSV loaded. Using sample dataset for demonstration.")
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))
    used_sample = True
else:
    used_sample = False

# ---------- Header + quick KPIs ----------
st.title("🚦 Bangalore Traffic — Data Load & Diagnostics (Step 1)")
st.markdown(f"**Data source:** {'sample' if used_sample else 'GitHub raw URL'}")
st.markdown(f"**Loaded at:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
if "Traffic Volume" in df.columns:
    c3.metric("Mean Traffic Volume", f"{df['Traffic Volume'].mean():,.0f}")
else:
    c3.metric("Mean Traffic Volume", "—")

st.markdown("---")

# ---------- Dataset preview ----------
st.subheader("Data preview")
st.dataframe(df.head(10))

# ---------- Column types and detection ----------
st.subheader("Column types & detection")
col1, col2 = st.columns([2,1])

with col1:
    dtypes = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index().rename(columns={"index":"column"})
    st.dataframe(dtypes, height=250)
with col2:
    # detected nominal vs numeric
    nominal_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write("🟦 Detected categorical (object/category):")
    st.write(nominal_cols if nominal_cols else "None")
    st.write("🟩 Detected numeric (number types):")
    st.write(numeric_cols if numeric_cols else "None")

st.markdown("---")

# ---------- Missing values and duplicates ----------
st.subheader("Missing values & duplicates")
mv = df.isnull().sum().sort_values(ascending=False)
st.write("Top missing values (column: missing_count)")
st.write(mv[mv>0].head(20) if (mv>0).any() else "No missing values detected.")

dups = df.duplicated().sum()
st.write(f"Duplicate rows count: **{dups}**")

st.markdown("---")

# ---------- Quick per-column summaries ----------
st.subheader("Quick summaries")

col_sel = st.selectbox("Select a column to inspect", options=df.columns.tolist())
if pd.api.types.is_numeric_dtype(df[col_sel]):
    st.write("Numeric summary:")
    st.write(df[col_sel].describe().to_frame().T)
    st.write("Histogram:")
    st.bar_chart(df[col_sel].dropna())
else:
    st.write("Categorical summary (top values)")
    st.write(df[col_sel].value_counts().head(20))

st.markdown("---")

# ---------- Download & next steps ----------
st.subheader("Download / Next steps")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download current dataset (CSV)", data=csv, file_name="bangalore_traffic_preview.csv", mime="text/csv")

st.markdown("""
### Ready for Step 2 — EDA
Now that data is loaded and checked, next steps (Step 2) will add:
- interactive time-series and heatmap visualizations,  
- filters for Area/Hour/DayOfWeek,  
- outlier detection and boxplots, and  
- initial feature engineering for ML.

If everything looks correct here (column names/types, no unexpected missing data), I will proceed to **Step 2: EDA & Visualizations** and produce the next Streamlit module.
""")
