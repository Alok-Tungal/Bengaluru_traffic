# app.py
import streamlit as st
st.set_page_config(page_title="🚦 Bangalore Traffic — RealTime Predictor", layout="wide")

# optional nicer sidebar menu
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

import pandas as pd
import numpy as np
import joblib, io, requests, time, os
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# ML helpers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# SHAP optional
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ----------------------
# CONFIG - change this to your raw GitHub CSV URL (or paste in sidebar)
GITHUB_RAW_URL = st.secrets.get("GITHUB_RAW_URL", "https://github.com/Alok-Tungal/Bengaluru_traffic/edit/main/app.py")
# ----------------------

# ---------- Styling ----------
st.markdown("""
    <style>
      body { background-color: #f7f8fb; font-family: 'Segoe UI', Roboto, sans-serif; }
      .stButton>button { background:#0f766e; color:white; border-radius:8px; }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Bangalore Traffic — Controls")
    github_url = st.text_input("GitHub raw CSV URL", value=GITHUB_RAW_URL)
    if st.button("Force reload data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.write("Model (optional)")
    uploaded_model = st.file_uploader("Upload pipeline/model (.joblib/.pkl)", type=["joblib","pkl"])
    st.markdown("If you upload a pipeline that includes preprocessing, app will use it directly.")
    st.markdown("---")
    st.caption("Presets: Rush Hour, Accident, Weekend → autofill the prediction form.")

# ---------- Robust loader ----------
@st.cache_data
def load_github_csv(url: str):
    if not url or "raw.githubusercontent" not in url:
        return pd.DataFrame()
    try:
        # try pandas read directly (works for raw)
        df = pd.read_csv(url, engine="python", on_bad_lines="skip")
        return df
    except Exception:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

df = load_github_csv(github_url)
if df is None or df.empty:
    st.warning("No data loaded from GitHub. Provide a valid raw URL or load sample data below.")
    if st.button("Load sample demo data"):
        SAMPLE = """Area Name,Road/Intersection Name,Traffic Volume,Average Speed,Travel Time Index,Congestion Level,Road Capacity Utilization,Incident Reports,Environmental Impact,Public Transport Usage,Traffic Signal Compliance,Parking Usage,Pedestrian and Cyclist Count,Weather Conditions,Roadwork and Construction Activity,Year,Month,Day,Hour,DayOfWeek,IsWeekend
Indiranagar,100 Feet Road,50590,50.23,1.5,100,100,0,151.18,70.63,84.04,85.40,111,Clear,No,2022,1,1,0,5,1
Indiranagar,CMH Road,30825,29.37,1.5,100,100,1,111.65,41.92,91.40,59.98,100,Clear,No,2022,1,1,0,5,1
"""
        df = pd.read_csv(io.StringIO(SAMPLE))
else:
    st.success(f"Loaded {len(df):,} rows from GitHub (last load: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")

# Basic preprocessing convenience
if {"Year","Month","Day","Hour"}.issubset(df.columns):
    try:
        df["timestamp"] = pd.to_datetime(df[["Year","Month","Day","Hour"]].rename(columns={"Hour":"hour"}), errors="coerce")
    except Exception:
        df["timestamp"] = pd.NaT
else:
    df["timestamp"] = pd.NaT

# ---------- Top KPIs ----------
st.title("🚦 Bangalore Traffic — RealTime Predictor")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{df.shape[0]:,}")
k2.metric("Columns", df.shape[1])
k3.metric("Avg Traffic Volume", f"{int(df['Traffic Volume'].mean()):,}" if "Traffic Volume" in df.columns else "—")
k4.metric("Last loaded", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

# ---------- Pages nav ----------
if HAS_OPTION_MENU:
    from streamlit_option_menu import option_menu
    page = option_menu(None, ["Live Dashboard","Predict Traffic","Train Model","About"], icons=["cloud","cpu","gear","info-circle"], default_index=0, orientation="horizontal")
else:
    page = st.radio("Page", ["Live Dashboard","Predict Traffic","Train Model","About"], horizontal=True)

# ------------------ Live Dashboard ------------------
if page == "Live Dashboard":
    st.header("📡 Live Dashboard & Playback")
    # filters
    areas = df['Area Name'].unique().tolist() if 'Area Name' in df.columns else []
    sel_areas = st.multiselect("Select Area(s)", options=areas, default=areas[:6])
    hr_min, hr_max = st.slider("Hour range", 0, 23, (0,23))
    df_f = df.copy()
    if sel_areas:
        df_f = df_f[df_f['Area Name'].isin(sel_areas)]
    if 'Hour' in df_f.columns:
        df_f = df_f[(df_f['Hour']>=hr_min) & (df_f['Hour']<=hr_max)]

    # Time series: daily aggregate (if timestamp exists)
    st.subheader("Traffic Volume — Time Series")
    if 'timestamp' in df_f.columns and not df_f['timestamp'].isna().all():
        ts = df_f.dropna(subset=['timestamp']).copy()
        ts['date'] = ts['timestamp'].dt.date
        agg = ts.groupby('date')['Traffic Volume'].sum().reset_index()
        fig = px.line(agg, x='date', y='Traffic Volume', title="Daily traffic volume (sum)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Timestamp not available; provide Year/Month/Day/Hour in CSV for time-series.")

    # Heatmap DayOfWeek x Hour
    st.subheader("Heatmap — Avg Traffic by DayOfWeek & Hour")
    if {'DayOfWeek','Hour','Traffic Volume'}.issubset(df_f.columns):
        pivot = df_f.pivot_table(index='DayOfWeek', columns='Hour', values='Traffic Volume', aggfunc='mean', fill_value=0)
        fig_h = px.imshow(pivot, aspect='auto', labels=dict(x='Hour', y='DayOfWeek', color='Avg Traffic'))
        st.plotly_chart(fig_h, use_container_width=True)

    # Top roads
    st.subheader("Top Roads by Avg Traffic")
    if 'Road/Intersection Name' in df_f.columns:
        top_n = st.slider("Top N", 5, 30, 10)
        top = df_f.groupby('Road/Intersection Name')['Traffic Volume'].mean().sort_values(ascending=False).head(top_n).reset_index()
        fig_bar = px.bar(top, x='Traffic Volume', y='Road/Intersection Name', orientation='h')
        st.plotly_chart(fig_bar, use_container_width=True)

    # Playback simulation
    st.subheader("Realtime Playback Simulation")
    sim_rows = st.slider("Rows to playback", 10, min(500, max(10,len(df_f))), 100)
    sim_interval = st.slider("Interval (sec)", 1, 3, 1)
    start = st.button("▶️ Start Playback")
    stop = st.button("⏹ Stop Playback")

    if 'play' not in st.session_state:
        st.session_state.play = False
    if start:
        st.session_state.play = True
    if stop:
        st.session_state.play = False

    placeholder = st.empty()
    if st.session_state.play:
        tail = df_f.tail(sim_rows).reset_index(drop=True)
        for i in range(len(tail)):
            if not st.session_state.play:
                break
            cur = tail.iloc[:i+1]
            # KPIs
            total_vol = int(cur['Traffic Volume'].sum())
            avg_speed = float(cur['Average Speed'].mean()) if 'Average Speed' in cur.columns else np.nan
            with placeholder.container():
                left, right = st.columns(2)
                left.metric("Live cumulative volume", f"{total_vol:,}")
                right.metric("Live avg speed", f"{avg_speed:.2f}" if not np.isnan(avg_speed) else "—")
                fig_play = px.line(cur, x=cur.index, y='Traffic Volume', title="Live playback slice")
                st.plotly_chart(fig_play, use_container_width=True)
            time.sleep(sim_interval)
        st.session_state.play = False
        placeholder.empty()

# ------------------ Predict Traffic ------------------
elif page == "Predict Traffic":
    st.header("🔮 Predict Traffic Volume — Single Sample (Real-time style form)")

    # If a model file was uploaded in sidebar, load it
    model = None
    preprocessor = None
    if uploaded_model is not None:
        try:
            b = uploaded_model.read()
            tmp = "/tmp/uploaded_model.joblib"
            with open(tmp, "wb") as fh:
                fh.write(b)
            loaded = joblib.load(tmp)
            # if uploaded is a dict/pipeline with preprocessor/model saved as dict
            if isinstance(loaded, dict) and 'model' in loaded and 'preprocessor' in loaded:
                preprocessor = loaded['preprocessor']
                model = loaded['model']
            else:
                model = loaded
            st.success("Uploaded model loaded.")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

    # If no uploaded model but a saved file exists in working dir, try to load it
    if model is None:
        for fname in ["best_rf_model.joblib", "rf_bangalore_model.joblib", "pipeline.joblib"]:
            if os.path.exists(fname):
                try:
                    loaded = joblib.load(fname)
                    if isinstance(loaded, dict) and 'model' in loaded and 'preprocessor' in loaded:
                        preprocessor = loaded['preprocessor']; model = loaded['model']
                    else:
                        model = loaded
                    st.info(f"Loaded model from {fname}")
                    break
                except Exception:
                    continue

    # Build a nice input form with presets
    st.markdown("### Input scenario (use presets for quick fill)")
    preset = st.selectbox("Preset", ["-- Select --", "Normal", "Rush Hour", "Accident", "Weekend"])
    presets = {
        "Normal": {"Average Speed":40, "Travel Time Index":1.2, "Congestion Level":50, "Road Capacity Utilization":60, "Incident Reports":0},
        "Rush Hour": {"Average Speed":18, "Travel Time Index":2.0, "Congestion Level":90, "Road Capacity Utilization":95, "Incident Reports":0},
        "Accident": {"Average Speed":8, "Travel Time Index":3.5, "Congestion Level":100, "Road Capacity Utilization":100, "Incident Reports":2},
        "Weekend": {"Average Speed":45, "Travel Time Index":1.0, "Congestion Level":40, "Road Capacity Utilization":50, "Incident Reports":0}
    }

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        defaults = presets.get(preset, {})
        with col1:
            traffic_volume = st.number_input("Traffic Volume (vehicles)", min_value=0, value=int(defaults.get("Traffic Volume", df['Traffic Volume'].median() if 'Traffic Volume' in df.columns else 30000)))
            avg_speed = st.number_input("Average Speed (km/h)", min_value=0.0, value=float(defaults.get("Average Speed", df['Average Speed'].median() if 'Average Speed' in df.columns else 30.0)))
            tti = st.number_input("Travel Time Index", min_value=0.1, value=float(defaults.get("Travel Time Index", 1.5)))
            congestion = st.slider("Congestion Level (%)", 0, 100, int(defaults.get("Congestion Level", 50)))
        with col2:
            capacity = st.slider("Road Capacity Utilization (%)", 0, 100, int(defaults.get("Road Capacity Utilization", 60)))
            incidents = st.number_input("Incident Reports", min_value=0, value=int(defaults.get("Incident Reports", 0)))
            env_imp = st.number_input("Environmental Impact (index)", min_value=0.0, value=float(df['Environmental Impact'].median() if 'Environmental Impact' in df.columns else 100.0))
            public_trans = st.slider("Public Transport Usage (%)", 0, 100, int(df['Public Transport Usage'].median() if 'Public Transport Usage' in df.columns else 50))

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        X_new = pd.DataFrame([{
            "Traffic Volume": traffic_volume,
            "Average Speed": avg_speed,
            "Travel Time Index": tti,
            "Congestion Level": congestion,
            "Road Capacity Utilization": capacity,
            "Incident Reports": incidents,
            "Environmental Impact": env_imp,
            "Public Transport Usage": public_trans
        }])

        # If model is pipeline expecting raw features, predict directly. If not, try preprocessor.
        try:
            if model is None:
                st.error("No model available. Train in 'Train Model' or upload a pipeline.")
            else:
                if preprocessor is not None:
                    X_en = preprocessor.transform(X_new)
                    yhat = model.predict(X_en)[0]
                else:
                    # try direct predict (maybe pipeline)
                    yhat = model.predict(X_new)[0]
                st.success(f"Predicted Traffic Volume: {yhat:,.0f}")

                # Advice based on prediction quantiles
                q75 = df['Traffic Volume'].quantile(0.75) if 'Traffic Volume' in df.columns else None
                q95 = df['Traffic Volume'].quantile(0.95) if 'Traffic Volume' in df.columns else None
                if q95 and yhat > q95:
                    st.warning("🚨 Very high traffic predicted — consider rerouting / traffic control.")
                elif q75 and yhat > q75:
                    st.info("⚠️ High traffic predicted — increase public transport frequency.")

                # Show feature importance if present
                if hasattr(model, "feature_importances_"):
                    feat_names = preprocessor.get_feature_names_out() if preprocessor is not None else X_new.columns.tolist()
                    fi = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)
                    fig = px.bar(fi.reset_index().rename(columns={'index':'feature', 0:'importance'}), x='importance', y='feature', orientation='h', title="Top features")
                    st.plotly_chart(fig, use_container_width=True)

                # SHAP (optional)
                if HAS_SHAP and isinstance(model, (RandomForestRegressor,)):
                    if st.checkbox("Explain prediction with SHAP"):
                        with st.spinner("Computing SHAP..."):
                            try:
                                explainer = shap.TreeExplainer(model)
                                X_for_shap = preprocessor.transform(X_new) if preprocessor is not None else X_new
                                shap_vals = explainer.shap_values(X_for_shap)
                                # simple bar of absolute SHAP contributions
                                arr = np.abs(shap_vals).mean(axis=0) if shap_vals.ndim>1 else np.abs(shap_vals).mean(axis=0)
                                shap_df = pd.DataFrame({'feature': (preprocessor.get_feature_names_out() if preprocessor is not None else X_new.columns), 'value':arr})
                                shap_df = shap_df.sort_values('value', ascending=False).head(15)
                                fig_sh = px.bar(shap_df, x='value', y='feature', orientation='h', title="SHAP (mean abs) top features")
                                st.plotly_chart(fig_sh, use_container_width=True)
                            except Exception as e:
                                st.error(f"SHAP failed: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------ Train Model (quick) ------------------
elif page == "Train Model":
    st.header("⚙️ Quick Train RandomForest (optional)")
    st.info("This trains a RandomForest (RobustScaler + OHE) on the currently loaded dataset.")
    if 'Traffic Volume' not in df.columns:
        st.error("No 'Traffic Volume' column present; cannot train.")
    else:
        if st.button("Train RandomForest now"):
            # auto-detect categorical cols
            categorical = df.select_dtypes(include=['object','category']).columns.tolist()
            if 'Traffic Volume' in categorical: categorical.remove('Traffic Volume')
            numeric = [c for c in df.columns if c not in categorical + ['timestamp','Traffic Volume']]
            X = df[numeric + categorical]
            y = df['Traffic Volume']
            # safe OHE
            try:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            preprocessor = ColumnTransformer([('num', RobustScaler(), numeric), ('cat', ohe, categorical)], remainder='drop')
            X_en = preprocessor.fit_transform(X)
            Xtr, Xte, ytr, yte = train_test_split(X_en, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            rf.fit(Xtr, ytr)
            st.session_state['preprocessor'] = preprocessor
            st.session_state['model'] = rf
            st.success("Trained RandomForest and saved to session.")

            ypred = rf.predict(Xte)
            st.write("R2:", r2_score(yte, ypred))
            st.write("RMSE:", mean_squared_error(yte, ypred, squared=False))

# ------------------ About ------------------
else:
    st.header("About")
    st.markdown("""
    **Bangalore Traffic — RealTime Predictor**  
    - Loads CSV directly from GitHub raw URL (no upload needed)  
    - Real-time playback simulation + prediction form  
    - Supports uploading a saved pipeline (.joblib) that includes preprocessing  
    - Optional SHAP explainability
    """)

# Save a trained model for download
if 'model' in st.session_state and st.button("Export model (.joblib)"):
    tmpn = "exported_bangalore_model.joblib"
    # store both preprocessor and model as dict if available
    tosave = {}
    if st.session_state.get('preprocessor') is not None:
        tosave['preprocessor'] = st.session_state['preprocessor']
    tosave['model'] = st.session_state['model']
    joblib.dump(tosave, tmpn)
    with open(tmpn, "rb") as f:
        st.download_button("Download pipeline", f, file_name=tmpn)
    os.remove(tmpn)
