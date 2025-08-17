# # ===== top of file =====
# import streamlit as st
# st.set_page_config(page_title="🚦 Bangalore Traffic Dashboard", layout="wide")

# # Safe import for option_menu (with graceful fallback)
# try:
#     from streamlit_option_menu import option_menu
#     HAS_OPTION_MENU = True
# except Exception:
#     HAS_OPTION_MENU = False

# import pandas as pd
# import matplotlib.pyplot as plt
# # import seaborn as sns
# df=pd.read_csv("Banglore_cleaned_data.csv")
# # ---------- Custom CSS ----------
# st.markdown("""
#     <style> 
#         body { background-color: #f9f9f9; font-family: 'Segoe UI', sans-serif; }
#         .main, .block-container { padding-top: 1rem; padding-bottom: 1rem; }
#         .stButton>button {
#             background-color: #4CAF50; color: white; border: none;
#             border-radius: 8px; font-size: 16px; padding: 8px 20px;
#         }
#         .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #1F2937; }
#     </style>
# """, unsafe_allow_html=True)

# # ---------- Sidebar Navigation ----------
# with st.sidebar:
#     if HAS_OPTION_MENU:
#         selected = option_menu(
#             "Main Menu",
#             ["🏠 Home", "📊 EDA", "📈 Visualizations", "🤖 Predict Traffic"],
#             icons=['house', 'bar-chart', 'graph-up', 'cpu'],
#             default_index=0
#         )
#     else:
#         # Fallback if streamlit-option-menu is not installed
#         selected = st.radio(
#             "Main Menu",
#             ["🏠 Home", "📊 EDA", "📈 Visualizations", "🤖 Predict Traffic"],
#             index=0
#         )
#         st.caption("Install `streamlit-option-menu` for nicer sidebar:\n`pip install streamlit-option-menu`")

# # ---------- Pages ----------
# if selected == "🏠 Home":
#     st.title("✅ Streamlit running on the correct port")
#     st.subheader("🚦 Bangalore Traffic Dashboard")
#     if not df.empty:
#         st.write("Sample rows:")
#         st.dataframe(df.head())
#     else:
#         st.info("Upload or place your `bangalore_traffic.csv` in the app folder to see data here.")

# elif selected == "📊 EDA":
#     st.title("📊 Exploratory Data Analysis")
#     if df.empty:
#         st.warning("No data loaded.")
#     else:
#         st.write(df.describe(include='all'))
#         col1, col2 = st.columns(2)
#         with col1:
#             if 'DayOfWeek' in df.columns:
#                 st.bar_chart(df['DayOfWeek'].value_counts())
#         with col2:
#             if 'Hour' in df.columns:
#                 st.bar_chart(df['Hour'].value_counts())

# elif selected == "📈 Visualizations":
#     st.title("📈 Visualizations")
#     if df.empty:
#         st.warning("No data loaded.")
#     else:
#         # Example: boxplot Traffic Volume by DayOfWeek
#         if {'DayOfWeek','Traffic Volume'}.issubset(df.columns):
#             fig, ax = plt.subplots(figsize=(8,4))
#             sns.boxplot(data=df, x="DayOfWeek", y="Traffic Volume", ax=ax)
#             st.pyplot(fig) 
#         else:
#             st.info("Add columns 'DayOfWeek' and 'Traffic Volume' for this plot.")

# elif selected == "🤖 Predict Traffic":
#     st.title("🤖 Predict Traffic (placeholder)")
#     st.write("Hook your saved model here and build the input form.")
#     st.caption("We can wire this to `best_rf_model.joblib` once your preprocessing is finalized.")

# app.py
import streamlit as st
st.set_page_config(page_title="🚦 Bangalore Traffic — RealTime Visual + ML", layout="wide")

# ---- Optional widget import (fallback handled) ----
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# ---- standard imports ----
import pandas as pd
import numpy as np
import requests, io, time, joblib, os
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# ML imports (used in Train page)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -------------------------
# CONFIG: set your GitHub raw CSV URL here
# (replace with your actual raw file URL)
GITHUB_RAW_URL = st.secrets.get("GITHUB_RAW_URL", "https://raw.githubusercontent.com/your-username/your-repo/main/bangalore_traffic.csv")
# -------------------------

# Custom CSS for nicer look
st.markdown(
    """
    <style>
      body { background-color: #fbfbfc; font-family: 'Segoe UI', Roboto, sans-serif; }
      .block-container { padding: 1rem 1.5rem; }
      .stButton>button { background:#0f766e; color:white; border-radius:8px; padding:6px 12px; }
      .kpi { background: white; border-radius:10px; padding:12px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("Bangalore Traffic")
    st.write("Data source: GitHub raw CSV")
    url_input = st.text_input("GitHub raw CSV URL", value=GITHUB_RAW_URL)
    refresh_interval = st.number_input("Refresh TTL (s) — cache age", min_value=10, max_value=3600, value=60, step=10)
    auto_simulate = st.checkbox("Enable auto-simulate (playback mode)", value=False)
    st.markdown("---")
    st.write("Model & Export")
    model_upload = st.file_uploader("Upload pipeline / model (.joblib/.pkl) optional", type=["joblib","pkl"])
    st.markdown("---")
    st.write("Controls")
    if st.button("Force reload from GitHub"):
        # clear cache and reload
        st.cache_data.clear()
        st.experimental_rerun()
    st.write("Tip: update CSV in your GitHub repo and press Force reload to pull latest data.")

# -------------------------
# Data loader (cache by URL + reload flag)

model = joblib.load("best_rf_model.zip")
# -------------------------
@st.cache_data
def load_from_github(url: str, force_reload: bool = False):
    """
    Load CSV from GitHub raw URL. Pass force_reload=True to bypass cache (useful when reloading).
    """
    if url.strip() == "":
        return pd.DataFrame()
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception as e:
        st.error(f"Failed to load from GitHub: {e}")
        return pd.DataFrame()

# load data (normal)
df = load_from_github(url_input, force_reload=False)

# record last load time
if df is None or df.empty:
    data_loaded = False
    last_load_time = None
else:
    data_loaded = True
    last_load_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# allow optional model load
if model_upload is not None:
    try:
        bytes_read = model_upload.read()
        tmp_model_path = "/tmp/uploaded_model.joblib"
        with open(tmp_model_path, "wb") as fh:
            fh.write(bytes_read)
        loaded_model = joblib.load(tmp_model_path)
        st.sidebar.success("Model uploaded and loaded.")
        st.session_state['uploaded_model'] = loaded_model
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# if user uploaded a pipeline earlier in session, prefer that
model_for_predict = st.session_state.get('uploaded_model', None)

# -------------------------
# Top row KPIs
# -------------------------
st.title("🚦 Bangalore Traffic — Live Visualizations & ML")
if not data_loaded:
    st.warning("No data loaded from GitHub. Please check the URL or press Force reload.")
    st.info("You can also paste your GitHub raw URL in the sidebar.")
    # offer sample load
    if st.button("Load sample data into session"):
        st.session_state.raw_df = pd.read_csv(io.StringIO("""Area Name,Road/Intersection Name,Traffic Volume,Average Speed,Travel Time Index,Congestion Level,Road Capacity Utilization,Incident Reports,Environmental Impact,Public Transport Usage,Traffic Signal Compliance,Parking Usage,Pedestrian and Cyclist Count,Weather Conditions,Roadwork and Construction Activity,Year,Month,Day,Hour,DayOfWeek,IsWeekend
Indiranagar,100 Feet Road,50590,50.230299,1.5,100,100,0,151.18,70.63233,84.0446,85.403629,111,Clear,No,2022,1,1,0,5,1
Indiranagar,CMH Road,30825,29.377125,1.5,100,100,1,111.65,41.924899,91.407038,59.983689,100,Clear,No,2022,1,1,0,5,1
"""))
        st.experimental_rerun()
    st.stop()

# show last load time and row/col
c1,c2,c3,c4 = st.columns(4)
total_rows = len(df)
c1.metric("Rows", total_rows)
c2.metric("Columns", df.shape[1])
c3.metric("Last loaded", last_load_time or "—")
# quick KPI: mean traffic volume
mean_tv = df["Traffic Volume"].mean() if "Traffic Volume" in df.columns else np.nan
c4.metric("Avg Traffic Volume", f"{mean_tv:,.0f}" if not np.isnan(mean_tv) else "—")

# -------------------------
# Quick preprocessing conveniences
# -------------------------
# Create datetime if possible
if {"Year","Month","Day","Hour"}.issubset(df.columns):
    try:
        df['timestamp'] = pd.to_datetime(df[['Year','Month','Day','Hour']].rename(columns={'Hour':'hour'}), errors='coerce')
    except Exception:
        # fallback: create date string then parse
        df['timestamp'] = pd.to_datetime(df[['Year','Month','Day']].astype(str).agg('-'.join, axis=1), errors='coerce')
else:
    df['timestamp'] = pd.NaT

# cast DayOfWeek/IsWeekend numeric if present
if 'DayOfWeek' in df.columns:
    df['DayOfWeek'] = pd.to_numeric(df['DayOfWeek'], errors='coerce')
if 'IsWeekend' in df.columns:
    df['IsWeekend'] = pd.to_numeric(df['IsWeekend'], errors='coerce')

# -------------------------
# Sidebar filter (interactive)
# -------------------------
st.sidebar.markdown("### Filters")
area_list = df['Area Name'].unique().tolist() if 'Area Name' in df.columns else []
sel_area = st.sidebar.multiselect("Select Area(s)", options=area_list, default=area_list[:6])
hour_range = st.sidebar.slider("Hour range", 0, 23, (0,23))
days = st.sidebar.multiselect("DayOfWeek", options=sorted(df['DayOfWeek'].dropna().unique().astype(int).tolist()) if 'DayOfWeek' in df.columns else [], default=sorted(df['DayOfWeek'].dropna().unique().astype(int).tolist()) if 'DayOfWeek' in df.columns else [])

# apply filters
df_filtered = df.copy()
if sel_area:
    df_filtered = df_filtered[df_filtered['Area Name'].isin(sel_area)]
df_filtered = df_filtered[(df_filtered['Hour']>=hour_range[0]) & (df_filtered['Hour']<=hour_range[1])]
if days:
    df_filtered = df_filtered[df_filtered['DayOfWeek'].isin(days)]

# -------------------------
# Visualization section
# -------------------------
st.header("Interactive Visualizations")

# 1) Time series (daily aggregate)
st.subheader("Traffic Volume — Time Series (daily aggregated)")
if 'timestamp' in df_filtered.columns and not df_filtered['timestamp'].isna().all():
    df_ts = df_filtered.dropna(subset=['timestamp']).copy()
    df_ts['date'] = df_ts['timestamp'].dt.date
    ts_agg = df_ts.groupby('date')['Traffic Volume'].sum().reset_index()
    fig_ts = px.line(ts_agg, x='date', y='Traffic Volume', title="Daily Traffic Volume")
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("Timestamp columns not detected or invalid (Year/Month/Day/Hour required).")

# 2) Hour vs Day Heatmap
st.subheader("Heatmap — Average Traffic Volume by DayOfWeek & Hour")
if {'DayOfWeek','Hour','Traffic Volume'}.issubset(df_filtered.columns):
    pivot = df_filtered.pivot_table(index='DayOfWeek', columns='Hour', values='Traffic Volume', aggfunc='mean', fill_value=0)
    fig_hm = px.imshow(pivot, aspect='auto', labels=dict(x="Hour", y="DayOfWeek", color="Avg Traffic"))
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Need columns DayOfWeek, Hour, and Traffic Volume to show heatmap.")

# 3) Top Roads/Intersections
st.subheader("Top Roads / Intersections by Average Traffic Volume")
if 'Road/Intersection Name' in df_filtered.columns:
    top_n = st.slider("Top N", 5, 30, 10)
    top_roads = df_filtered.groupby('Road/Intersection Name')['Traffic Volume'].mean().sort_values(ascending=False).head(top_n)
    fig_bar = px.bar(top_roads.reset_index(), x='Traffic Volume', y='Road/Intersection Name', orientation='h', title="Top Roads by Avg Traffic")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No Road/Intersection Name column found.")

# 4) Scatter: Average Speed vs Traffic Volume
st.subheader("Average Speed vs Traffic Volume (scatter)")
if {'Average Speed','Traffic Volume'}.issubset(df_filtered.columns):
    fig_sc = px.scatter(df_filtered, x='Traffic Volume', y='Average Speed', color='Area Name' if 'Area Name' in df_filtered.columns else None,
                        hover_data=['Road/Intersection Name'] if 'Road/Intersection Name' in df_filtered.columns else None,
                        title="Traffic Volume vs Average Speed")
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("Need Average Speed and Traffic Volume columns for scatter.")

# 5) Distribution and Boxplots
st.subheader("Distribution & Boxplots")
col1, col2 = st.columns(2)
with col1:
    if 'Traffic Volume' in df_filtered.columns:
        fig_dist = px.histogram(df_filtered, x='Traffic Volume', nbins=50, title="Traffic Volume Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)
with col2:
    if {'DayOfWeek','Traffic Volume'}.issubset(df_filtered.columns):
        fig_box = px.box(df_filtered, x='DayOfWeek', y='Traffic Volume', title="Traffic Volume by DayOfWeek")
        st.plotly_chart(fig_box, use_container_width=True)

# 6) Correlation heatmap
st.subheader("Correlation (numeric features)")
numcols = df_filtered.select_dtypes(include=np.number).columns.tolist()
if len(numcols) >= 3:
    corr = df_filtered[numcols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", title="Correlation matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# Real-time simulation (playback)
# -------------------------
st.header("Realtime Simulation / Live Playback")
sim_container = st.empty()
sim_controls = st.columns([1,1,1,2])
with sim_controls[0]:
    sim_play = st.button("▶️ Start Playback")
with sim_controls[1]:
    sim_stop = st.button("⏹ Stop Playback")
with sim_controls[2]:
    sim_speed = st.slider("Interval (sec)", 1, 5, 2)
with sim_controls[3]:
    sim_rows = st.slider("Use last N rows to playback", 10, min(500, len(df)), 100)

if sim_play:
    st.session_state['sim_stop'] = False
if sim_stop:
    st.session_state['sim_stop'] = True

if st.session_state.get('sim_stop', True) == False:
    # playback
    tail = df_filtered.tail(sim_rows).reset_index(drop=True)
    placeholder = sim_container.empty()
    for i in range(len(tail)):
        if st.session_state.get('sim_stop', False):
            break
        current = tail.iloc[:i+1]
        # small live KPI
        live_volume = int(current['Traffic Volume'].sum())
        live_avg_speed = float(current['Average Speed'].mean()) if 'Average Speed' in current.columns else np.nan
        # draw
        with placeholder.container():
            r1, r2 = st.columns(2)
            r1.metric("Live cumulative volume", f"{live_volume:,}")
            r2.metric("Live avg speed", f"{live_avg_speed:.2f}" if not np.isnan(live_avg_speed) else "—")
            fig = px.line(current, x=current.index, y='Traffic Volume', title="Playback Traffic Volume (recent slice)")
            st.plotly_chart(fig, use_container_width=True)
        time.sleep(sim_speed)
    st.session_state['sim_stop'] = True
    sim_container.empty()

# -------------------------
# Train Model quick (optional)
# -------------------------
st.header("Model: Quick Train (optional)")
if st.button("Train quick RandomForest on current filtered data"):
    # pick features automatically
    df_train = df_filtered.copy()
    target = "Traffic Volume"
    if target not in df_train.columns:
        st.error("Traffic Volume column not present to train on.")
    else:
        # detect nominal cols
        nominal = df_train.select_dtypes(include=['object','category']).columns.tolist()
        if target in nominal: nominal.remove(target)
        numeric = [c for c in df_train.columns if c not in nominal + [target,'timestamp']]
        st.info(f"Detected numeric: {numeric[:10]} ...  nominal: {nominal[:10]} ...")
        X = df_train[numeric + nominal]
        y = df_train[target]
        # simple preprocessing: numeric scaled, cat OHE
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        preprocessor = ColumnTransformer([('num', RobustScaler(), numeric), ('cat', ohe, nominal)])
        X_en = preprocessor.fit_transform(X)
        # fit
        rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_en, y)
        st.session_state['preprocessor'] = preprocessor
        st.session_state['model'] = rf
        st.success("Trained RandomForest and stored in session.")
        # quick eval: train/test split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        Xte_en = preprocessor.transform(Xte)
        ypred = rf.predict(Xte_en)
        metrics = {
            'r2': r2_score(yte, ypred),
            'rmse': mean_squared_error(yte, ypred, squared=False),
            'mae': mean_absolute_error(yte, ypred)
        }
        st.write("Eval on holdout:", metrics)

# -------------------------
# Predict UI (using session model or uploaded model)
# -------------------------
st.header("Predict — use model stored in session (or upload a pipeline)")
model = st.session_state.get('model', model_for_predict)
preproc = st.session_state.get('preprocessor', None)

if model is None:
    st.info("No model available in session. Train locally (above) or upload a pipeline that includes preprocessing.")
else:
    st.write("Model loaded. Provide new input values.")
    # attempt to infer numeric/categorical original columns
    # if preproc available, get numeric/cat from it
    if preproc is not None:
        try:
            orig_num = preproc.transformers_[0][2]
            orig_cat = preproc.transformers_[1][2]
        except Exception:
            orig_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'Traffic Volume']
            orig_cat = [c for c in df.columns if c not in orig_num and c != 'Traffic Volume']
    else:
        orig_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'Traffic Volume']
        orig_cat = [c for c in df.columns if c not in orig_num and c != 'Traffic Volume']
    form = st.form("predict_form")
    inputs = {}
    cols_num = form.columns(2)
    for i, col in enumerate(orig_num):
        default = float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else 0.0
        inputs[col] = cols_num[i % 2].number_input(col, value=float(default))
    for col in orig_cat:
        opts = df[col].unique().tolist() if col in df.columns else []
        inputs[col] = form.selectbox(col, options=opts) if opts else form.text_input(col, value="")
    submitted = form.form_submit_button("Predict")

    if submitted:
        X_new = pd.DataFrame([inputs])
        if preproc is not None:
            X_new_en = preproc.transform(X_new)
            yhat = model.predict(X_new_en)[0]
        else:
            # if model is a pipeline that includes preprocessing it will accept X_new
            try:
                yhat = model.predict(X_new)[0]
            except Exception as e:
                st.error(f"Prediction failed: model likely expects preprocessed input. Error: {e}")
                yhat = None
        if yhat is not None:
            st.success(f"Predicted Traffic Volume: {yhat:,.2f}")

            if HAS_SHAP and isinstance(model, (RandomForestRegressor,)):
                if st.checkbox("Show SHAP for this prediction"):
                    st.info("Computing SHAP (may take several seconds)...")
                    explainer = shap.TreeExplainer(model)
                    pre = preproc.transform(X_new) if preproc is not None else X_new
                    shap_values = explainer.shap_values(pre)
                    fig_shap = shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=preproc.get_feature_names_out())
                    st.pyplot(fig_shap)

# -------------------------
# Footer / Export model
# -------------------------
st.markdown("---")
if st.session_state.get('model', None) is not None:
    if st.button("Export model (.joblib)"):
        fname = "exported_model.joblib"
        joblib.dump(st.session_state['model'], fname)
        with open(fname, "rb") as f:
            st.download_button("Download model", f, file_name=fname)
        os.remove(fname)


