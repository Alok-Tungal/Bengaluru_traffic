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

# app.py
import streamlit as st
st.set_page_config(page_title="🚦 Bangalore Traffic — Live Dashboard", layout="wide")

# optional fancy sidebar menu (fallback handled)
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

import pandas as pd
import numpy as np
import requests, io, time, os, joblib
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# ML imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# shap (optional)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# --------------------------
# CONFIG: Put your GitHub raw file URL here (or paste it in sidebar)
# Example raw URL:
# "https://raw.githubusercontent.com/your-username/your-repo/main/bangalore_traffic.csv"
DEFAULT_GITHUB_RAW_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/bangalore_traffic.csv"
# --------------------------

# ---------- CSS ----------
st.markdown("""
    <style>
        body { background-color: #f7f8fb; font-family: 'Segoe UI', Roboto, sans-serif; }
        .block-container { padding: 1rem 1.5rem; }
        .stButton>button { background:#0f766e; color:white; border-radius:8px; padding:6px 12px; }
        .kpi { background: white; border-radius:10px; padding:12px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
        .small { font-size:0.9rem; color:#6b7280; }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Bangalore Traffic")
    st.write("Load dataset from GitHub (raw CSV). No upload required.")
    github_url = st.text_input("GitHub raw CSV URL", value=st.secrets.get("GITHUB_RAW_URL", DEFAULT_GITHUB_RAW_URL))
    st.write("⚙️ Refresh & simulation settings")
    refresh_cache = st.button("Force reload from GitHub")
    refresh_ttl = st.number_input("Cache TTL (seconds)", min_value=10, max_value=3600, value=60)
    sim_autoplay = st.checkbox("Enable playback simulation by default", value=False)
    st.markdown("---")
    st.write("Model (optional)")
    uploaded_model = st.file_uploader("Upload trained pipeline / joblib (.joblib or .pkl)", type=["joblib","pkl"])
    st.markdown("If you upload a pipeline that already includes preprocessing, the app will use it directly.")
    st.markdown("---")
    if not HAS_OPTION_MENU:
        st.caption("Install `streamlit-option-menu` for better sidebar UI: `pip install streamlit-option-menu`")

# ---------- Utilities ----------
def safe_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def load_github_csv(url):
    """Load raw CSV from GitHub URL (simple). Returns pd.DataFrame or empty df on failure."""
    if not url or "githubusercontent" not in url and "raw.githubusercontent" not in url:
        return pd.DataFrame()
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        st.error(f"Failed to load data from GitHub: {e}")
        return pd.DataFrame()

def get_feature_names_from_preprocessor(ct):
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        names = []
        for name, transformer, cols in ct.transformers_:
            if transformer == 'passthrough':
                names.extend(cols)
            else:
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        names.extend(transformer.get_feature_names_out(cols))
                    except Exception:
                        names.extend(cols)
                else:
                    names.extend(cols)
        return names

# ---------- Data Loading & Caching ----------
@st.cache_data(ttl=refresh_ttl)
def load_data_cached(url):
    return load_github_csv(url)

if refresh_cache:
    st.cache_data.clear()
    st.rerun()


df = load_data_cached(github_url)

# fallback sample if no data
if df is None or df.empty:
    st.warning("No dataset loaded from GitHub. Use sidebar to paste a raw GitHub URL. Loading sample data for demo.")
    SAMPLE = """Area Name,Road/Intersection Name,Traffic Volume,Average Speed,Travel Time Index,Congestion Level,Road Capacity Utilization,Incident Reports,Environmental Impact,Public Transport Usage,Traffic Signal Compliance,Parking Usage,Pedestrian and Cyclist Count,Weather Conditions,Roadwork and Construction Activity,Year,Month,Day,Hour,DayOfWeek,IsWeekend
Indiranagar,100 Feet Road,50590,50.230299,1.5,100,100,0,151.18,70.63233,84.0446,85.403629,111,Clear,No,2022,1,1,0,5,1
Indiranagar,CMH Road,30825,29.377125,1.5,100,100,1,111.65,41.924899,91.407038,59.983689,100,Clear,No,2022,1,1,0,5,1
"""
    df = pd.read_csv(io.StringIO(SAMPLE))

# ---------- Preprocessing helpers ----------
# Auto-detect nominal columns (object / category) — allow user to override later if needed
detected_nominal = df.select_dtypes(include=['object','category']).columns.tolist()
# remove any accidental target from nominals if present
detected_nominal = [c for c in detected_nominal if c not in ['Traffic Volume', 'Average Speed', 'Travel Time Index', 'Congestion Level']]

# Create timestamp if possible
if {"Year","Month","Day","Hour"}.issubset(df.columns):
    try:
        df['timestamp'] = pd.to_datetime(df[['Year','Month','Day','Hour']].rename(columns={'Hour':'hour'}), errors='coerce')
    except Exception:
        df['timestamp'] = pd.NaT
else:
    df['timestamp'] = pd.NaT

# ---------- Top KPIs ----------
st.title("🚦 Bangalore Traffic — Live Dashboard & Predictor")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
mean_tv = df["Traffic Volume"].mean() if "Traffic Volume" in df.columns else np.nan
c3.metric("Avg Traffic Volume", f"{mean_tv:,.0f}" if not np.isnan(mean_tv) else "—")
c4.metric("Last load", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

# ---------- Navigation ----------
if HAS_OPTION_MENU:
    with st.sidebar:
        selected = option_menu("Main Menu",
                               ["Live Dashboard", "Visualizations", "Predict Traffic", "Train Model", "About"],
                               icons=['cloud', 'bar-chart', 'cpu', 'gear', 'info-circle'],
                               default_index=0)
else:
    selected = st.sidebar.radio("Main Menu", ["Live Dashboard", "Visualizations", "Predict Traffic", "Train Model", "About"])

# ---------- Live Dashboard ----------
if selected == "Live Dashboard":
    st.header("📡 Live Dashboard (filters & playback)")
    # Filters
    st.sidebar.markdown("### Filters")
    area_options = df['Area Name'].unique().tolist() if 'Area Name' in df.columns else []
    selected_areas = st.sidebar.multiselect("Area Name", options=area_options, default=area_options[:6])
    hour_range = st.sidebar.slider("Hour range", 0, 23, (0,23))
    dow_options = sorted(df['DayOfWeek'].dropna().unique().astype(int).tolist()) if 'DayOfWeek' in df.columns else []
    selected_dows = st.sidebar.multiselect("DayOfWeek", options=dow_options, default=dow_options)

    df_filtered = df.copy()
    if selected_areas:
        df_filtered = df_filtered[df_filtered['Area Name'].isin(selected_areas)]
    df_filtered = df_filtered[(df_filtered['Hour'] >= hour_range[0]) & (df_filtered['Hour'] <= hour_range[1])]
    if selected_dows:
        df_filtered = df_filtered[df_filtered['DayOfWeek'].isin(selected_dows)]

    # Time series
    st.subheader("Traffic Volume — Time Series (daily aggregate)")
    if 'timestamp' in df_filtered.columns and not df_filtered['timestamp'].isna().all():
        df_ts = df_filtered.dropna(subset=['timestamp']).copy()
        df_ts['date'] = pd.to_datetime(df_ts['timestamp']).dt.date
        ts_agg = df_ts.groupby('date')['Traffic Volume'].sum().reset_index()
        fig = px.line(ts_agg, x='date', y='Traffic Volume', title="Daily Total Traffic Volume")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Timestamp not available (need Year/Month/Day/Hour columns)")

    # Heatmap DayOfWeek x Hour
    st.subheader("Heatmap — Avg Traffic by DayOfWeek & Hour")
    if {'DayOfWeek','Hour','Traffic Volume'}.issubset(df_filtered.columns):
        pivot = df_filtered.pivot_table(index='DayOfWeek', columns='Hour', values='Traffic Volume', aggfunc='mean', fill_value=0)
        fig_hm = px.imshow(pivot, aspect='auto', labels=dict(x="Hour", y="DayOfWeek", color="Avg Traffic"))
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Need DayOfWeek, Hour, and Traffic Volume for heatmap")

    # Top roads
    st.subheader("Top Roads by Average Traffic Volume")
    if 'Road/Intersection Name' in df_filtered.columns:
        top_n = st.slider("Top N roads", 5, 30, 10)
        top_roads = df_filtered.groupby('Road/Intersection Name')['Traffic Volume'].mean().sort_values(ascending=False).head(top_n).reset_index()
        fig_bar = px.bar(top_roads, x='Traffic Volume', y='Road/Intersection Name', orientation='h', title="Top Roads by Avg Traffic")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No Road/Intersection Name column found")

    # Scatter Speed vs Volume
    st.subheader("Average Speed vs Traffic Volume")
    if {'Average Speed','Traffic Volume'}.issubset(df_filtered.columns):
        fig_sc = px.scatter(df_filtered, x='Traffic Volume', y='Average Speed', color='Area Name' if 'Area Name' in df_filtered.columns else None,
                            hover_data=['Road/Intersection Name'] if 'Road/Intersection Name' in df_filtered.columns else None,
                            title="Traffic Volume vs Average Speed")
        st.plotly_chart(fig_sc, use_container_width=True)

    # Distribution
    st.subheader("Distribution & Boxplots")
    col1, col2 = st.columns(2)
    with col1:
        if 'Traffic Volume' in df_filtered.columns:
            fig_dist = px.histogram(df_filtered, x='Traffic Volume', nbins=60, title="Traffic Volume Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        if {'DayOfWeek','Traffic Volume'}.issubset(df_filtered.columns):
            fig_box = px.box(df_filtered, x='DayOfWeek', y='Traffic Volume', title="Traffic Volume by DayOfWeek")
            st.plotly_chart(fig_box, use_container_width=True)

    # Playback simulation
    st.subheader("Realtime Playback Simulation (demo)")
    sim_rows = st.slider("Playback: number of recent rows", 10, min(500, len(df)), 100)
    interval = st.slider("Playback interval (sec)", 1, 5, 2)
    start_play = st.button("▶️ Start Playback")
    stop_play = st.button("⏹ Stop Playback")
    if 'play' not in st.session_state:
        st.session_state.play = False
    if start_play:
        st.session_state.play = True
    if stop_play:
        st.session_state.play = False

    if st.session_state.play:
        tail = df_filtered.tail(sim_rows).reset_index(drop=True)
        placeholder = st.empty()
        for i in range(len(tail)):
            if not st.session_state.play:
                break
            cur = tail.iloc[:i+1]
            with placeholder.container():
                left, right = st.columns(2)
                left.metric("Cumulative Traffic (slice)", f"{int(cur['Traffic Volume'].sum()):,}")
                right.metric("Avg Speed (slice)", f"{cur['Average Speed'].mean():.2f}" if 'Average Speed' in cur.columns else "—")
                fig_play = px.line(cur, x=cur.index, y='Traffic Volume', title="Live playback")
                st.plotly_chart(fig_play, use_container_width=True)
            time.sleep(interval)
        st.session_state.play = False
        placeholder.empty()

# ---------- Visualizations ----------
elif selected == "Visualizations":
    st.header("📈 Visualizations (interactive)")
    # correlation matrix
    numcols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numcols) >= 3:
        st.subheader("Correlation matrix (numeric features)")
        corr = df[numcols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Correlation matrix")
        st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.subheader("Feature Distributions")
    col = st.selectbox("Choose numeric column", numcols, index= numcols.index("Traffic Volume") if "Traffic Volume" in numcols else 0)
    fig = px.histogram(df, x=col, nbins=50, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

    # Boxplot by area
    if {'Area Name','Traffic Volume'}.issubset(df.columns):
        st.subheader("Traffic Volume by Area (boxplot)")
        fig_box = px.box(df, x='Area Name', y='Traffic Volume', title="Traffic Volume by Area")
        st.plotly_chart(fig_box, use_container_width=True)

# ---------- Predict Traffic ----------
elif selected == "Predict Traffic":
    st.header("🔮 Predict Traffic Volume (single-sample)")

    # Load or train model decision
    model = st.session_state.get('model', None)
    preprocessor = st.session_state.get('preprocessor', None)

    if uploaded_model is not None:
        try:
            # load uploaded model file (it may be a pipeline or model)
            bytes_read = uploaded_model.read()
            tmp_name = "/tmp/uploaded_model.joblib"
            with open(tmp_name, "wb") as fh:
                fh.write(bytes_read)
            model = joblib.load(tmp_name)
            st.success("Uploaded model loaded into session and will be used for predictions.")
            st.session_state['model'] = model
        except Exception as e:
            st.error(f"Failed loading uploaded model: {e}")

    if model is None:
        st.info("No trained model in session. You can Train Model (menu) or upload a pipeline including preprocessing.")
    else:
        st.success(f"Model available: {type(model).__name__}")

    # Build dynamic form from original columns (exclude Traffic Volume target)
    feature_cols = [c for c in df.columns if c != "Traffic Volume" and c != "timestamp"]
    st.markdown("### Input values (use presets to autofill)")
    preset = st.selectbox("Presets", ["-- Select --", "Normal", "Rush Hour", "Accident", "Weekend"])
    presets_map = {
        "Normal": {"Average Speed": 40, "Travel Time Index": 1.2, "Congestion Level": 50, "Road Capacity Utilization": 60, "Incident Reports": 0},
        "Rush Hour": {"Average Speed": 18, "Travel Time Index": 2.0, "Congestion Level": 90, "Road Capacity Utilization": 95, "Incident Reports": 0},
        "Accident": {"Average Speed": 8, "Travel Time Index": 3.5, "Congestion Level": 100, "Road Capacity Utilization": 100, "Incident Reports": 2},
        "Weekend": {"Average Speed": 45, "Travel Time Index": 1.0, "Congestion Level": 40, "Road Capacity Utilization": 50, "Incident Reports": 0}
    }

    # Build the form
    with st.form("predict_form"):
        inputs = {}
        # numeric first
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        # apply preset defaults
        preset_defaults = presets_map.get(preset, {})

        for col in numeric_cols:
            default = float(df[col].median()) if col in df.columns else 0.0
            default = float(preset_defaults.get(col, default))
            inputs[col] = st.number_input(col, value=default, format="%.3f")

        for col in categorical_cols:
            opts = df[col].dropna().unique().tolist() if col in df.columns else []
            if opts:
                inputs[col] = st.selectbox(col, options=opts)
            else:
                inputs[col] = st.text_input(col, value="")

        submitted = st.form_submit_button("Predict Traffic Volume")

    if submitted:
        X_new = pd.DataFrame([inputs])

        # If model is a pipeline that includes preprocessing, we can pass raw X_new
        try:
            if hasattr(model, "predict") and preprocessor is None:
                # try predict directly; many uploaded pipelines include preprocessing
                yhat = model.predict(X_new)[0]
            else:
                if preprocessor is None:
                    st.error("No preprocessor available in session; the model likely expects preprocessed input or pipeline. Train or upload a pipeline including preprocessing.")
                    yhat = None
                else:
                    X_new_en = preprocessor.transform(X_new)
                    yhat = model.predict(X_new_en)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            yhat = None

        if yhat is not None:
            st.success(f"Predicted Traffic Volume: {yhat:,.0f}")

            # quick advice belt
            if yhat > df['Traffic Volume'].quantile(0.95):
                st.warning("Predicted traffic volume is very high — consider rerouting or traffic control measures.")
            elif yhat > df['Traffic Volume'].quantile(0.75):
                st.info("High traffic expected — alert transport authorities / increase public transport frequency.")
            else:
                st.success("Traffic predicted within normal range.")

            # Feature importance if model has it
            if hasattr(model, "feature_importances_"):
                st.subheader("Feature importance (model)")
                # get feature names if preprocessor present
                if preprocessor is not None:
                    try:
                        fnames = get_feature_names_from_preprocessor(preprocessor)
                    except Exception:
                        fnames = [f"f{i}" for i in range(len(model.feature_importances_))]
                else:
                    fnames = numeric_cols + categorical_cols
                fi = pd.Series(model.feature_importances_, index=fnames).sort_values(ascending=False).head(20)
                fig = px.bar(fi.reset_index().rename(columns={'index':'feature', 0:'importance'}), x='importance', y='feature', orientation='h', title="Top Features")
                st.plotly_chart(fig, use_container_width=True)

            # SHAP explanation (if shap installed and tree model)
            if HAS_SHAP and isinstance(model, (RandomForestRegressor,)):
                if st.checkbox("Show SHAP explanation for this prediction"):
                    with st.spinner("Computing SHAP values..."):
                        try:
                            explainer = shap.TreeExplainer(model)
                            if preprocessor is not None:
                                X_for_shap = preprocessor.transform(X_new)
                                shap_values = explainer.shap_values(X_for_shap)
                                # shap bar / waterfall for single sample
                                shap.initjs()
                                st.subheader("SHAP (bar)")
                                shap_df = pd.DataFrame({'feature': get_feature_names_from_preprocessor(preprocessor), 'shap': shap_values[0]})
                                shap_df = shap_df.sort_values(by='shap', key=lambda x: np.abs(x), ascending=False).head(15)
                                fig = px.bar(shap_df, x='shap', y='feature', orientation='h', title="SHAP (top influences)")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"SHAP failed: {e}")

# ---------- Train Model ----------
elif selected == "Train Model":
    st.header("⚙️ Train a quick RandomForest (preprocessing included)")
    st.info("This trains a RandomForest on the current loaded dataset (useful for quick experimentation).")

    target = st.selectbox("Choose target column", options=[c for c in df.columns if c != 'timestamp'], index=df.columns.get_loc("Traffic Volume") if "Traffic Volume" in df.columns else 0)
    # detect nominals automatically
    detected_nom = df.select_dtypes(include=['object','category']).columns.tolist()
    detected_nom = [c for c in detected_nom if c != target]
    chosen_nom = st.multiselect("Categorical columns (one-hot encoded)", options=detected_nom, default=detected_nom)
    numeric_cols = [c for c in df.columns if c not in chosen_nom + [target, 'timestamp']]
    st.write("Numeric columns:", numeric_cols)

    test_size = st.slider("Test size (%)", 10, 40, 20)
    run_train = st.button("Train RandomForest")

    if run_train:
        X = df[[c for c in df.columns if c != target and c != 'timestamp']]
        y = df[target].copy()
        ohe = safe_ohe()
        preprocessor = ColumnTransformer(transformers=[('num', RobustScaler(), numeric_cols),
                                                      ('cat', ohe, chosen_nom)], remainder='drop')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
        X_train_en = preprocessor.fit_transform(X_train)
        X_test_en = preprocessor.transform(X_test)

        rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_train_en, y_train)

        # store in session
        st.session_state.preprocessor = preprocessor
        st.session_state.model = rf
        st.session_state.feature_names = get_feature_names_from_preprocessor(preprocessor)

        # evaluate
        y_pred = rf.predict(X_test_en)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        st.success("Training finished and model stored in session.")
        st.write("Metrics:", metrics)

        # download model
        fname = "rf_bangalore_model.joblib"
        joblib.dump({'preprocessor': preprocessor, 'model': rf}, fname)
        with open(fname, "rb") as f:
            st.download_button("Download trained pipeline (.joblib)", f, file_name=fname)
        os.remove(fname)

# ---------- About ----------
elif selected == "About":
    st.header("ℹ️ About this app")
    st.markdown("""
    **Project:** Bangalore Traffic — Dataset-driven ML & Visualizations  
    **Creator:** Alok Tungal (you can update author info)  
    **Features built:**  
    - GitHub raw CSV loading (no upload)  
    - Interactive visualizations (Plotly + Seaborn)  
    - Real-time playback simulation  
    - On-the-fly training and prediction with preprocessing  
    - SHAP explainability (optional)
    """)
    st.markdown("## How to use")
    st.markdown("""
    1. Put your `bangalore_traffic.csv` in a GitHub repo and paste the **raw** URL in sidebar.  
    2. Use **Live Dashboard** and **Visualizations** to explore data.  
    3. Train a quick model in **Train Model** or upload a pre-trained pipeline with preprocessing.  
    4. Go to **Predict Traffic** to input a scenario and get a prediction & explanation.
    """)

# End of app
