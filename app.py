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
st.set_page_config(page_title="🚦 Bangalore Traffic — Advanced", layout="wide")

# ---- safe optional imports ----
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

import pandas as pd
import numpy as np
import joblib
import os
import io
import time

# ML imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set_style("whitegrid")

# SHAP (optional)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ----------------------
# Custom CSS (nice UI)
# ----------------------
st.markdown(
    """
    <style>
      body { background-color: #f7f8fb; font-family: 'Segoe UI', Roboto, sans-serif; }
      .block-container { padding: 1.25rem 2rem; }
      .stButton>button { background-color:#0f766e; color:white; border-radius:8px; }
      .metric-label { color: #6b7280; font-size:0.9rem; }
      .kpi { background: white; border-radius:10px; padding:12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Utilities
# ----------------------
def safe_ohe(sparse_default=False):
    """Return OneHotEncoder instance using correct keyword for sklearn version."""
    # sklearn >=1.2 uses sparse_output, older versions use sparse
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=not sparse_default)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=not sparse_default)

def get_feature_names_from_column_transformer(ct, input_features):
    """
    Attempt to return feature names after ColumnTransformer transformation.
    Works for modern sklearn; falls back to combining numeric and OHE names.
    """
    try:
        # sklearn >=1.0
        names = ct.get_feature_names_out()
        return list(names)
    except Exception:
        # fallback: build manually
        feature_names = []
        for name, transformer, cols in ct.transformers_:
            if name == 'remainder':
                continue
            if transformer == 'passthrough':
                feature_names.extend(list(cols))
                continue
            # transformer might be OneHotEncoder or scaler
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    out = transformer.get_feature_names_out(cols)
                    feature_names.extend(list(out))
                except Exception:
                    feature_names.extend(list(cols))
            else:
                feature_names.extend(list(cols))
        return feature_names

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

# ----------------------
# Sidebar navigation
# ----------------------
with st.sidebar:
    st.image("https://i.imgur.com/0Z6fVQ5.png", width=180) if False else None
    if HAS_OPTION_MENU:
        selection = option_menu(
            "Menu",
            ["Home", "Upload Data", "EDA", "Train Model", "Evaluate", "Predict & Explain"],
            icons=["house", "cloud-upload", "bar-chart", "robot", "check-square", "lightbulb"],
            menu_icon="cast",
            default_index=0
        )
    else:
        selection = st.radio("Menu", ["Home", "Upload Data", "EDA", "Train Model", "Evaluate", "Predict & Explain"])

    st.markdown("---")
    st.write("Model file:")
    uploaded_model = st.file_uploader("Upload `.joblib` model (optional)", type=["joblib", "pkl"])
    st.write("Tip: you can upload a pre-trained `best_rf_model.joblib` here.")
    st.markdown("---")
    st.write("About")
    st.info("Bangalore Traffic — advanced Streamlit app. Built for your project.")

# ----------------------
# Data loader (cached)
# ----------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    return pd.read_csv(file)

# allow sample dataset if none provided
SAMPLE_CSV = """Area Name,Road/Intersection Name,Traffic Volume,Average Speed,Travel Time Index,Congestion Level,Road Capacity Utilization,Incident Reports,Environmental Impact,Public Transport Usage,Traffic Signal Compliance,Parking Usage,Pedestrian and Cyclist Count,Weather Conditions,Roadwork and Construction Activity,Year,Month,Day,Hour,DayOfWeek,IsWeekend
Indiranagar,100 Feet Road,50590,50.230299,1.5,100,100,0,151.18,70.63233,84.0446,85.403629,111,Clear,No,2022,1,1,0,5,1
Indiranagar,CMH Road,30825,29.377125,1.5,100,100,1,111.65,41.924899,91.407038,59.983689,100,Clear,No,2022,1,1,0,5,1
"""

# ----------------------
# Global session-state helpers
# ----------------------
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'X_train_en' not in st.session_state:
    st.session_state.X_train_en = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None

# ----------------------
# Page: HOME
# ----------------------
if selection == "Home":
    st.title("🚦 Bangalore Traffic — Advanced Dashboard")
    st.markdown(
        """
        This app helps you:
        - Explore Bangalore traffic dataset
        - Train and tune ML models (RandomForest/XGBoost if available)
        - Evaluate & explain predictions (feature importance, SHAP)
        - Export prediction model for deployment
        """
    )
    st.markdown("### Quick actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Load sample data"):
            st.session_state.raw_df = pd.read_csv(io.StringIO(SAMPLE_CSV))
            st.success("Sample data loaded into session.")
    with c2:
        if st.button("Clear session"):
            for k in ['raw_df','preprocessor','model','feature_names','X_train_en','y_train']:
                st.session_state[k] = None
            st.success("Session cleared.")
    with c3:
        st.write("Model loaded?" )
        st.write("Yes" if st.session_state.model is not None else "No")

# ----------------------
# Page: Upload Data
# ----------------------
if selection == "Upload Data":
    st.header("Upload your Bangalore traffic CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        st.session_state.raw_df = df
        st.success("File uploaded and loaded into session.")
        st.dataframe(df.head(10))
    else:
        st.info("No file uploaded. You can load sample data using Home > Load sample data.")
    st.markdown("### Current session dataset preview")
    if st.session_state.raw_df is not None:
        st.dataframe(st.session_state.raw_df.head())

# ----------------------
# Page: EDA
# ----------------------
if selection == "EDA":
    st.header("Exploratory Data Analysis")
    df = st.session_state.raw_df
    if df is None or df.empty:
        st.warning("No dataset loaded. Upload or load sample data first.")
    else:
        st.subheader("Basic info")
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write(df.describe(include='all').T)

        st.subheader("Interactive filters")
        cols = df.columns.tolist()
        filter_col = st.selectbox("Filter by column (optional)", options=["None"] + cols)
        if filter_col != "None":
            unique_vals = df[filter_col].unique().tolist()[:100]
            sel = st.multiselect(f"Select values for {filter_col}", options=unique_vals, default=unique_vals[:5])
            df_filtered = df[df[filter_col].isin(sel)]
        else:
            df_filtered = df

        st.subheader("Plots")
        # Traffic Volume distribution
        if "Traffic Volume" in df_filtered.columns:
            fig = px.histogram(df_filtered, x="Traffic Volume", nbins=40, title="Traffic Volume distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Boxplot by Hour
        if {"Hour","Traffic Volume"}.issubset(df_filtered.columns):
            fig2, ax = plt.subplots(figsize=(10,4))
            sns.boxplot(x="Hour", y="Traffic Volume", data=df_filtered, ax=ax)
            st.pyplot(fig2)

        # Correlation heatmap (numeric)
        st.subheader("Correlation (numeric columns)")
        numcols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if len(numcols) >= 2:
            corr = df_filtered[numcols].corr()
            fig3, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig3)
        else:
            st.info("Not enough numeric columns for correlation.")

# ----------------------
# Page: Train Model
# ----------------------
if selection == "Train Model":
    st.header("Train / Tune Model")
    df = st.session_state.raw_df
    if df is None or df.empty:
        st.warning("Please upload or load sample data first.")
    else:
        st.info("Select target and feature columns. Defaults chosen for your dataset.")
        all_cols = df.columns.tolist()

        # Choose target
        default_target = "Traffic Volume" if "Traffic Volume" in all_cols else all_cols[-1]
        target_col = st.selectbox("Select target column (what to predict)", options=all_cols, index=all_cols.index(default_target))

        # Detect categorical candidates (string/object)
        detected_nominal = df.select_dtypes(include=['object','category']).columns.tolist()
        st.write("Detected categorical columns:", detected_nominal)
        nominal_cols = st.multiselect("Confirm categorical columns (one-hot encoded)", options=detected_nominal, default=detected_nominal)

        # Numeric columns
        numeric_cols = [c for c in all_cols if c not in nominal_cols + [target_col]]
        st.write("Numeric columns (will be scaled):", numeric_cols)

        # Train/test split
        test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20)
        random_state = st.number_input("Random state", value=42, step=1)
        X = df[[c for c in all_cols if c != target_col]]
        y = df[target_col]
        btn_train = st.button("🔧 Preprocess & Train Default RandomForest")

        if btn_train:
            with st.spinner("Preprocessing and training... this may take some seconds"):
                # Preprocessor
                ohe = safe_ohe()
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', RobustScaler(), numeric_cols),
                        ('cat', ohe, nominal_cols)
                    ],
                    remainder='drop'
                )
                # Prepare data arrays
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))
                X_train_en = preprocessor.fit_transform(X_train)
                X_test_en = preprocessor.transform(X_test)

                # Get feature names
                try:
                    feature_names = get_feature_names_from_column_transformer(preprocessor, X_train.columns)
                except Exception:
                    feature_names = None

                # Train RF
                rf = RandomForestRegressor(n_estimators=100, random_state=int(random_state), n_jobs=-1)
                rf.fit(X_train_en, y_train)

                # store in session
                st.session_state.preprocessor = preprocessor
                st.session_state.model = rf
                st.session_state.feature_names = feature_names
                st.session_state.X_train_en = X_train_en
                st.session_state.y_train = y_train
                st.success("Model trained and stored in session.")

                # Basic eval
                y_pred = rf.predict(X_test_en)
                metrics = evaluate_regression(y_test, y_pred)
                st.metric("Test R²", f"{metrics['r2']:.4f}")
                st.metric("Test RMSE", f"{metrics['rmse']:.2f}")
                st.metric("Test MAE", f"{metrics['mae']:.2f}")

        # Optional: Hyperparameter tuning (RandomizedSearchCV)
        st.markdown("#### Optional: Randomized Hyperparameter Tuning")
        tune = st.checkbox("Run randomized tuning (slower, recommended with small n_iter)", value=False)
        if tune:
            n_iter = st.slider("n_iter (random combos)", 5, 60, 20)
            cv = st.slider("CV folds", 2, 5, 3)
            if st.button("Run RandomizedSearchCV"):
                if st.session_state.preprocessor is None:
                    st.error("Please run default training step first to setup preprocessor.")
                else:
                    with st.spinner("Tuning hyperparameters..."):
                        # Use already transformed matrix to speed up (avoid re-transform)
                        Xtr = st.session_state.X_train_en
                        ytr = st.session_state.y_train

                        param_grid = {
                            'n_estimators': [100, 150, 200],
                            'max_depth': [None, 10, 20],
                            'max_features': ['sqrt', 'log2', None],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'bootstrap': [True, False]
                        }
                        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
                        rnd = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=n_iter, cv=cv, n_jobs=-1, scoring='r2', verbose=1, random_state=42)
                        rnd.fit(Xtr, ytr)
                        best = rnd.best_estimator_
                        st.session_state.model = best
                        st.success("RandomizedSearch finished. Best model stored in session.")
                        st.write("Best params:", rnd.best_params_)

# ----------------------
# Page: Evaluate
# ----------------------
if selection == "Evaluate":
    st.header("Model Evaluation & Diagnostics")
    df = st.session_state.raw_df
    model = st.session_state.model
    preprocessor = st.session_state.preprocessor

    if model is None or preprocessor is None:
        st.warning("Train a model first in 'Train Model' or upload a model in the sidebar.")
    else:
        st.write("Model type:", type(model).__name__)
        # If dataset loaded, do test evaluation
        if df is None or df.empty:
            st.info("No dataset loaded to evaluate against. Upload dataset in Upload Data.")
        else:
            # ask for target
            default_target = "Traffic Volume" if "Traffic Volume" in df.columns else df.columns[-1]
            target_col = st.selectbox("Select target column for evaluation", options=df.columns.tolist(), index=df.columns.tolist().index(default_target))
            # Prepare X/y and split
            X = df[[c for c in df.columns if c != target_col]]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test_en = preprocessor.transform(X_test)
            y_pred = model.predict(X_test_en)
            metrics = evaluate_regression(y_test, y_pred)
            st.subheader("Metrics on holdout set")
            st.metric("R²", f"{metrics['r2']:.4f}")
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
            st.metric("MAE", f"{metrics['mae']:.2f}")

            # Residual plot
            fig, ax = plt.subplots(figsize=(8,4))
            residuals = y_test - y_pred
            ax.scatter(y_pred, residuals, alpha=0.6)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            st.pyplot(fig)

            # Pred vs actual
            fig2 = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, title="Actual vs Predicted")
            st.plotly_chart(fig2, use_container_width=True)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Importance")
                # get names
                fnames = st.session_state.feature_names
                if fnames is None:
                    # attempt to build names
                    try:
                        fnames = get_feature_names_from_column_transformer(preprocessor, X.columns)
                    except Exception:
                        fnames = [f"f{i}" for i in range(len(model.feature_importances_))]
                fi = pd.Series(model.feature_importances_, index=fnames).sort_values(ascending=False)
                st.dataframe(fi.head(30))
                fig3 = px.bar(fi.head(20).reset_index().rename(columns={'index':'feature', 0:'importance'}), x='importance', y='feature', orientation='h')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Model has no feature_importances_ attribute.")

            # Optional SHAP (if available)
            if HAS_SHAP:
                if st.checkbox("Show SHAP summary (may be slow)", value=False):
                    with st.spinner("Computing SHAP values..."):
                        explainer = shap.TreeExplainer(model)
                        X_sample = X_test_en if hasattr(X_test_en, "shape") else preprocessor.transform(X_test)
                        shap_vals = explainer.shap_values(X_sample if isinstance(X_sample, np.ndarray) else X_test_en)
                        st.subheader("SHAP summary")
                        fig_shap = shap.summary_plot(shap_vals, X_sample if isinstance(X_sample, (np.ndarray, pd.DataFrame)) else X_test_en, show=False)
                        st.pyplot(bbox_inches='tight')

# ----------------------
# Page: Predict & Explain
# ----------------------
if selection == "Predict & Explain":
    st.header("Predict on new input / Explain predictions")
    model = st.session_state.model
    preprocessor = st.session_state.preprocessor
    df = st.session_state.raw_df

    if model is None or preprocessor is None:
        st.warning("Train a model first or upload a pre-trained model in the sidebar.")
    else:
        # Build input widgets dynamically using feature names from preprocessor
        # If we don't have mapping, ask user to provide sample row
        fnames = st.session_state.feature_names
        if fnames is None:
            try:
                fnames = get_feature_names_from_column_transformer(preprocessor, df[[c for c in df.columns if c not in [st.session_state.get('target_col','Traffic Volume')]]].columns)
            except Exception:
                fnames = None

        st.subheader("Provide values for prediction (you can copy-paste a CSV row too)")

        if df is not None and not df.empty:
            # Use original column groups for building form
            all_cols = df.columns.tolist()
            target_default = "Traffic Volume" if "Traffic Volume" in all_cols else all_cols[-1]
            target_col = st.selectbox("Target column (just for reference)", options=all_cols, index=all_cols.index(target_default))
        else:
            target_col = "Traffic Volume"

        # We'll ask for the original features (before encoding): numeric_cols and nominal_cols
        # If session preprocessor exists -> try to extract original cols from preprocessor
        try:
            # Attempt to recover original transformer columns
            orig_num = preprocessor.transformers_[0][2]
            orig_cat = preprocessor.transformers_[1][2]
        except Exception:
            # fallback ask user
            st.write("Could not infer original feature groups; please input using the simple fields below.")
            orig_num = []
            orig_cat = []

        st.write("Numeric features (provide values):")
        input_vals = {}
        for col in orig_num:
            col_min = int(df[col].min()) if col in df.columns and np.issubdtype(df[col].dtype, np.number) else 0
            col_max = int(df[col].max()) if col in df.columns and np.issubdtype(df[col].dtype, np.number) else 100000
            default = int(df[col].median()) if col in df.columns and np.issubdtype(df[col].dtype, np.number) else 0
            input_vals[col] = st.number_input(col, value=float(default), format="%.3f")

        st.write("Categorical features:")
        for col in orig_cat:
            opts = df[col].unique().tolist() if col in df.columns else []
            if len(opts) > 0:
                input_vals[col] = st.selectbox(col, options=opts)
            else:
                input_vals[col] = st.text_input(col, value="")

        if st.button("Predict"):
            # Build DataFrame
            inp_df = pd.DataFrame([input_vals])
            X_en = preprocessor.transform(inp_df)
            pred = model.predict(X_en)[0]
            st.success(f"Prediction: {pred:.3f}")

            if HAS_SHAP:
                if st.checkbox("Explain with SHAP (single prediction)"):
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_en)
                    st.subheader("SHAP values (bar)")
                    shap.force_plot(explainer.expected_value, shap_vals, inp_df, matplotlib=True, show=False)
                    st.pyplot()

        # Allow download of model
        st.markdown("### Export model")
        if st.button("Download model (.joblib)"):
            # save to server temp and provide link
            fname = "exported_model.joblib"
            joblib.dump(model, fname)
            with open(fname, "rb") as f:
                btn = st.download_button(label="Download model", data=f, file_name=fname)
            os.remove(fname)

# ----------------------
# If user uploaded a model in sidebar, load it into session
# ----------------------
if uploaded_model is not None:
    try:
        model_bytes = uploaded_model.read()
        tmp_path = "/tmp/uploaded_model.joblib"
        with open(tmp_path, "wb") as fh:
            fh.write(model_bytes)
        st.session_state.model = joblib.load(tmp_path)
        st.success("Uploaded model loaded into session.")
    except Exception as e:
        st.error(f"Failed to load uploaded model: {e}")
