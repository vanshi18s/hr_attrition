import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Attrition Predictor", layout="wide")
plt.style.use('default')
sns.set_palette("husl")

st.title("🔥 Employee Attrition Predictor - Full Pipeline (Flexible)")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FILE UPLOAD & AUTO-CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
st.header("📁 1. Upload & Clean Data")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file is not None:
    @st.cache_data
    def load_and_clean(file):
        df = pd.read_csv(file)
        
        # 1. Handle Target if exists
        if "Attrition" in df.columns:
            # Check if Attrition is Yes/No or already 1/0
            if df["Attrition"].dtype == 'object':
                 df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
            df["Attrition"] = df["Attrition"].fillna(0).astype(int)
        
        # 2. Drop standard useless columns if they exist
        cols_to_drop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
        existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
        if existing_drop_cols:
            df.drop(columns=existing_drop_cols, inplace=True)
        
        # 3. Dynamic Label Encoding for ANY object column
        # We process all object columns EXCEPT Attrition (target)
        obj_cols = df.select_dtypes(include=["object"]).columns
        le = LabelEncoder()
        
        for col in obj_cols:
            if col != "Attrition":
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    df = load_and_clean(uploaded_file)
    st.success(f"✅ Cleaned {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    
    st.dataframe(df.head())
else:
    st.info("👆 Upload CSV to start")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model Training", "📈 Results & Test"])
with tab1:
    st.header("🔍 Exploratory Data Analysis")
    
    # Target distribution - SIMPLE PIE
    if "Attrition" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=df["Attrition"].value_counts(), 
                             names=["Stay (0)", "Leave (1)"], 
                             title="Attrition Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            count_0 = (df["Attrition"] == 0).sum()
            count_1 = (df["Attrition"] == 1).sum()
            if count_1 > 0:
                imbalance = count_0 / count_1
                st.metric("Imbalance Ratio", f"{imbalance:.1f}:1")
            else:
                st.metric("Imbalance Ratio", "N/A (No positive samples)")
    
    # SIMPLE CORRELATION TABLE
    num_cols = df.select_dtypes(include=['number']).columns.drop("Attrition", errors='ignore')
    if len(num_cols) > 0 and "Attrition" in df.columns:
        st.subheader("🔗 Correlation with Attrition")
        corrs = df[num_cols].corrwith(df["Attrition"]).sort_values(ascending=False)
        st.dataframe(corrs.to_frame("Correlation"), use_container_width=True)
    
    # Color map
    attrition_color_map = {1: "#FF4B4B", 0: "#1C83E1"} # Mapped to int because we encoded it

    if "Attrition" in df.columns:
        st.subheader("📈 Top Features Distribution")
        
        # Select features (Dynamic check)
        potential_targets = ["Age", "MonthlyIncome", "DistanceFromHome", "DailyRate"]
        target_cols = [c for c in potential_targets if c in df.columns]
        
        if target_cols:
            col1, col2 = st.columns(2)
            for i, feat in enumerate(target_cols):
                with col1 if i % 2 == 0 else col2:
                    fig = px.histogram(
                        df, x=feat, color="Attrition",
                        title=f"{feat} Distribution", barmode='overlay', opacity=0.7,
                        color_discrete_map=attrition_color_map
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        if "MonthlyIncome" in df.columns:
            st.subheader("💰 Income Analysis")
            fig_box = px.box(
                df, x="Attrition", y="MonthlyIncome", color="Attrition",
                points="all", title="Monthly Income vs. Attrition Status",
                color_discrete_map=attrition_color_map
            )
            st.plotly_chart(fig_box, use_container_width=True)

        if "Department" in df.columns:
            st.subheader("🏢 Attrition by Department")
            # Note: Department is label encoded now, so it will show as numbers (0, 1, 2)
            dept_counts = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
            fig_bar = px.bar(
                dept_counts, x="Department", y="Count", color="Attrition", 
                barmode='group', title="Attrition Counts by Department (Encoded)",
                text="Count", color_discrete_map=attrition_color_map
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. YOUR EXACT ML PIPELINE (DYNAMIC)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🚀 Train Imbalanced XGBoost Model")
    
    if "Attrition" not in df.columns:
        st.error("Attrition column missing!")
        st.stop()
    
    if st.button("🎯 RUN FULL PIPELINE", type="primary"):
        with st.spinner("🔄 Running your exact pipeline..."):
            
            y = df["Attrition"]
            X = df.drop(columns=["Attrition"])
            
            # Dynamic Feature Identification
            # Since we already label encoded objects in step 1, 
            # everything in X is technically numeric now. 
            # However, for the Pipeline to make sense (standard scaling vs one hot), 
            # we usually distinguish continuous vs categorical. 
            # BUT, since we already LabelEncoded, passing everything as numeric 
            # to XGBoost is often fine. 
            # To keep your OneHot structure for categorical features effectively:
            
            # We will treat columns with < 20 unique values as categorical for OneHot,
            # and others as continuous for Scaling.
            
            cat_candidates = [c for c in X.columns if X[c].nunique() < 20]
            num_candidates = [c for c in X.columns if c not in cat_candidates]
            
            numeric_features = num_candidates
            categorical_features = cat_candidates
            
            # Save feature names for later prediction
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.num_feats_model = numeric_features
            st.session_state.cat_feats_model = categorical_features

            pos_ratio = (y == 1).sum() / (y == 0).sum()
            base_scale_pos = max(1.0, round(1 / pos_ratio)) if pos_ratio > 0 else 1.0
            
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
            )
            
            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
            
            # Note: Input to OneHot here are integers (from label encoding), which is fine
            categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ],
                remainder="drop" 
            )
            
            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist", # Compatible with most systems
                random_state=42,
                n_jobs=-1
            )
            
            pipe = ImbPipeline(steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", xgb)
            ])
            
            param_distributions = {
                "clf__n_estimators": [300, 500],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
                "clf__scale_pos_weight": [base_scale_pos, int(base_scale_pos * 2)],
                "smote__k_neighbors": [3, 5]
            }
            
            def f1_minority_score(y_true, y_pred):
                return precision_recall_fscore_support(
                    y_true, y_pred, average="binary", pos_label=1
                )[2]
            
            f1_minority_scorer = make_scorer(f1_minority_score)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            search = RandomizedSearchCV(
                pipe, param_distributions, n_iter=10, cv=cv, # Reduced iter for speed/reliability in demo
                scoring=f1_minority_scorer, verbose=1, n_jobs=-1, random_state=42
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # Threshold tuning
            y_val_proba = best_model.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.05, 0.95, 37)
            best_thr = 0.5
            best_f1 = 0
            
            for thr in thresholds:
                y_val_pred_thr = (y_val_proba >= thr).astype(int)
                f1 = f1_minority_score(y_val, y_val_pred_thr)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            
            # Test predictions
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            y_test_pred_thr = (y_test_proba >= best_thr).astype(int)
            
            st.session_state.results = {
                'model': best_model,
                'best_params': search.best_params_,
                'best_thr': best_thr,
                'y_test': y_test,
                'y_test_proba': y_test_proba,
                'y_test_pred': y_test_pred_thr,
                'cm': confusion_matrix(y_test, y_test_pred_thr),
                'report': classification_report(y_test, y_test_pred_thr, output_dict=True),
                'roc_auc': roc_auc_score(y_test, y_test_proba),
                'base_scale_pos': base_scale_pos
            }
            
            st.success("✅ Pipeline complete!")
            st.write("**Best Params:**", search.best_params_)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RESULTS & SINGLE TEST PREDICTION (DYNAMIC)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🎯 Results & Test Prediction")
    
    if 'results' not in st.session_state:
        st.warning("👆 Run pipeline first!")
        st.stop()
    
    results = st.session_state.results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{(results['y_test_pred'] == results['y_test']).mean():.3f}")
    col2.metric("Best Threshold", f"{results['best_thr']:.2f}")
    col3.metric("AUC Score", f"{results['roc_auc']:.3f}")
    
    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    st.subheader("📋 Detailed Metrics")
    st.dataframe(pd.DataFrame(results['report']).T)
    
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_test_proba'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={results["roc_auc"]:.3f})', 
                            line=dict(color='darkorange')))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(color='grey', dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
    st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # DYNAMIC TEST PREDICTION SECTION
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("🧪 Test One Employee (Dynamic Inputs)")

    model = results['model']
    
    # Retrieve the features that were actually used in training
    train_num = st.session_state.get('num_feats_model', [])
    train_cat = st.session_state.get('cat_feats_model', [])
    
    user_input = {}
    
    # Dynamic Input Fields
    if train_num:
        st.markdown("### Numeric Features")
        cols = st.columns(3)
        for i, feat in enumerate(train_num):
            with cols[i % 3]:
                # Safe checking for min/max to avoid errors if column is constant or weird
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                mid_val = float(df[feat].median())
                val = st.number_input(feat, min_val, max_val, mid_val)
                user_input[feat] = val

    if train_cat:
        st.markdown("### Categorical Features (Encoded)")
        # Note: These are now integers because we Label Encoded them at start.
        cols = st.columns(3)
        for i, feat in enumerate(train_cat):
            with cols[i % 3]:
                options = sorted(df[feat].unique())
                # Default to mode
                default_idx = 0
                mode_val = df[feat].mode()
                if not mode_val.empty:
                    if mode_val[0] in options:
                        default_idx = list(options).index(mode_val[0])
                        
                val = st.selectbox(f"{feat}", options, index=default_idx)
                user_input[feat] = val

    if st.button("🔮 Predict Risk"):
        # Create DF respecting exact column order
        feature_order = st.session_state.feature_names
        input_df = pd.DataFrame([user_input])
        
        # Reorder columns to match training
        input_df = input_df[feature_order]
        
        pred_proba = model.predict_proba(input_df)[:, 1][0]
        pred_class = 1 if pred_proba >= results['best_thr'] else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Score", f"{pred_proba:.1%}")
            st.metric("Prediction", "HIGH RISK" if pred_class == 1 else "LOW RISK")
        with col2:
            color = "🔴" if pred_class == 1 else "🟢"
            st.markdown(f"### {color} **{pred_class}**")