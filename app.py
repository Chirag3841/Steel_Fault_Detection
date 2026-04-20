import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Faults Detection",
    page_icon="🔩",
    layout="wide",
)

# ─── Constants ──────────────────────────────────────────────────────────────────
FAULT_COLS = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
MAPPING = {
    'Pastry': 0, 'Dirtiness': 1, 'K_Scatch': 2,
    'Bumps': 3, 'Other_Faults': 4, 'Stains': 5, 'Z_Scratch': 6
}
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}

DROP_COLS = [
    'Square_Index', 'Sum_of_Luminosity', 'X_Minimum', 'X_Perimeter',
    'SigmoidOfAreas', 'Edges_X_Index', 'Y_Minimum', 'Y_Maximum'
]

# ─── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(uploaded_file):
    d = pd.read_csv(uploaded_file)
    d.drop_duplicates(inplace=True)

    d['Fault_Types'] = d[FAULT_COLS].idxmax(axis=1).map(MAPPING)
    d.drop(columns=FAULT_COLS, inplace=True)

    existing_drop = [c for c in DROP_COLS if c in d.columns]
    d.drop(columns=existing_drop, inplace=True)

    d['Fault_Label'] = d['Fault_Types'].map(REVERSE_MAPPING)

    X = d.drop(columns=['Fault_Types', 'Fault_Label'])
    y = d['Fault_Types']

    # Clip outliers (same logic as training)
    X_clipped = X.copy()
    for col in X_clipped.columns:
        q1, q3 = X_clipped[col].quantile([0.01, 0.99])
        X_clipped[col] = X_clipped[col].clip(q1, q3)

    # Fill nulls if any
    if X_clipped.isnull().sum().sum() > 0:
        X_clipped = X_clipped.fillna(X_clipped.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clipped)

    return d, X_clipped, X_scaled, y, scaler


@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


@st.cache_resource
def load_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    return None


def predict_single(input_df, scaler, model):
    X_scaled = scaler.transform(input_df)
    pred = model.predict(X_scaled)[0]
    try:
        proba = model.predict_proba(X_scaled)[0]
    except Exception:
        proba = None
    return REVERSE_MAPPING.get(int(pred), str(pred)), proba


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🔩 Steel Faults Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 EDA", "🤖 Predict", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload `steel_plates_faults.csv`", type=["csv"]
)

# ─── EDA Page ────────────────────────────────────────────────────────────────────
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if uploaded_file is None:
        st.info("👈 Please upload `steel_plates_faults.csv` from the sidebar to begin.")
        st.stop()

    d, X, X_scaled, y, scaler = load_and_preprocess(uploaded_file)
    feature_cols = X.columns.tolist()

    # Overview metrics
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", d.shape[0])
    c2.metric("Features", X.shape[1])
    c3.metric("Fault Classes", y.nunique())

    with st.expander("Show raw data (first 10 rows)"):
        st.dataframe(d.head(10))
    with st.expander("Descriptive Statistics"):
        st.dataframe(d.describe().T)

    # Fault class distribution
    st.subheader("Fault Class Distribution")
    fig, ax = plt.subplots()
    counts = y.map(REVERSE_MAPPING).value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="coral", ax=ax)
    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Fault Types")
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close()

    # Steel type distribution
    if 'TypeOfSteel_A300' in d.columns and 'TypeOfSteel_A400' in d.columns:
        st.subheader("Steel Type Distribution")
        fig2, ax2 = plt.subplots()
        d[['TypeOfSteel_A300', 'TypeOfSteel_A400']].sum().plot(
            kind='bar', color='steelblue', ax=ax2
        )
        ax2.set_title("Type of Steel Distribution")
        plt.xticks(rotation=0)
        st.pyplot(fig2)
        plt.close()

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    corr_matrix = np.corrcoef(X_scaled.T)
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_cols, yticklabels=feature_cols,
        cmap='coolwarm', ax=ax3, annot=False
    )
    ax3.set_title("Feature Correlation Heatmap")
    st.pyplot(fig3)
    plt.close()

    # PCA scatter — fixed: use .values to avoid pandas index mismatch
    st.subheader("PCA 2D Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    y_values = y.values
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10", n_colors=y.nunique())
    for idx, fault_id in enumerate(sorted(y.unique())):
        mask = y_values == fault_id
        ax4.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=REVERSE_MAPPING[fault_id], alpha=0.6, color=palette[idx], s=15
        )
    ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax4.set_title("PCA Visualization of Fault Classes")
    ax4.legend(fontsize=8)
    st.pyplot(fig4)
    plt.close()

    # Per-feature boxplot
    st.subheader("Feature Boxplots")
    selected_feat = st.selectbox("Select a feature", feature_cols)
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    sns.boxplot(y=X[selected_feat], color="lightgreen", ax=ax5)
    ax5.set_title(f"Boxplot: {selected_feat}")
    st.pyplot(fig5)
    plt.close()

    # Histograms (all features)
    st.subheader("Feature Histograms")
    ncols = 4
    nrows = (len(feature_cols) + ncols - 1) // ncols
    fig6, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 3))
    axes_flat = axes.flatten()
    for i, col in enumerate(feature_cols):
        axes_flat[i].hist(X[col], bins=30, color='skyblue', edgecolor='white')
        axes_flat[i].set_title(col, fontsize=8)
    for j in range(len(feature_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()


# ─── Predict Page ────────────────────────────────────────────────────────────────
elif page == "🤖 Predict":
    st.title("🤖 Fault Prediction")

    if uploaded_file is None:
        st.info("👈 Please upload `steel_plates_faults.csv` first (needed for feature ranges).")
        st.stop()

    d, X, X_scaled, y, scaler = load_and_preprocess(uploaded_file)
    feature_cols = X.columns.tolist()

    # Prefer the saved scaler from training
    saved_scaler = load_scaler()
    if saved_scaler is not None:
        scaler = saved_scaler

    model = load_model()

    if model is None:
        st.warning("⚠️ No trained model found. Place `model.pkl` in the same folder as `app.py`.")
        st.code(
            "# Add this at the end of your training script:\n"
            "import joblib\n"
            "joblib.dump(stacking_clf, 'model.pkl')\n"
            "joblib.dump(s, 'scaler.pkl')",
            language="python"
        )
        st.stop()

    # Single prediction
    st.subheader("Single Sample Prediction")
    st.caption("Default values are the median of each feature from your dataset.")

    medians = X.median()
    input_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            input_values[feat] = st.number_input(
                feat,
                value=float(medians[feat]),
                format="%.4f",
                key=feat
            )

    if st.button("🔍 Predict Fault Type", type="primary"):
        input_df = pd.DataFrame([input_values])
        for col in input_df.columns:
            q1, q3 = X[col].quantile([0.01, 0.99])
            input_df[col] = input_df[col].clip(q1, q3)

        fault_label, proba = predict_single(input_df, scaler, model)
        st.success(f"### 🏷️ Predicted Fault: **{fault_label}**")

        if proba is not None:
            st.subheader("Prediction Probabilities")
            try:
                classes = [REVERSE_MAPPING.get(int(c), str(c)) for c in model.classes_]
            except Exception:
                classes = [str(c) for c in model.classes_]
            prob_df = (
                pd.DataFrame({"Fault Type": classes, "Probability": proba})
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.barplot(data=prob_df, x="Fault Type", y="Probability",
                        palette="Blues_r", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Class Probabilities")
            plt.xticks(rotation=30)
            st.pyplot(fig)
            plt.close()
            st.dataframe(prob_df, use_container_width=True)

    # Batch prediction
    st.markdown("---")
    st.subheader("Batch Prediction")
    st.caption("Upload a CSV with the same feature columns (no fault label columns needed).")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            batch_input = batch_df[feature_cols].copy()
            for col in batch_input.columns:
                q1, q3 = X[col].quantile([0.01, 0.99])
                batch_input[col] = batch_input[col].clip(q1, q3)
            X_batch = scaler.transform(batch_input)
            preds = model.predict(X_batch)
            batch_df['Predicted_Fault'] = [
                REVERSE_MAPPING.get(int(p), str(p)) for p in preds
            ]
            st.success(f"✅ Predicted {len(preds)} samples.")
            st.dataframe(batch_df.head(20), use_container_width=True)
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Full Predictions", csv,
                "predictions.csv", "text/csv"
            )


# ─── About Page ──────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ## Steel Surface Fault Detection

    A Streamlit app that detects surface faults in steel plates using an ensemble ML model.

    ### Fault Types
    | ID | Fault |
    |----|-------|
    | 0 | Pastry |
    | 1 | Dirtiness |
    | 2 | K_Scatch |
    | 3 | Bumps |
    | 4 | Other_Faults |
    | 5 | Stains |
    | 6 | Z_Scratch |

    ### ML Pipeline
    1. **Preprocessing** — deduplication, outlier clipping (1st–99th percentile), StandardScaler
    2. **Class Imbalance** — SMOTE oversampling on training set
    3. **Base Models** — SVM, Random Forest, XGBoost (each tuned with GridSearchCV + StratifiedKFold)
    4. **Ensemble** — StackingClassifier (SVM + RF + XGBoost → Logistic Regression meta-learner)

    ### Files needed
    | File | Purpose |
    |------|---------|
    | `app.py` | This Streamlit app |
    | `model.pkl` | Trained StackingClassifier |
    | `scaler.pkl` | Fitted StandardScaler |
    | `requirements.txt` | Python dependencies |

    ### Save model from training script
    ```python
    import joblib
    joblib.dump(stacking_clf, 'model.pkl')
    joblib.dump(s, 'scaler.pkl')
    ```

    ### Run locally
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)