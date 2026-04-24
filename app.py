import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="SteelGuard — Fault Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Force sidebar always open via JS ────────────────────────────────────────
st.markdown("""
<script>
(function keepSidebarOpen() {
    const hide = setInterval(() => {
        const btns = window.parent.document.querySelectorAll(
            '[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="baseButton-headerNoPadding"]'
        );
        btns.forEach(b => b.style.display = 'none');

        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.transform = 'translateX(0)';
            sidebar.style.minWidth = '260px';
            sidebar.style.width = '260px';
            sidebar.style.marginLeft = '0';
            if (sidebar.getAttribute('aria-expanded') === 'false') {
                sidebar.setAttribute('aria-expanded', 'true');
            }
        }
    }, 400);
    setTimeout(() => clearInterval(hide), 10000);
})();
</script>
""", unsafe_allow_html=True)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Barlow+Condensed:wght@400;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0c0f !important;
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1c2333 !important;
    padding-top: 0 !important;
    min-width: 300px !important;
    max-width: 300px !important;
    width: 300px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0; min-width: 300px !important; }
[data-testid="stSidebar"] section { min-width: 300px !important; }

/* ── Sidebar top brand strip ── */
.brand-strip {
    background: #f97316;
    padding: 22px 24px 18px 24px;
    margin-bottom: 0;
    width: 100%;
    box-sizing: border-box;
}
.brand-strip .company { font-family: 'Barlow Condensed', sans-serif; font-size: 1.8rem; font-weight: 800; color: #fff; letter-spacing: 3px; text-transform: uppercase; margin: 0; line-height: 1.1; white-space: nowrap; }
.brand-strip .tagline { font-size: 0.62rem; color: rgba(255,255,255,0.85); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ── Sidebar nav label ── */
.nav-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4a5568; padding: 18px 20px 8px 20px; display: block; }

/* ── Navigation radio buttons ── */
[data-testid="stRadio"] { padding: 0 12px; }
[data-testid="stRadio"] > div { gap: 4px !important; }
[data-testid="stRadio"] label {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    padding: 8px 10px !important;
    border-radius: 6px !important;
    width: 100% !important;
    display: block !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stRadio"] label:hover { background: #1c2333 !important; color: #f97316 !important; }
[data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stRadio"] label[data-checked="true"] { color: #f97316 !important; background: #1a0e04 !important; }

/* ── Sidebar status ── */
.status-block { margin: 0 12px; padding: 10px 14px; border-radius: 6px; font-size: 0.82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.status-ok   { background: #0d2818; border: 1px solid #16a34a; color: #4ade80; }
.status-warn { background: #1c1404; border: 1px solid #d97706; color: #fbbf24; }

/* ── Main content area ── */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

/* ── Force sidebar always visible ── */
[data-testid="collapsedControl"]                { display: none !important; }
[data-testid="stSidebarCollapsedControl"]       { display: none !important; }
button[data-testid="baseButton-headerNoPadding"]{ display: none !important; }
[data-testid="stSidebar"][aria-expanded="false"]{ margin-left: 0 !important; transform: translateX(0) !important; display: flex !important; min-width: 300px !important; }

/* ── Top bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 0 20px 0;
    border-bottom: 1px solid #1c2333;
    margin-bottom: 24px;
}
.topbar .page-title { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; color: #e2e8f0 !important; letter-spacing: 1px; text-transform: uppercase; margin: 0; display: block !important; visibility: visible !important; }
.topbar .breadcrumb { font-size: 0.7rem; color: #4a5568; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; display: block !important; }
.topbar .badge { background: #f97316; color: #fff; font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding: 4px 10px; border-radius: 3px; white-space: nowrap; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 24px; }
.kpi {
    flex: 1;
    background: #0d1117;
    border: 1px solid #1c2333;
    border-top: 3px solid #f97316;
    border-radius: 4px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.kpi::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle at top right, #f9731610, transparent);
}
.kpi .kpi-val { font-family: 'Barlow Condensed', sans-serif; font-size: 2.2rem; font-weight: 700; color: #f97316; line-height: 1; }
.kpi .kpi-lbl { font-size: 0.65rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4a5568; margin-top: 5px; }
.kpi .kpi-sub { font-size: 0.7rem; color: #2d3748; margin-top: 2px; }

/* ── Section header ── */
.sec-hdr {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1c2333;
}
.sec-hdr .sec-num { font-family: 'Barlow Condensed', sans-serif; font-size: 0.7rem; font-weight: 700; color: #f97316; letter-spacing: 2px; background: #1a0e04; border: 1px solid #f9731640; padding: 2px 7px; border-radius: 3px; }
.sec-hdr .sec-ttl { font-family: 'Barlow Condensed', sans_serif; font-size: 1rem; font-weight: 700; color: #cbd5e1; letter-spacing: 1px; text-transform: uppercase; }

/* ── Chart panels ── */
.chart-panel {
    background: #0d1117;
    border: 1px solid #1c2333;
    border-radius: 6px;
    padding: 16px;
    height: 100%;
}
.chart-title { font-size: 0.7rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4a5568; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1c2333; }

/* ── Result card ── */
.result-panel {
    background: #0d1117;
    border: 1px solid #1c2333;
    border-left: 4px solid #f97316;
    border-radius: 4px;
    padding: 24px 28px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 24px;
}
.result-icon { font-size: 2.5rem; }
.result-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; color: #4a5568; }
.result-value { font-family: 'Barlow Condensed', sans-serif; font-size: 2.8rem; font-weight: 800; color: #f97316; letter-spacing: 2px; text-transform: uppercase; line-height: 1; margin-top: 4px; }
.result-conf { font-size: 0.75rem; color: #4a5568; margin-top: 6px; }

/* ── Input grid label ── */
.input-grid-header { font-size: 0.65rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4a5568; margin-bottom: 10px; }

/* ── Inline tag ── */
.tag { display: inline-block; background: #1a0e04; border: 1px solid #f9731640; color: #f97316; font-size: 0.72rem; font-weight: 500; padding: 3px 10px; border-radius: 3px; margin: 3px 2px; letter-spacing: 0.5px; }

/* ── Table styling ── */
[data-testid="stDataFrame"] { border: 1px solid #1c2333 !important; border-radius: 6px; }

/* ── Alert/info override ── */
[data-testid="stAlert"] { background: #0d1117 !important; border: 1px solid #1c2333 !important; }

/* ── Divider ── */
.divider { height: 1px; background: #1c2333; margin: 24px 0; }

/* ── About table ── */
.spec-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.spec-table th { font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: #4a5568; font-weight: 600; padding: 8px 12px; border-bottom: 1px solid #1c2333; text-align: left; }
.spec-table td { padding: 8px 12px; border-bottom: 1px solid #0d1117; color: #94a3b8; vertical-align: top; }
.spec-table td:first-child { color: #cbd5e1; font-weight: 500; white-space: nowrap; }
.spec-table tr:last-child td { border-bottom: none; }

/* ── About panel ── */
.about-panel { background: #0d1117; border: 1px solid #1c2333; border-radius: 6px; padding: 20px; margin-bottom: 14px; }
.about-panel h4 { font-family: 'Barlow Condensed', sans-serif; font-size: 0.8rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #f97316; margin: 0 0 14px 0; padding-bottom: 8px; border-bottom: 1px solid #1c2333; }
.about-panel p, .about-panel li { color: #64748b; font-size: 0.82rem; line-height: 1.6; }
.about-panel code { background: #161b22; color: #f97316; padding: 1px 6px; border-radius: 3px; font-size: 0.8rem; }

/* ── Streamlit radio override ── */
[data-testid="stRadio"] label { font-size: 0.82rem !important; color: #94a3b8 !important; }
[data-testid="stRadio"] label:hover { color: #f97316 !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input { background: #0d1117 !important; border-color: #1c2333 !important; color: #e2e8f0 !important; font-size: 0.82rem !important; }
[data-testid="stNumberInput"] input:focus { border-color: #f97316 !important; box-shadow: 0 0 0 1px #f97316 !important; }

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #f97316 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #fff !important;
    padding: 10px 24px !important;
    transition: background 0.15s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { background: #ea6c0a !important; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib industrial style ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#0d1117',
    'axes.edgecolor':   '#1c2333',
    'axes.labelcolor':  '#4a5568',
    'xtick.color':      '#4a5568',
    'ytick.color':      '#4a5568',
    'text.color':       '#94a3b8',
    'grid.color':       '#1c2333',
    'axes.grid':        True,
    'grid.alpha':       0.5,
    'grid.linestyle':   '--',
    'axes.titlecolor':  '#cbd5e1',
    'axes.titlesize':   10,
    'axes.titleweight': '600',
    'axes.labelsize':   8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ─── Constants ──────────────────────────────────────────────────────────────────
FAULT_COLS    = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
MAPPING       = {'Pastry':0,'Dirtiness':1,'K_Scatch':2,'Bumps':3,'Other_Faults':4,'Stains':5,'Z_Scratch':6}
REVERSE_MAPPING = {v:k for k,v in MAPPING.items()}
DROP_COLS     = ['Square_Index','Sum_of_Luminosity','X_Minimum','X_Perimeter','SigmoidOfAreas','Edges_X_Index','Y_Minimum','Y_Maximum']
ACCENT        = '#f97316'
PALETTE       = ['#f97316','#3b82f6','#10b981','#f59e0b','#8b5cf6','#ec4899','#06b6d4']

# ─── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(uploaded_file):
    d = pd.read_csv(uploaded_file)
    d.drop_duplicates(inplace=True)
    d['Fault_Types'] = d[FAULT_COLS].idxmax(axis=1).map(MAPPING)
    d.drop(columns=FAULT_COLS, inplace=True)
    d.drop(columns=[c for c in DROP_COLS if c in d.columns], inplace=True)
    d['Fault_Label'] = d['Fault_Types'].map(REVERSE_MAPPING)
    X = d.drop(columns=['Fault_Types','Fault_Label'])
    y = d['Fault_Types']
    Xc = X.copy()
    for col in Xc.columns:
        q1,q3 = Xc[col].quantile([0.01,0.99])
        Xc[col] = Xc[col].clip(q1,q3)
    if Xc.isnull().sum().sum() > 0:
        Xc = Xc.fillna(Xc.mean())
    sc = StandardScaler()
    Xs = sc.fit_transform(Xc)
    return d, Xc, Xs, y, sc

@st.cache_resource
def load_model():
    p = os.path.join(os.path.dirname(__file__), "model.pkl")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def load_scaler():
    p = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    return joblib.load(p) if os.path.exists(p) else None

def predict_single(df, scaler, model):
    Xs = scaler.transform(df)
    pred = model.predict(Xs)[0]
    try:    proba = model.predict_proba(Xs)[0]
    except: proba = None
    return REVERSE_MAPPING.get(int(pred), str(pred)), proba

def sec(num, title):
    st.markdown(f'<div class="sec-hdr"><span class="sec-num">{num}</span><span class="sec-ttl">{title}</span></div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="brand-strip">
    <div class="company">⬡ SteelGuard</div>
    <div class="tagline">Surface Fault Detection System</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["📊  Analysis", "🔬  Predict", "📋  System Info"],
    label_visibility="collapsed"
)

st.sidebar.markdown('<div class="nav-label">Data Source</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    st.sidebar.markdown('<div class="status-block status-ok">▶ Dataset connected</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-block status-warn">⚠ No dataset — upload CSV</div>', unsafe_allow_html=True)

model_loaded = load_model() is not None
st.sidebar.markdown('<div class="nav-label">Model Status</div>', unsafe_allow_html=True)
if model_loaded:
    st.sidebar.markdown('<div class="status-block status-ok">▶ Model ready</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-block status-warn">⚠ model.pkl not found</div>', unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:0.65rem;color:#4a5568;text-align:center;letter-spacing:2px;text-transform:uppercase;padding:8px 0'>SteelGuard v1.0 · Industrial ML</p>", unsafe_allow_html=True)

# ── Force sidebar to always stay open ─────────────────────────────────────────
st.markdown("""
<script>
(function keepSidebarOpen() {
    function expand() {
        // Find the collapse button and hide it
        var btns = window.parent.document.querySelectorAll('[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]');
        btns.forEach(function(b) { b.style.display = 'none'; });

        // If sidebar is collapsed, click the expand button
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            var expandBtn = window.parent.document.querySelector('[data-testid="stSidebarContent"]');
            // Try to find and click the open button
            var openBtns = window.parent.document.querySelectorAll('button[kind="header"]');
            openBtns.forEach(function(b) {
                if (b.getAttribute('aria-expanded') === 'false') { b.click(); }
            });
        }
    }
    // Run on load and observe DOM changes
    expand();
    setInterval(expand, 500);
})();
</script>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS PAGE
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊  Analysis":

    st.markdown("""
    <div class="topbar">
        <div>
            <div class="page-title">Surface Analysis Dashboard</div>
            <div class="breadcrumb">SteelGuard / Analysis / Overview</div>
        </div>
        <div class="badge">Live Data</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.warning("Upload the steel_plates_faults.csv dataset using the sidebar to begin analysis.")
        st.stop()

    d, X, Xs, y, sc = load_and_preprocess(uploaded_file)
    fcols = X.columns.tolist()

    # KPI row
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-val">{d.shape[0]:,}</div><div class="kpi-lbl">Total Plates</div><div class="kpi-sub">in dataset</div></div>
        <div class="kpi"><div class="kpi-val">{X.shape[1]}</div><div class="kpi-lbl">Sensor Features</div><div class="kpi-sub">per inspection</div></div>
        <div class="kpi"><div class="kpi-val">{y.nunique()}</div><div class="kpi-lbl">Fault Types</div><div class="kpi-sub">classified</div></div>
        <div class="kpi"><div class="kpi-val">{d.isnull().sum().sum()}</div><div class="kpi-lbl">Missing Values</div><div class="kpi-sub">after cleaning</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("▶  Raw Inspection Records"):
        st.dataframe(d.head(15), use_container_width=True)
    with st.expander("▶  Feature Statistics"):
        st.dataframe(d.describe().T.style.background_gradient(cmap='YlOrBr'), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Row 1
    sec("01", "Fault Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-panel"><div class="chart-title">Fault Type Frequency</div>', unsafe_allow_html=True)
        counts = y.map(REVERSE_MAPPING).value_counts()
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor='none', width=0.55)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                    f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=7, color='#4a5568')
        ax.set_xlabel("Fault Category"); ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha='right', fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-panel"><div class="chart-title">Steel Grade Breakdown</div>', unsafe_allow_html=True)
        if 'TypeOfSteel_A300' in d.columns:
            vals = d[['TypeOfSteel_A300','TypeOfSteel_A400']].sum()
            fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
            wedges, texts, autotexts = ax2.pie(
                vals.values, labels=vals.index,
                colors=['#f97316','#3b82f6'],
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(edgecolor='#0d1117', linewidth=2),
                textprops=dict(color='#94a3b8', fontsize=8)
            )
            for at in autotexts: at.set_color('#fff'); at.set_fontsize(8)
            ax2.set_facecolor('#0d1117'); fig2.patch.set_facecolor('#0d1117')
            plt.tight_layout()
            st.pyplot(fig2); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    sec("02", "Feature Intelligence")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-panel"><div class="chart-title">Sensor Correlation Matrix</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(np.corrcoef(Xs.T), xticklabels=fcols, yticklabels=fcols,
                    cmap='RdYlBu_r', ax=ax3, annot=False,
                    linewidths=0.4, linecolor='#0a0c0f',
                    cbar_kws={'shrink':0.7})
        ax3.set_title(""); plt.xticks(fontsize=5.5, rotation=90); plt.yticks(fontsize=5.5)
        plt.tight_layout(); st.pyplot(fig3); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-panel"><div class="chart-title">PCA Fault Separation</div>', unsafe_allow_html=True)
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(Xs)
        yv = y.values
        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        for idx, fid in enumerate(sorted(y.unique())):
            m = yv == fid
            ax4.scatter(Xp[m,0], Xp[m,1], label=REVERSE_MAPPING[fid],
                        alpha=0.45, color=PALETTE[idx % len(PALETTE)], s=10, linewidths=0)
        ax4.set_xlabel(f"PC1  {pca.explained_variance_ratio_[0]:.1%}")
        ax4.set_ylabel(f"PC2  {pca.explained_variance_ratio_[1]:.1%}")
        ax4.legend(fontsize=7, facecolor='#0d1117', edgecolor='#1c2333',
                   labelcolor='#94a3b8', markerscale=1.8, framealpha=0.8)
        plt.tight_layout(); st.pyplot(fig4); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3 — feature inspector
    sec("03", "Feature Inspector")
    sel = st.selectbox("Select sensor/feature", fcols, label_visibility="visible")
    c5, c6 = st.columns(2)

    with c5:
        st.markdown('<div class="chart-panel"><div class="chart-title">Distribution</div>', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.hist(X[sel].dropna(), bins=35, color=ACCENT, edgecolor='none', alpha=0.85)
        ax5.set_xlabel(sel); ax5.set_ylabel("Count")
        plt.tight_layout(); st.pyplot(fig5); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="chart-panel"><div class="chart-title">Box — Spread & Outliers</div>', unsafe_allow_html=True)
        fig6, ax6 = plt.subplots(figsize=(5, 3))
        bp = ax6.boxplot(X[sel].dropna(), patch_artist=True, vert=False,
                    boxprops=dict(facecolor='#1a0e04', color=ACCENT),
                    medianprops=dict(color='#10b981', linewidth=2),
                    whiskerprops=dict(color='#4a5568'),
                    capprops=dict(color='#4a5568'),
                    flierprops=dict(marker='|', color='#4a5568', markersize=4, alpha=0.4))
        ax6.set_xlabel(sel); ax6.set_yticks([])
        plt.tight_layout(); st.pyplot(fig6); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # All histograms
    sec("04", "All Sensor Readings")
    nc = 4
    nr = (len(fcols)+nc-1)//nc
    fig7, axes = plt.subplots(nr, nc, figsize=(18, nr*2.6))
    flat = axes.flatten()
    for i, col in enumerate(fcols):
        flat[i].hist(X[col], bins=25, color=PALETTE[i%len(PALETTE)], edgecolor='none', alpha=0.8)
        flat[i].set_title(col, fontsize=7.5, color='#94a3b8')
        flat[i].tick_params(labelsize=6)
    for j in range(len(fcols), len(flat)): flat[j].set_visible(False)
    plt.tight_layout(pad=1.2); st.pyplot(fig7); plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# PREDICT PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Predict":

    st.markdown("""
    <div class="topbar">
        <div>
            <div class="page-title">Fault Classification Engine</div>
            <div class="breadcrumb">SteelGuard / Predict / Single &amp; Batch</div>
        </div>
        <div class="badge">ML Model</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.warning("Upload steel_plates_faults.csv via the sidebar first.")
        st.stop()

    d, X, Xs, y, sc = load_and_preprocess(uploaded_file)
    fcols = X.columns.tolist()
    saved_sc = load_scaler()
    if saved_sc is not None: sc = saved_sc
    model = load_model()

    if model is None:
        st.error("model.pkl not found at the configured path. Train and save your model first.")
        st.code("joblib.dump(stacking_clf, 'model.pkl')\njoblib.dump(s, 'scaler.pkl')", language="python")
        st.stop()

    sec("01", "Sensor Input — Single Plate")
    st.markdown('<p class="input-grid-header">Enter sensor readings below. Default values are dataset medians.</p>', unsafe_allow_html=True)

    medians = X.median()
    input_values = {}
    cols = st.columns(4)
    for i, feat in enumerate(fcols):
        with cols[i % 4]:
            input_values[feat] = st.number_input(feat, value=float(medians[feat]), format="%.4f", key=feat)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("▶  Run Fault Classification", type="primary", use_container_width=True)

    if run:
        idf = pd.DataFrame([input_values])
        for col in idf.columns:
            q1,q3 = X[col].quantile([0.01,0.99])
            idf[col] = idf[col].clip(q1,q3)
        label, proba = predict_single(idf, sc, model)

        st.markdown(f"""
        <div class="result-panel">
            <div class="result-icon">⬡</div>
            <div>
                <div class="result-label">Classification Result</div>
                <div class="result-value">{label}</div>
                <div class="result-conf">Surface defect type identified by StackingClassifier ensemble</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if proba is not None:
            sec("02", "Confidence Breakdown")
            try:    classes = [REVERSE_MAPPING.get(int(c), str(c)) for c in model.classes_]
            except: classes = [str(c) for c in model.classes_]

            prob_df = (pd.DataFrame({"Fault Type": classes, "Probability": proba})
                       .sort_values("Probability", ascending=False).reset_index(drop=True))

            cc, ct = st.columns([3,2])
            with cc:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                colors = [ACCENT if ft == label else '#1c2333' for ft in prob_df["Fault Type"]]
                bars = ax.barh(prob_df["Fault Type"], prob_df["Probability"],
                               color=colors, edgecolor='none', height=0.55)
                ax.set_xlim(0,1); ax.set_xlabel("Confidence Score")
                ax.invert_yaxis()
                for bar, val in zip(bars, prob_df["Probability"]):
                    ax.text(min(bar.get_width()+0.02, 0.97), bar.get_y()+bar.get_height()/2,
                            f'{val:.3f}', va='center', fontsize=8,
                            color='#f97316' if bar.get_facecolor()[0] > 0.5 else '#4a5568')
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with ct:
                st.markdown("<br>", unsafe_allow_html=True)
                prob_df["Confidence"] = prob_df["Probability"].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(prob_df[["Fault Type","Confidence"]], use_container_width=True, hide_index=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    sec("03", "Batch Processing")
    st.caption("Upload a CSV file with matching sensor columns for bulk classification.")

    batch = st.file_uploader("Batch CSV", type=["csv"], key="batch", label_visibility="collapsed")
    if batch:
        bdf = pd.read_csv(batch)
        missing = [c for c in fcols if c not in bdf.columns]
        if missing:
            st.error(f"Missing sensor columns: {missing}")
        else:
            with st.spinner("Classifying plates..."):
                bi = bdf[fcols].copy()
                for col in bi.columns:
                    q1,q3 = X[col].quantile([0.01,0.99])
                    bi[col] = bi[col].clip(q1,q3)
                preds = model.predict(sc.transform(bi))
                bdf['Fault_Class'] = [REVERSE_MAPPING.get(int(p), str(p)) for p in preds]

            st.markdown(f"""
            <div class="kpi-row" style="margin-top:12px">
                <div class="kpi"><div class="kpi-val">{len(preds):,}</div><div class="kpi-lbl">Plates Processed</div></div>
                <div class="kpi"><div class="kpi-val">{bdf['Fault_Class'].nunique()}</div><div class="kpi-lbl">Fault Types Found</div></div>
                <div class="kpi"><div class="kpi-val">{bdf['Fault_Class'].value_counts().index[0]}</div><div class="kpi-lbl">Most Common Fault</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(bdf.head(25), use_container_width=True)
            st.download_button("⬇  Export Classification Report", bdf.to_csv(index=False).encode(),
                               "fault_report.csv", "text/csv", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# SYSTEM INFO PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📋  System Info":

    st.markdown("""
    <div class="topbar">
        <div>
            <div class="page-title">System Information</div>
            <div class="breadcrumb">SteelGuard / System / Specification</div>
        </div>
        <div class="badge">Docs</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-panel">
            <h4>Detectable Fault Classes</h4>
            <table class="spec-table">
                <tr><th>ID</th><th>Fault Type</th><th>Description</th></tr>
                <tr><td>0</td><td>Pastry</td><td>Surface irregularity pattern</td></tr>
                <tr><td>1</td><td>Dirtiness</td><td>Contamination on surface</td></tr>
                <tr><td>2</td><td>K_Scatch</td><td>Lateral scratch defect</td></tr>
                <tr><td>3</td><td>Bumps</td><td>Raised surface anomaly</td></tr>
                <tr><td>4</td><td>Other_Faults</td><td>Unclassified defects</td></tr>
                <tr><td>5</td><td>Stains</td><td>Discoloration marks</td></tr>
                <tr><td>6</td><td>Z_Scratch</td><td>Diagonal scratch pattern</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-panel">
            <h4>Required Files</h4>
            <table class="spec-table">
                <tr><th>File</th><th>Purpose</th></tr>
                <tr><td><code>steel_fault_detection.py</code></td><td>Fault detection logic</td></tr>
                <tr><td><code>app.py</code></td><td>Application entry point</td></tr>
                <tr><td><code>model.pkl</code></td><td>Trained StackingClassifier</td></tr>
                <tr><td><code>scaler.pkl</code></td><td>Fitted StandardScaler</td></tr>
                <tr><td><code>requirements.txt</code></td><td>Python dependencies</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-panel">
            <h4>ML Pipeline Specification</h4>
            <table class="spec-table">
                <tr><th>Stage</th><th>Method</th></tr>
                <tr><td>Deduplication</td><td>Exact row matching</td></tr>
                <tr><td>Outlier handling</td><td>1st–99th percentile clip</td></tr>
                <tr><td>Scaling</td><td>StandardScaler (zero mean, unit var)</td></tr>
                <tr><td>Class balancing</td><td>SMOTE oversampling</td></tr>
                <tr><td>Base learner 1</td><td>SVC (RBF kernel, OvR)</td></tr>
                <tr><td>Base learner 2</td><td>RandomForestClassifier (OvR)</td></tr>
                <tr><td>Base learner 3</td><td>XGBClassifier (OvR)</td></tr>
                <tr><td>Tuning</td><td>GridSearchCV · StratifiedKFold k=10</td></tr>
                <tr><td>Meta learner</td><td>LogisticRegression</td></tr>
                <tr><td>Scoring metric</td><td>F1-macro</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-panel">
            <h4>Save Model Artifacts</h4>
        </div>
        """, unsafe_allow_html=True)
        st.code("# Run at end of training\nimport joblib\njoblib.dump(stacking_clf, 'model.pkl')\njoblib.dump(s, 'scaler.pkl')", language="python")

        st.markdown("""
        <div class="about-panel" style="margin-top:14px">
            <h4>Launch Application</h4>
        </div>
        """, unsafe_allow_html=True)
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
