import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import make_classification, make_moons, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tree Explorer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 2. THEME DEFINITIONS
# ─────────────────────────────────────────────────────────────
THEMES = {
    "Light": {
        "fig_bg": "#FFFFFF",
        "ax_bg": "#F8F9FA",
        "sidebar_bg": "#F8F9FA",
        "card_bg": "#FFFFFF",
        "text": "#202124",
        "muted": "#5F6368",
        "border": "#DADCE0",
        "accent": "#1A73E8",
        "c0": "#E84135", "c1": "#34A853",
        "c0_bg": "#FEEBE9", "c1_bg": "#E6F4EA",
        "grid": "#E8EAED"
    },
    "Dark": {
        "fig_bg": "#202124",
        "ax_bg": "#303134",
        "sidebar_bg": "#2D2E30",
        "card_bg": "#202124",
        "text": "#E8EAED",
        "muted": "#9AA0A6",
        "border": "#3C4043",
        "accent": "#8AB4F8",
        "c0": "#F28B82", "c1": "#81C995",
        "c0_bg": "#3C1E1E", "c1_bg": "#0F2618",
        "grid": "#3C4043"
    }
}

def apply_theme_engine(mode):
    """The Single Command: Syncs CSS variables and Matplotlib defaults."""
    T = THEMES[mode]
    
    # Inject CSS Variables into assets/style.css
    try:
        with open("assets/style.css", "r") as f:
            css_base = f.read()
        st.markdown(f"""
            <style>
            :root {{
                --bg-color: {T['fig_bg']};
                --sidebar-bg: {T['sidebar_bg']};
                --card-bg: {T['card_bg']};
                --text-color: {T['text']};
                --muted-color: {T['muted']};
                --border-color: {T['border']};
                --accent-color: {T['accent']};
            }}
            {css_base}
            </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Error: 'assets/style.css' not found.")

    # Matplotlib Global Overrides
    plt.rcParams.update({
        "figure.facecolor": T["fig_bg"],
        "axes.facecolor": T["ax_bg"],
        "axes.edgecolor": T["border"],
        "axes.labelcolor": T["muted"],
        "axes.titlecolor": T["text"],
        "grid.color": T["grid"],
        "text.color": T["text"],
        "xtick.color": T["muted"],
        "ytick.color": T["muted"],
        "legend.facecolor": T["fig_bg"],
        "legend.edgecolor": T["border"],
        "font.family": "sans-serif"
    })
    return T

# ─────────────────────────────────────────────────────────────
# 3. SIDEBAR & THEME TOGGLE
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌳 Tree Explorer")
    mode = st.radio("Theme Mode", ["Light", "Dark"], horizontal=True, label_visibility="collapsed")
    T = apply_theme_engine(mode)
    
    st.divider()
    st.markdown("**Dataset Configuration**")
    dataset_type = st.selectbox("Select Data", ["Two Moons", "Two Circles", "Breast Cancer", "Random Blobs"])
    n_samples = st.slider("Samples", 100, 2000, 600, 100)
    noise = st.slider("Noise", 0.0, 0.5, 0.2, 0.05)
    
    st.divider()
    st.markdown("**Tree Hyperparameters**")
    max_depth = st.slider("Max Depth", 1, 15, 4)
    min_split = st.slider("Min Samples Split", 2, 50, 2)
    min_leaf = st.slider("Min Samples Leaf", 1, 50, 1)
    criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])

# ─────────────────────────────────────────────────────────────
# 4. DATA PROCESSING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def get_cached_data(name, n, noise_val):
    if name == "Two Moons":
        X, y = make_moons(n_samples=n, noise=noise_val, random_state=42)
        cols = ["X1", "X2"]
    elif name == "Two Circles":
        X, y = make_circles(n_samples=n, noise=noise_val, factor=0.5, random_state=42)
        cols = ["X1", "X2"]
    elif name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data[:, :2], data.target
        cols = data.feature_names[:2]
    else:
        X, y = make_classification(n_samples=n, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        cols = ["Feature A", "Feature B"]
    return X, y, cols

X, y, feature_names = get_cached_data(dataset_type, n_samples, noise)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Fit Model
clf = DecisionTreeClassifier(
    max_depth=max_depth, 
    min_samples_split=min_split, 
    min_samples_leaf=min_leaf, 
    criterion=criterion,
    random_state=42
)
clf.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────
# 5. DASHBOARD LAYOUT
# ─────────────────────────────────────────────────────────────
st.title("Decision Tree Explorer")
st.markdown(f"Currently exploring **{dataset_type}** dataset with **{max_depth}** depth.")

# Metrics Row
tr_acc = accuracy_score(y_train, clf.predict(X_train))
te_acc = accuracy_score(y_test, clf.predict(X_test))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Train Accuracy", f"{tr_acc:.1%}")
m2.metric("Test Accuracy", f"{te_acc:.1%}", delta=f"{te_acc-tr_acc:+.1%}")
m3.metric("Tree Nodes", clf.tree_.node_count)
m4.metric("Leaf Nodes", clf.get_n_leaves())

st.divider()

# Tabs for visual modularity
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Boundary", "🌳 Structure", "📈 Importance", "📋 Report"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap_bg = ListedColormap([T["c0_bg"], T["c1_bg"]])
    
    DecisionBoundaryDisplay.from_estimator(
        clf, X, ax=ax, response_method="predict", 
        cmap=cmap_bg, alpha=0.8, grid_resolution=200
    )
    
    # Plot points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=[T['c0'] if i==0 else T['c1'] for i in y_train], 
               edgecolor=T['fig_bg'], s=30, label="Train", alpha=0.7)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=[T['c0'] if i==0 else T['c1'] for i in y_test], 
               edgecolor=T['fig_bg'], s=60, marker='X', label="Test")
    
    ax.set_title("Decision Boundary & Data Distribution")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.markdown("### Tree Hierarchy")
    fig_tree, ax_tree = plt.subplots(figsize=(14, 7))
    plot_tree(
        clf, feature_names=feature_names, 
        class_names=["Class 0", "Class 1"], 
        filled=True, rounded=True, ax=ax_tree, fontsize=9
    )
    st.pyplot(fig_tree)

with tab3:
    st.markdown("### Feature Importances")
    fi_data = pd.Series(clf.feature_importances_, index=feature_names)
    st.bar_chart(fi_data, color=T["accent"])
    st.dataframe(fi_data.to_frame(name="Importance Score").style.background_gradient(cmap="Blues"))

with tab4:
    col_cm, col_rep = st.columns([1, 1])
    with col_cm:
        st.markdown("**Confusion Matrix (Test)**")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test, y_test, ax=ax_cm, 
            cmap="Blues" if mode=="Light" else "Greys", colorbar=False
        )
        st.pyplot(fig_cm)
    with col_rep:
        st.markdown("**Classification Metrics**")
        report = classification_report(y_test, clf.predict(X_test), output_dict=True)
        st.table(pd.DataFrame(report).T.iloc[:2, :3])

st.divider()
st.caption("Google Minimalist Design — Fully Modular Theme Sync.")