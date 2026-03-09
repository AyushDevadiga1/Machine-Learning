"""
Decision Tree Hyperparameter Explorer
======================================
Run:   streamlit run app.py
Setup: pip install -r requirements.txt
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
warnings.filterwarnings("ignore")

# ── must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Decision Tree Explorer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour tokens (matches config.toml + chart palette) ──────
BLUE   = "#2563EB"
TEAL   = "#0D9488"
ROSE   = "#E11D48"
AMBER  = "#D97706"
GREEN  = "#16A34A"
VIOLET = "#7C3AED"
SLATE  = "#64748B"
INK    = "#0F172A"

BLUE_BG   = "#EFF6FF"
TEAL_BG   = "#F0FDFA"
ROSE_BG   = "#FFF1F2"
AMBER_BG  = "#FFFBEB"
GREEN_BG  = "#F0FDF4"
VIOLET_BG = "#F5F3FF"

# ── minimal CSS — only things config.toml can't do ───────────
st.markdown("""
<style>
/* hide the default Streamlit header chrome */
#MainMenu  { visibility: hidden; }
header     { visibility: hidden; }
footer     { visibility: hidden; }

/* tighter top padding */
.block-container { padding-top: 1.6rem !important; max-width: 1400px !important; }

/* metric card polish */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px 12px 20px;
    box-shadow: 0 1px 4px rgba(15,23,42,0.06);
}
[data-testid="stMetricLabel"] > div {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: #94A3B8 !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}
[data-testid="stMetricDelta"] > div { font-size: 12px !important; }

/* sidebar width */
[data-testid="stSidebar"] { min-width: 270px !important; max-width: 270px !important; }

/* divider colour */
hr { border-color: #E2E8F0 !important; margin: 0.8rem 0 !important; }

/* tab strip */
[data-baseweb="tab-list"] { gap: 2px !important; }
[data-baseweb="tab"] { padding: 8px 18px !important; border-radius: 8px !important; }

/* expander border */
[data-testid="stExpander"] { border: 1px solid #E2E8F0 !important; border-radius: 10px !important; }

/* button */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── matplotlib style ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FFFFFF",
    "axes.facecolor":    "#F8FAFC",
    "axes.edgecolor":    "#E2E8F0",
    "axes.labelcolor":   SLATE,
    "axes.titlecolor":   INK,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.grid":         True,
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.7,
    "xtick.color":       SLATE,
    "ytick.color":       SLATE,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#E2E8F0",
    "legend.fontsize":   9,
    "font.size":         9,
    "text.color":        INK,
})


# ════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset(name, n_samples, noise, seed):
    if name == "Two Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        fn = ["x₁", "x₂"]

    elif name == "Two Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.45, random_state=seed)
        fn = ["x₁", "x₂"]

    elif name == "XOR Blobs":
        rng = np.random.default_rng(seed)
        pts = []
        for cx, cy, c in [(-1.5,1.5,0),(1.5,-1.5,0),(1.5,1.5,1),(-1.5,-1.5,1)]:
            n = n_samples // 4
            pts.extend(zip(rng.normal(cx, .55 + noise*2.5, n),
                           rng.normal(cy, .55 + noise*2.5, n), [c]*n))
        arr = np.array(pts)
        X, y = arr[:, :2], arr[:, 2].astype(int)
        fn = ["x₁", "x₂"]

    elif name == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        X, y = d.data[:, :2], d.target
        fn = list(d.feature_names[:2])
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), min(n_samples, len(y)), replace=False)
        X, y = X[idx], y[idx]

    else:  # Linear
        X, y = make_classification(
            n_samples=n_samples, n_features=2,
            n_informative=2, n_redundant=0,
            n_clusters_per_class=2,
            class_sep=max(0.1, 1.2 - noise*3),
            random_state=seed,
        )
        fn = ["Feature A", "Feature B"]

    return X, y, fn


# ════════════════════════════════════════════════════════════
#  CHART HELPERS
# ════════════════════════════════════════════════════════════
def make_fig(w=10, h=4.5, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor("#FFFFFF")
    return fig, axes


def style_ax(ax):
    ax.set_facecolor("#F8FAFC")
    for sp in ax.spines.values():
        sp.set_edgecolor("#E2E8F0")
    ax.tick_params(colors=SLATE, labelsize=8)
    return ax


# ── Decision Boundary ────────────────────────────────────────
def fig_boundary(clf, X_tr, X_te, y_tr, y_te, fn):
    fig, axes = make_fig(w=13, h=5, ncols=2)
    cmap_bg = ListedColormap([ROSE_BG, TEAL_BG])

    for ax, (Xs, ys), title in zip(
        axes,
        [(X_tr, y_tr), (X_te, y_te)],
        ["Training set", "Test set"],
    ):
        style_ax(ax)
        try:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.6,
                response_method="predict_proba",
                plot_method="pcolormesh", cmap=cmap_bg,
            )
        except Exception:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.6,
                response_method="predict",
                plot_method="pcolormesh", cmap=cmap_bg,
            )
        cols = [ROSE if v == 0 else TEAL for v in ys]
        ax.scatter(Xs[:, 0], Xs[:, 1], c=cols, s=24,
                   edgecolors="white", linewidths=0.7, zorder=5)
        ax.legend(handles=[
            mpatches.Patch(color=TEAL, label="Class 1"),
            mpatches.Patch(color=ROSE, label="Class 0"),
        ], framealpha=0.9)
        ax.set_title(title, pad=10)
        ax.set_xlabel(fn[0])
        ax.set_ylabel(fn[1])

    fig.tight_layout(pad=2.5)
    return fig


# ── Tree Plot ────────────────────────────────────────────────
def fig_tree(clf, fn, max_show):
    d  = min(clf.get_depth(), max_show)
    w  = max(10, 5 * (2**d))
    h  = max(5, 2.8 * (d + 1))
    fig, ax = plt.subplots(figsize=(min(w, 28), h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    plot_tree(clf, feature_names=fn, class_names=["Class 0", "Class 1"],
              filled=True, rounded=True, impurity=True, proportion=False,
              max_depth=max_show, ax=ax, fontsize=8, precision=3)
    ax.set_title(
        f"Tree structure — showing top {d+1} of {clf.get_depth()+1} levels"
        if clf.get_depth() > max_show else "Tree structure", pad=8,
    )
    fig.tight_layout()
    return fig


# ── Depth vs Accuracy ────────────────────────────────────────
def fig_depth_acc(X_tr, y_tr, X_te, y_te, cur, criterion):
    depths = list(range(1, 16))
    tr_a, te_a = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=42)
        m.fit(X_tr, y_tr)
        tr_a.append(accuracy_score(y_tr, m.predict(X_tr)) * 100)
        te_a.append(accuracy_score(y_te, m.predict(X_te)) * 100)

    fig, ax = make_fig(w=9, h=4)
    style_ax(ax)
    ax.plot(depths, tr_a, color=BLUE,  lw=2.2, marker="o", ms=5, label="Train accuracy")
    ax.plot(depths, te_a, color=TEAL,  lw=2.2, marker="o", ms=5, label="Test accuracy")
    ax.fill_between(depths, tr_a, te_a,
                    where=[t > v for t, v in zip(tr_a, te_a)],
                    alpha=0.1, color=ROSE, label="Overfit gap")
    ax.axvline(x=cur, color=ROSE, lw=1.8, linestyle="--",
               label=f"Current depth = {cur}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Bias–Variance Tradeoff — Depth vs Accuracy")
    ax.set_ylim(30, 103)
    ax.set_xlim(0.5, 15.5)
    ax.legend()
    fig.tight_layout()
    return fig


# ── Feature Importance ───────────────────────────────────────
def fig_feat_imp(clf, fn):
    fi  = clf.feature_importances_
    fig, ax = make_fig(w=7, h=max(3, len(fn)*0.6+1.5))
    style_ax(ax)
    colors = [BLUE if v == max(fi) else "#93C5FD" for v in fi]
    bars = ax.barh(fn, fi, color=colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars, fi):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color=SLATE, fontsize=9)
    ax.set_xlim(0, max(fi)*1.35)
    ax.set_title("Feature Importances (Gini-based)")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ── Confusion Matrix ─────────────────────────────────────────
def fig_confusion(clf, X_te, y_te):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, clf.predict(X_te)),
        display_labels=["Class 0", "Class 1"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test set")
    for txt in ax.texts:
        txt.set_color(INK)
        txt.set_fontsize(13)
        txt.set_fontweight("bold")
    for sp in ax.spines.values(): sp.set_edgecolor("#E2E8F0")
    fig.tight_layout()
    return fig


# ── Leaf distributions ───────────────────────────────────────
def fig_leaves(clf):
    s_list, g_list = [], []
    def walk(nid):
        lc = clf.tree_.children_left[nid]
        if lc == -1:
            s_list.append(clf.tree_.n_node_samples[nid])
            g_list.append(clf.tree_.impurity[nid])
        else:
            walk(lc); walk(clf.tree_.children_right[nid])
    walk(0)

    fig, axes = make_fig(w=10, h=4, ncols=2)
    for ax, data, title, color in [
        (axes[0], s_list, "Samples per Leaf",       BLUE),
        (axes[1], g_list, "Gini Impurity per Leaf",  VIOLET),
    ]:
        style_ax(ax)
        ax.hist(data, bins=min(30, max(5, len(data)//3)),
                color=color, alpha=0.75, edgecolor="white", linewidth=0.6)
        ax.axvline(np.mean(data), color=ROSE, lw=1.8, linestyle="--",
                   label=f"mean = {np.mean(data):.2f}")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()
    fig.tight_layout(pad=2.5)
    return fig


# ════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════
def badge(text, bg, fg):
    return (
        f"<span style='background:{bg};color:{fg};padding:4px 14px;"
        f"border-radius:20px;font-size:12px;font-weight:700'>{text}</span>"
    )


def card(title, body, border_color, bg_color):
    st.markdown(
        f"<div style='background:{bg_color};border-left:4px solid {border_color};"
        f"border-radius:10px;padding:14px 16px;margin-bottom:4px'>"
        f"<div style='font-weight:700;font-size:13px;color:{INK};margin-bottom:4px'>{title}</div>"
        f"<div style='font-size:12.5px;color:{SLATE};line-height:1.65'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌳 Decision Tree\n**Hyperparameter Explorer**")
    st.caption("Adjust any slider and every chart updates instantly.")
    st.divider()

    # ── Dataset ──
    st.markdown("**📦 Dataset**")
    dataset_name = st.selectbox(
        "Dataset",
        ["Two Moons", "Two Circles", "XOR Blobs", "Linear Separable", "Breast Cancer"],
        label_visibility="collapsed",
    )
    n_samples = st.slider("Samples", 200, 2000, 600, step=100)
    noise     = st.slider("Noise level", 0.00, 0.50, 0.20, step=0.05,
                          help="Random scatter added to data points")
    test_pct  = st.slider("Test split %", 10, 40, 25, step=5)
    seed      = st.slider("Random seed", 0, 99, 42)

    st.divider()

    # ── Hyperparameters ──
    st.markdown("**⚙️ Hyperparameters**")
    max_depth = st.slider(
        "max_depth", 1, 15, 4,
        help="Maximum number of levels. Higher → more complex → risk of overfitting."
    )
    min_samples_split = st.slider(
        "min_samples_split", 2, 60, 2,
        help="A node only splits when it has ≥ this many samples. Higher → simpler tree."
    )
    min_samples_leaf = st.slider(
        "min_samples_leaf", 1, 40, 1,
        help="Every leaf must hold ≥ this many samples. Higher → smoother boundaries."
    )
    max_features = st.selectbox(
        "max_features",
        ["all features", "sqrt", "log2"],
        help="Features to consider at each split candidate.",
    )
    criterion = st.selectbox(
        "criterion",
        ["gini", "entropy", "log_loss"],
        help="Impurity measure used to evaluate split quality.",
    )
    splitter = st.selectbox(
        "splitter", ["best", "random"],
        help="'best' always picks the best split; 'random' samples among random candidates.",
    )
    class_weight = st.selectbox(
        "class_weight", ["None", "balanced"],
        help="'balanced' upweights minority-class samples.",
    )

    st.divider()

    # ── Quick presets ──
    st.markdown("**🎯 Quick Presets**")
    c1, c2 = st.columns(2)
    stump    = c1.button("Stump",    use_container_width=True)
    overfit  = c1.button("Overfit",  use_container_width=True)
    balanced = c2.button("Balanced", use_container_width=True)
    pruned   = c2.button("Pruned",   use_container_width=True)

    if stump:    max_depth, min_samples_split, min_samples_leaf = 1, 2, 1
    elif overfit:  max_depth, min_samples_split, min_samples_leaf = 15, 2, 1
    elif balanced: max_depth, min_samples_split, min_samples_leaf = 4, 5, 3
    elif pruned:   max_depth, min_samples_split, min_samples_leaf = 5, 20, 8

    max_features_val  = None if max_features  == "all features" else max_features
    class_weight_val  = None if class_weight  == "None"         else "balanced"


# ════════════════════════════════════════════════════════════
#  FIT MODEL
# ════════════════════════════════════════════════════════════
X, y, fn = load_dataset(dataset_name, n_samples, noise, seed)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=test_pct/100, random_state=seed, stratify=y,
)

clf = DecisionTreeClassifier(
    max_depth=max_depth, min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf, max_features=max_features_val,
    criterion=criterion, splitter=splitter,
    class_weight=class_weight_val, random_state=42,
).fit(X_tr, y_tr)

tr_acc = accuracy_score(y_tr, clf.predict(X_tr)) * 100
te_acc = accuracy_score(y_te, clf.predict(X_te)) * 100
gap    = tr_acc - te_acc
depth  = clf.get_depth()
leaves = clf.get_n_leaves()
nodes  = clf.tree_.node_count

# model-state badge
if max_depth <= 1:
    _badge_txt, _bg, _fg = "Decision Stump",   BLUE_BG,   BLUE
elif tr_acc > 97 and gap > 15:
    _badge_txt, _bg, _fg = "Overfitting",       ROSE_BG,   ROSE
elif te_acc < 68:
    _badge_txt, _bg, _fg = "Underfitting",      AMBER_BG,  AMBER
elif gap < 5 and te_acc > 85:
    _badge_txt, _bg, _fg = "Well Generalizing", GREEN_BG,  GREEN
elif gap > 8:
    _badge_txt, _bg, _fg = "Slight Overfit",    AMBER_BG,  AMBER
else:
    _badge_txt, _bg, _fg = "Balanced",          BLUE_BG,   BLUE


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
hcol, bcol = st.columns([5, 1])
with hcol:
    st.markdown(
        f"## 🌳 Decision Tree Hyperparameter Explorer\n"
        f"<span style='color:{SLATE};font-size:14px'>"
        f"**{dataset_name}** &nbsp;·&nbsp; {len(X_tr)} train / {len(X_te)} test "
        f"&nbsp;·&nbsp; Features: <code>{fn[0]}</code> & <code>{fn[1]}</code>"
        f"</span>",
        unsafe_allow_html=True,
    )
with bcol:
    st.markdown(
        f"<div style='margin-top:22px;text-align:right'>{badge(_badge_txt,_bg,_fg)}</div>",
        unsafe_allow_html=True,
    )

# ── Metrics row ───────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Train Accuracy", f"{tr_acc:.1f}%")
c2.metric("Test Accuracy",  f"{te_acc:.1f}%", delta=f"gap {gap:+.1f}%")
c3.metric("Tree Depth",     str(depth))
c4.metric("Total Nodes",    str(nodes))
c5.metric("Leaf Nodes",     str(leaves))
c6.metric("Train Samples",  str(len(X_tr)))
c7.metric("Test Samples",   str(len(X_te)))

# ── Inline alerts ─────────────────────────────────────────────
if gap > 15:
    st.warning(
        f"⚠️ **Overfitting detected** — train accuracy is {gap:.1f}% above test. "
        "Try raising `min_samples_split` / `min_samples_leaf`, or lowering `max_depth`."
    )
elif te_acc < 68:
    st.info(
        "ℹ️ **Underfitting** — the model is too simple to capture the pattern. "
        "Try increasing `max_depth` or reducing `min_samples_split`."
    )

st.divider()


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️  Boundary",
    "🌳  Tree",
    "📊  Depth vs Acc",
    "🍃  Leaves",
    "📈  Importance",
    "📋  Report",
])

# ── Tab 1 ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### Decision Boundary")
    st.caption(
        "Coloured regions show the predicted class for every point in feature space. "
        "Decision trees can only cut perpendicular to axes — all boundaries are "
        "horizontal or vertical lines."
    )
    st.pyplot(fig_boundary(clf, X_tr, X_te, y_tr, y_te, fn),
              use_container_width=True)

    st.divider()
    lc, rc = st.columns(2)
    with lc:
        card("What to look for",
             "📐 All splits are axis-aligned<br>"
             "🏝️ Tiny islands → overfitting noise<br>"
             "🌊 Smooth large regions → good generalisation<br>"
             "❌ Dots in the wrong colour → misclassified",
             BLUE, BLUE_BG)
    with rc:
        if max_depth <= 2:
            card("Your current settings", "Shallow tree → coarse regions. "
                 "The boundary cannot follow the true data shape.", AMBER, AMBER_BG)
        elif gap > 12:
            card("Your current settings", "Many small jagged regions → "
                 "memorising training noise. Won't generalise.", ROSE, ROSE_BG)
        elif min_samples_leaf >= 8:
            card("Your current settings", "Large min-leaf → smooth, robust boundaries. "
                 "Good regularisation.", GREEN, GREEN_BG)
        else:
            card("Your current settings", "Looks well-balanced. Try pushing "
                 "max_depth higher to watch the boundary fragment.", TEAL, TEAL_BG)


# ── Tab 2 ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Tree Structure")
    st.caption(
        "Each internal node shows: **feature ≤ threshold · Gini impurity · sample count**. "
        "Leaves show the predicted class. Left branch = condition True, right = False."
    )
    max_show = st.slider(
        "Show top N levels", 1, min(10, depth+1), min(4, depth+1),
        key="tree_lvl",
    )
    st.pyplot(fig_tree(clf, fn, max_show), use_container_width=True)

    if depth > max_show:
        st.caption(
            f"Full tree has **{depth+1}** levels — use the slider above to reveal more."
        )
    with st.expander("📜 View raw text decision rules"):
        st.code(export_text(clf, feature_names=fn, max_depth=6), language="text")


# ── Tab 3 ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### Bias–Variance Tradeoff")
    st.caption(
        "We sweep `max_depth` 1 → 15 while keeping all other settings fixed. "
        "The **red dashed line** marks your current depth. "
        "The shaded gap between the two curves is your overfit indicator."
    )
    st.pyplot(fig_depth_acc(X_tr, y_tr, X_te, y_te, max_depth, criterion),
              use_container_width=True)

    st.divider()
    ca, cb, cc = st.columns(3)
    with ca:
        card("Left side — Underfitting",
             "Both curves are low. High bias. The model is too simple.",
             BLUE, BLUE_BG)
    with cb:
        card("Sweet Spot",
             "Train and test are high and close together. Best generalisation.",
             GREEN, GREEN_BG)
    with cc:
        card("Right side — Overfitting",
             "Train → 100% but test starts to drop. Tree memorises noise.",
             ROSE, ROSE_BG)


# ── Tab 4 ─────────────────────────────────────────────────────
with tab4:
    st.markdown("### Leaf Node Analysis")
    st.caption(
        "Pure leaves (Gini = 0) mean the tree has perfectly separated those training points. "
        "Many pure leaves + a large train-test gap = overfitting."
    )
    st.pyplot(fig_leaves(clf), use_container_width=True)

    st.divider()
    st.markdown("**Leaf statistics table**")
    tree_ = clf.tree_
    leaf_ids = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1]
    rows = []
    for lid in leaf_ids:
        n   = tree_.n_node_samples[lid]
        imp = tree_.impurity[lid]
        val = tree_.value[lid][0]
        cls = int(np.argmax(val))
        conf = val[cls] / val.sum() * 100
        rows.append({"Leaf ID": lid, "Samples": n,
                     "Gini Impurity": round(imp, 4),
                     "Predicted Class": cls,
                     "Confidence %": round(conf, 1)})

    df_lv = pd.DataFrame(rows).sort_values("Samples", ascending=False)
    st.dataframe(
        df_lv.style
            .background_gradient(subset=["Gini Impurity"], cmap="YlOrRd")
            .background_gradient(subset=["Confidence %"],  cmap="Greens"),
        use_container_width=True, height=270,
    )
    pure_pct   = (df_lv["Gini Impurity"] < 0.01).mean() * 100
    avg_leaf_n = df_lv["Samples"].mean()
    mA, mB = st.columns(2)
    mA.metric("Pure leaves (Gini < 0.01)", f"{pure_pct:.0f}%")
    mB.metric("Avg samples per leaf",       f"{avg_leaf_n:.1f}")

    if pure_pct > 80 and gap > 10:
        st.error("🔴 Most leaves pure + large train-test gap → classic overfitting.")
    elif pure_pct < 30:
        st.info("ℹ️ Many impure leaves → tree stopped early. May be underfitting.")


# ── Tab 5 ─────────────────────────────────────────────────────
with tab5:
    st.markdown("### Feature Importances")
    st.caption(
        "Gini importance = total impurity reduction contributed by a feature across all "
        "splits. The highlighted bar is the most useful feature for this tree."
    )
    st.pyplot(fig_feat_imp(clf, fn), use_container_width=True)

    fi_df = pd.DataFrame({
        "Feature":   fn,
        "Importance": clf.feature_importances_,
        "Rank": pd.Series(clf.feature_importances_).rank(ascending=False).astype(int).values,
    }).sort_values("Importance", ascending=False)
    st.dataframe(
        fi_df.style.background_gradient(subset=["Importance"], cmap="Blues"),
        use_container_width=True,
    )


# ── Tab 6 ─────────────────────────────────────────────────────
with tab6:
    st.markdown("### Full Classification Report")
    cl, cr = st.columns([1, 1])

    with cl:
        st.markdown("**Confusion Matrix — Test Set**")
        st.pyplot(fig_confusion(clf, X_te, y_te), use_container_width=True)

    with cr:
        st.markdown("**Precision / Recall / F1**")
        report = classification_report(
            y_te, clf.predict(X_te),
            target_names=["Class 0", "Class 1"],
            output_dict=True,
        )
        rdf = pd.DataFrame(report).T
        st.dataframe(
            rdf.style
                .format("{:.3f}", subset=["precision","recall","f1-score"])
                .background_gradient(subset=["f1-score"], cmap="Greens"),
            use_container_width=True,
        )

    st.divider()
    st.markdown("**Current config**")
    st.dataframe(pd.DataFrame([{
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "criterion": criterion,
        "splitter": splitter,
        "class_weight": class_weight,
    }]), use_container_width=True)

    st.markdown("**Equivalent scikit-learn code**")
    st.code(f"""\
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples={n_samples}, noise={noise:.2f}, random_state={seed})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_pct/100:.2f}, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(
    max_depth          = {max_depth},
    min_samples_split  = {min_samples_split},
    min_samples_leaf   = {min_samples_leaf},
    max_features       = {repr(max_features_val)},
    criterion          = "{criterion}",
    splitter           = "{splitter}",
    class_weight       = {repr(class_weight_val)},
    random_state       = 42,
)
clf.fit(X_train, y_train)
print(f"Train: {{accuracy_score(y_train, clf.predict(X_train)):.3f}}")
print(f"Test:  {{accuracy_score(y_test,  clf.predict(X_test)):.3f}}")
""", language="python")