"""
Decision Tree Hyperparameter Explorer
======================================
Run:   streamlit run app.py
Deps:  pip install -r requirements.txt
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

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Tree Explorer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  THEME DICT  — single source of truth for every colour
#  Light and dark variants defined here once.
#  Everything downstream reads from T = THEMES[mode].
# ─────────────────────────────────────────────────────────────
THEMES = {
    "Light": {
        # ── backgrounds ──
        "fig_bg":    "#FFFFFF",
        "ax_bg":     "#F8FAFC",
        "card_bg":   "#FFFFFF",
        "border":    "#E2E8F0",
        # ── text ──
        "text":      "#0F172A",
        "muted":     "#64748B",
        "metric_val":"#0F172A",
        # ── data colours ──
        "c0":        "#E11D48",   # class 0 dots
        "c1":        "#0D9488",   # class 1 dots
        "c0_bg":     "#FFF1F2",   # boundary region 0
        "c1_bg":     "#F0FDFA",   # boundary region 1
        # ── accent lines ──
        "acc_train": "#2563EB",
        "acc_test":  "#0D9488",
        "acc_gap":   "#E11D48",
        "bar_hi":    "#2563EB",
        "bar_lo":    "#93C5FD",
        # ── info cards ──
        "info_border":  "#2563EB", "info_bg":   "#EFF6FF",
        "warn_border":  "#D97706", "warn_bg":   "#FFFBEB",
        "good_border":  "#16A34A", "good_bg":   "#F0FDF4",
        "bad_border":   "#E11D48", "bad_bg":    "#FFF1F2",
        "teal_border":  "#0D9488", "teal_bg":   "#F0FDFA",
        "violet_border":"#7C3AED", "violet_bg": "#F5F3FF",
        # ── grid / legend ──
        "grid":      "#E2E8F0",
        "legend_fc": "#FFFFFF",
        "legend_ec": "#E2E8F0",
        # ── metric CSS overrides ──
        "metric_card_bg":    "#FFFFFF",
        "metric_card_border":"#E2E8F0",
        "metric_label_color":"#94A3B8",
    },
    "Dark": {
        "fig_bg":    "#1E1816",
        "ax_bg":     "#141210",
        "card_bg":   "#1E1816",
        "border":    "#2E2926",
        "text":      "#EDE6DC",
        "muted":     "#8B7D70",
        "metric_val":"#EDE6DC",
        "c0":        "#F87171",
        "c1":        "#34D399",
        "c0_bg":     "#2D0A0A",
        "c1_bg":     "#0A1F18",
        "acc_train": "#60A5FA",
        "acc_test":  "#34D399",
        "acc_gap":   "#F87171",
        "bar_hi":    "#E8845F",
        "bar_lo":    "#7C3A28",
        "info_border":  "#60A5FA", "info_bg":   "#0C1A2E",
        "warn_border":  "#FBBF24", "warn_bg":   "#1C1400",
        "good_border":  "#34D399", "good_bg":   "#061A10",
        "bad_border":   "#F87171", "bad_bg":    "#1C0505",
        "teal_border":  "#2DD4BF", "teal_bg":   "#061818",
        "violet_border":"#C084FC", "violet_bg": "#160A24",
        "grid":      "#2E2926",
        "legend_fc": "#1E1816",
        "legend_ec": "#2E2926",
        "metric_card_bg":    "#1E1816",
        "metric_card_border":"#2E2926",
        "metric_label_color":"#8B7D70",
    },
}

# ─────────────────────────────────────────────────────────────
#  MODE TOGGLE  — top of sidebar, before anything else
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    mode = st.radio("🎨 Theme", ["Light", "Dark"],
                    horizontal=True, label_visibility="collapsed")

T = THEMES[mode]

# ─────────────────────────────────────────────────────────────
#  CSS  — only structural; colours come from T at runtime
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
#MainMenu, header, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.4rem !important; max-width: 1400px !important; }}

[data-testid="stMetric"] {{
    background:    {T['metric_card_bg']} !important;
    border:        1px solid {T['metric_card_border']} !important;
    border-radius: 12px;
    padding:       16px 20px 12px 20px;
    box-shadow:    0 1px 4px rgba(0,0,0,0.08);
}}
[data-testid="stMetricLabel"] > div {{
    font-size:      11px   !important;
    font-weight:    700    !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: {T['metric_label_color']} !important;
}}
[data-testid="stMetricValue"] > div {{
    font-size:   1.5rem !important;
    font-weight: 700    !important;
    color: {T['metric_val']} !important;
}}
[data-testid="stMetricDelta"] > div {{ font-size: 12px !important; }}
[data-testid="stSidebar"] {{ min-width: 270px !important; max-width: 270px !important; }}
hr {{ border-color: {T['border']} !important; margin: 0.7rem 0 !important; }}
[data-baseweb="tab-list"] {{ gap: 2px !important; }}
[data-baseweb="tab"] {{ padding: 8px 18px !important; border-radius: 8px !important; }}
[data-testid="stExpander"] {{
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
}}
.stButton > button {{
    border-radius: 8px    !important;
    font-weight:   600    !important;
    font-size:     13px   !important;
    padding:       6px 14px !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB DEFAULTS  — set from T every run
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  T["fig_bg"],
    "axes.facecolor":    T["ax_bg"],
    "axes.edgecolor":    T["border"],
    "axes.labelcolor":   T["muted"],
    "axes.titlecolor":   T["text"],
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.grid":         True,
    "grid.color":        T["grid"],
    "grid.linewidth":    0.7,
    "xtick.color":       T["muted"],
    "ytick.color":       T["muted"],
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.facecolor":  T["legend_fc"],
    "legend.edgecolor":  T["legend_ec"],
    "legend.framealpha": 0.95,
    "legend.fontsize":   9,
    "font.size":         9,
    "text.color":        T["text"],
})


# ─────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────
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
            pts.extend(zip(rng.normal(cx, .55+noise*2.5, n),
                           rng.normal(cy, .55+noise*2.5, n), [c]*n))
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
    else:
        X, y = make_classification(
            n_samples=n_samples, n_features=2,
            n_informative=2, n_redundant=0,
            n_clusters_per_class=2,
            class_sep=max(0.1, 1.2-noise*3),
            random_state=seed,
        )
        fn = ["Feature A", "Feature B"]
    return X, y, fn


# ─────────────────────────────────────────────────────────────
#  PLOT HELPERS  — all use T, no hardcoded colours
# ─────────────────────────────────────────────────────────────
def new_fig(w, h, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor(T["fig_bg"])
    return fig, axes


def style_ax(ax):
    ax.set_facecolor(T["ax_bg"])
    for sp in ax.spines.values():
        sp.set_edgecolor(T["border"])
    ax.tick_params(colors=T["muted"], labelsize=8)


def fig_boundary(clf, X_tr, X_te, y_tr, y_te, fn):
    fig, axes = new_fig(w=13, h=5, ncols=2)
    cmap_bg = ListedColormap([T["c0_bg"], T["c1_bg"]])
    for ax, (Xs, ys), title in zip(
        axes,
        [(X_tr, y_tr), (X_te, y_te)],
        ["Training set", "Test set"],
    ):
        style_ax(ax)
        try:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.65,
                response_method="predict_proba",
                plot_method="pcolormesh", cmap=cmap_bg,
            )
        except Exception:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.65,
                response_method="predict",
                plot_method="pcolormesh", cmap=cmap_bg,
            )
        dot_c = [T["c0"] if v == 0 else T["c1"] for v in ys]
        ax.scatter(Xs[:, 0], Xs[:, 1], c=dot_c, s=24,
                   edgecolors=T["fig_bg"], linewidths=0.7, zorder=5)
        ax.legend(handles=[
            mpatches.Patch(color=T["c1"], label="Class 1"),
            mpatches.Patch(color=T["c0"], label="Class 0"),
        ], framealpha=0.9)
        ax.set_title(title, pad=10)
        ax.set_xlabel(fn[0])
        ax.set_ylabel(fn[1])
    fig.tight_layout(pad=2.5)
    return fig


def fig_tree(clf, fn, max_show):
    d = min(clf.get_depth(), max_show)
    w = max(10, 5*(2**d))
    h = max(5, 2.8*(d+1))
    fig, ax = plt.subplots(figsize=(min(w, 28), h))
    fig.patch.set_facecolor(T["fig_bg"])
    ax.set_facecolor(T["fig_bg"])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    plot_tree(clf, feature_names=fn, class_names=["Class 0","Class 1"],
              filled=True, rounded=True, impurity=True, proportion=False,
              max_depth=max_show, ax=ax, fontsize=8, precision=3)
    ax.set_title(
        f"Tree structure — top {d+1} of {clf.get_depth()+1} levels"
        if clf.get_depth() > max_show else "Tree structure", pad=8,
    )
    fig.tight_layout()
    return fig


def fig_depth_acc(X_tr, y_tr, X_te, y_te, cur, criterion):
    depths = list(range(1, 16))
    tr_a, te_a = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=42)
        m.fit(X_tr, y_tr)
        tr_a.append(accuracy_score(y_tr, m.predict(X_tr)) * 100)
        te_a.append(accuracy_score(y_te, m.predict(X_te)) * 100)
    fig, ax = new_fig(w=9, h=4)
    style_ax(ax)
    ax.plot(depths, tr_a, color=T["acc_train"], lw=2.2, marker="o", ms=5, label="Train accuracy")
    ax.plot(depths, te_a, color=T["acc_test"],  lw=2.2, marker="o", ms=5, label="Test accuracy")
    ax.fill_between(depths, tr_a, te_a,
                    where=[t > v for t, v in zip(tr_a, te_a)],
                    alpha=0.12, color=T["acc_gap"], label="Overfit gap")
    ax.axvline(x=cur, color=T["acc_gap"], lw=1.8, linestyle="--",
               label=f"Current depth = {cur}")
    ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Bias–Variance Tradeoff — Depth vs Accuracy")
    ax.set_ylim(30, 103); ax.set_xlim(0.5, 15.5)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_feat_imp(clf, fn):
    fi = clf.feature_importances_
    fig, ax = new_fig(w=7, h=max(3, len(fn)*0.6+1.5))
    style_ax(ax)
    colors = [T["bar_hi"] if v == max(fi) else T["bar_lo"] for v in fi]
    bars = ax.barh(fn, fi, color=colors, edgecolor=T["fig_bg"], height=0.5)
    for bar, val in zip(bars, fi):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color=T["muted"], fontsize=9)
    ax.set_xlim(0, max(fi)*1.35)
    ax.set_title("Feature Importances (Gini-based)")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def fig_confusion(clf, X_te, y_te):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.patch.set_facecolor(T["fig_bg"])
    ax.set_facecolor(T["fig_bg"])
    cmap = "YlOrBr" if mode == "Dark" else "Blues"
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, clf.predict(X_te)),
        display_labels=["Class 0","Class 1"],
    ).plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title("Confusion Matrix — Test set")
    for txt in ax.texts:
        txt.set_color(T["text"]); txt.set_fontsize(13); txt.set_fontweight("bold")
    for sp in ax.spines.values(): sp.set_edgecolor(T["border"])
    fig.tight_layout()
    return fig


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
    fig, axes = new_fig(w=10, h=4, ncols=2)
    for ax, data, title, color in [
        (axes[0], s_list, "Samples per Leaf",        T["acc_train"]),
        (axes[1], g_list, "Gini Impurity per Leaf",  T["violet_border"]),
    ]:
        style_ax(ax)
        ax.hist(data, bins=min(30, max(5, len(data)//3)),
                color=color, alpha=0.78, edgecolor=T["fig_bg"], linewidth=0.6)
        ax.axvline(np.mean(data), color=T["acc_gap"], lw=1.8, linestyle="--",
                   label=f"mean = {np.mean(data):.2f}")
        ax.set_title(title); ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.legend()
    fig.tight_layout(pad=2.5)
    return fig


# ─────────────────────────────────────────────────────────────
#  UI HELPERS  — use T colours
# ─────────────────────────────────────────────────────────────
def card(title, body, style="info"):
    """style: info | warn | good | bad | teal | violet"""
    bc = T[f"{style}_border"]
    bg = T[f"{style}_bg"]
    st.markdown(
        f"<div style='background:{bg};border-left:4px solid {bc};"
        f"border-radius:10px;padding:14px 16px;margin-bottom:4px'>"
        f"<div style='font-weight:700;font-size:13px;color:{T['text']};margin-bottom:4px'>{title}</div>"
        f"<div style='font-size:12.5px;color:{T['muted']};line-height:1.65'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def badge(text, style="info"):
    bc = T[f"{style}_border"]
    bg = T[f"{style}_bg"]
    return (f"<span style='background:{bg};color:{bc};padding:4px 14px;"
            f"border-radius:20px;font-size:12px;font-weight:700'>{text}</span>")


# ─────────────────────────────────────────────────────────────
#  SIDEBAR  — controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌳 Decision Tree\n**Hyperparameter Explorer**")
    st.caption("Adjust any slider — every chart updates instantly.")
    st.divider()

    st.markdown("**📦 Dataset**")
    dataset_name = st.selectbox("Dataset",
        ["Two Moons","Two Circles","XOR Blobs","Linear Separable","Breast Cancer"],
        label_visibility="collapsed")
    n_samples = st.slider("Samples", 200, 2000, 600, step=100)
    noise     = st.slider("Noise level", 0.00, 0.50, 0.20, step=0.05)
    test_pct  = st.slider("Test split %", 10, 40, 25, step=5)
    seed      = st.slider("Random seed", 0, 99, 42)
    st.divider()

    st.markdown("**⚙️ Hyperparameters**")
    max_depth = st.slider("max_depth", 1, 15, 4,
        help="Max levels the tree can grow. Higher = more complex = risk of overfitting.")
    min_samples_split = st.slider("min_samples_split", 2, 60, 2,
        help="A node only splits when it holds ≥ this many samples.")
    min_samples_leaf = st.slider("min_samples_leaf", 1, 40, 1,
        help="Every leaf must contain ≥ this many samples. Higher = smoother boundaries.")
    max_features = st.selectbox("max_features", ["all features","sqrt","log2"])
    criterion    = st.selectbox("criterion",    ["gini","entropy","log_loss"])
    splitter     = st.selectbox("splitter",     ["best","random"])
    class_weight = st.selectbox("class_weight", ["None","balanced"])
    st.divider()

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

max_features_val = None if max_features == "all features" else max_features
class_weight_val = None if class_weight == "None"         else "balanced"


# ─────────────────────────────────────────────────────────────
#  FIT MODEL
# ─────────────────────────────────────────────────────────────
X, y, fn = load_dataset(dataset_name, n_samples, noise, seed)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=test_pct/100, random_state=seed, stratify=y)

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

# ── model-state badge (style keys must exist in T) ──
if max_depth <= 1:
    _badge_txt, _badge_style = "Decision Stump",    "info"
elif tr_acc > 97 and gap > 15:
    _badge_txt, _badge_style = "Overfitting",        "bad"
elif te_acc < 68:
    _badge_txt, _badge_style = "Underfitting",       "warn"
elif gap < 5 and te_acc > 85:
    _badge_txt, _badge_style = "Well Generalizing",  "good"
elif gap > 8:
    _badge_txt, _badge_style = "Slight Overfit",     "warn"
else:
    _badge_txt, _badge_style = "Balanced",           "good"


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
hcol, bcol = st.columns([5, 1])
with hcol:
    st.markdown(
        f"## 🌳 Decision Tree Hyperparameter Explorer\n"
        f"<span style='color:{T['muted']};font-size:14px'>"
        f"**{dataset_name}** &nbsp;·&nbsp; {len(X_tr)} train / {len(X_te)} test "
        f"&nbsp;·&nbsp; <code style='background:{T['card_bg']};color:{T['info_border']};padding:2px 6px;border-radius:4px'>"
        f"{fn[0]}</code> & <code style='background:{T['card_bg']};color:{T['info_border']};padding:2px 6px;border-radius:4px'>"
        f"{fn[1]}</code></span>",
        unsafe_allow_html=True,
    )
with bcol:
    st.markdown(
        f"<div style='margin-top:22px;text-align:right'>{badge(_badge_txt, _badge_style)}</div>",
        unsafe_allow_html=True,
    )

# ── Metrics ──────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Train Accuracy", f"{tr_acc:.1f}%")
c2.metric("Test Accuracy",  f"{te_acc:.1f}%", delta=f"gap {gap:+.1f}%")
c3.metric("Tree Depth",     str(depth))
c4.metric("Total Nodes",    str(nodes))
c5.metric("Leaf Nodes",     str(leaves))
c6.metric("Train Samples",  str(len(X_tr)))
c7.metric("Test Samples",   str(len(X_te)))

if gap > 15:
    st.warning(f"⚠️ **Overfitting** — train is {gap:.1f}% above test. "
               "Raise `min_samples_split` / `min_samples_leaf` or lower `max_depth`.")
elif te_acc < 68:
    st.info("ℹ️ **Underfitting** — too simple. Increase `max_depth` or lower `min_samples_split`.")

st.divider()


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️  Boundary", "🌳  Tree", "📊  Depth vs Acc",
    "🍃  Leaves",   "📈  Importance", "📋  Report",
])

# ── Tab 1 ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### Decision Boundary")
    st.caption("Coloured regions = predicted class for every point. "
               "All splits are axis-aligned (horizontal or vertical lines only).")
    st.pyplot(fig_boundary(clf, X_tr, X_te, y_tr, y_te, fn), use_container_width=True)
    st.divider()
    lc, rc = st.columns(2)
    with lc:
        card("What to look for",
             "📐 All splits are axis-aligned<br>"
             "🏝️ Tiny islands → overfitting noise<br>"
             "🌊 Large smooth regions → good generalisation<br>"
             "❌ Dot in wrong-colour region = misclassified", "info")
    with rc:
        if max_depth <= 2:
            card("Your settings", "Shallow tree → coarse regions. Can't follow the true data shape.", "warn")
        elif gap > 12:
            card("Your settings", "Jagged regions → memorising training noise. Won't generalise.", "bad")
        elif min_samples_leaf >= 8:
            card("Your settings", "Large min-leaf → smooth, robust boundaries. Good regularisation.", "good")
        else:
            card("Your settings", "Balanced. Try pushing max_depth higher to watch it fragment.", "teal")

# ── Tab 2 ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Tree Structure")
    st.caption("Internal node: **feature ≤ threshold · Gini · samples**. "
               "Leaf: predicted class. Left = True, Right = False.")
    max_show = st.slider("Show top N levels", 1, min(10, depth+1), min(4, depth+1), key="tlvl")
    st.pyplot(fig_tree(clf, fn, max_show), use_container_width=True)
    if depth > max_show:
        st.caption(f"Full tree has **{depth+1}** levels — slide above to reveal more.")
    with st.expander("📜 Raw text decision rules"):
        st.code(export_text(clf, feature_names=fn, max_depth=6), language="text")

# ── Tab 3 ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### Bias–Variance Tradeoff")
    st.caption("Sweeps max_depth 1 → 15. Red dashed line = your current depth. "
               "Shaded gap = overfit indicator.")
    st.pyplot(fig_depth_acc(X_tr, y_tr, X_te, y_te, max_depth, criterion), use_container_width=True)
    st.divider()
    ca, cb, cc = st.columns(3)
    with ca: card("Left — Underfitting", "High bias. Both curves low. Model too simple.", "info")
    with cb: card("Sweet Spot", "Both curves high and close. Best generalisation.", "good")
    with cc: card("Right — Overfitting", "Train → 100%, test drops. Tree memorises noise.", "bad")

# ── Tab 4 ─────────────────────────────────────────────────────
with tab4:
    st.markdown("### Leaf Analysis")
    st.caption("Pure leaves (Gini = 0) = tree perfectly separated those training points.")
    st.pyplot(fig_leaves(clf), use_container_width=True)
    st.divider()
    tree_ = clf.tree_
    leaf_ids = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1]
    rows = []
    for lid in leaf_ids:
        n, imp = tree_.n_node_samples[lid], tree_.impurity[lid]
        val = tree_.value[lid][0]
        cls = int(np.argmax(val))
        rows.append({"Leaf ID": lid, "Samples": n,
                     "Gini Impurity": round(imp, 4),
                     "Predicted Class": cls,
                     "Confidence %": round(val[cls]/val.sum()*100, 1)})
    df_lv = pd.DataFrame(rows).sort_values("Samples", ascending=False)
    st.dataframe(
        df_lv.style
            .background_gradient(subset=["Gini Impurity"], cmap="YlOrRd")
            .background_gradient(subset=["Confidence %"],  cmap="Greens"),
        use_container_width=True, height=270)
    pure_pct = (df_lv["Gini Impurity"] < 0.01).mean() * 100
    mA, mB = st.columns(2)
    mA.metric("Pure leaves (Gini < 0.01)", f"{pure_pct:.0f}%")
    mB.metric("Avg samples per leaf",       f"{df_lv['Samples'].mean():.1f}")
    if pure_pct > 80 and gap > 10:
        st.error("🔴 Most leaves pure + large gap → classic overfitting.")
    elif pure_pct < 30:
        st.info("ℹ️ Many impure leaves → tree stopped early. May be underfitting.")

# ── Tab 5 ─────────────────────────────────────────────────────
with tab5:
    st.markdown("### Feature Importances")
    st.caption("Total Gini impurity reduction contributed by each feature across all splits.")
    st.pyplot(fig_feat_imp(clf, fn), use_container_width=True)
    fi_df = pd.DataFrame({
        "Feature": fn, "Importance": clf.feature_importances_,
        "Rank": pd.Series(clf.feature_importances_).rank(ascending=False).astype(int).values,
    }).sort_values("Importance", ascending=False)
    st.dataframe(fi_df.style.background_gradient(subset=["Importance"], cmap="Blues"),
                 use_container_width=True)

# ── Tab 6 ─────────────────────────────────────────────────────
with tab6:
    st.markdown("### Full Classification Report")
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Confusion Matrix — Test Set**")
        st.pyplot(fig_confusion(clf, X_te, y_te), use_container_width=True)
    with cr:
        st.markdown("**Precision / Recall / F1**")
        rdf = pd.DataFrame(classification_report(
            y_te, clf.predict(X_te),
            target_names=["Class 0","Class 1"], output_dict=True)).T
        st.dataframe(
            rdf.style
                .format("{:.3f}", subset=["precision","recall","f1-score"])
                .background_gradient(subset=["f1-score"], cmap="Greens"),
            use_container_width=True)
    st.divider()
    st.markdown("**Current config**")
    st.dataframe(pd.DataFrame([{
        "max_depth": max_depth, "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf, "max_features": max_features,
        "criterion": criterion, "splitter": splitter, "class_weight": class_weight,
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