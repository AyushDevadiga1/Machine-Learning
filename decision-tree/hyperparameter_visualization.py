"""
Decision Tree Hyperparameter Explorer
--------------------------------------
Folder layout:
    app.py
    requirements.txt
    assets/style.css
    .streamlit/config.toml

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from matplotlib.colors import ListedColormap
from sklearn.datasets import (
    make_moons, make_circles, make_classification, load_breast_cancer
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Tree Explorer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme tokens ─────────────────────────────────────────────
# Change values here to retheme the entire app.
# These are injected into CSS :root, so style.css never needs editing.
THEME = {
    "--bg-main":      "#ffffff",
    "--bg-sidebar":   "#f8f9fa",
    "--text-main":    "#202124",
    "--text-muted":   "#5f6368",
    "--border":       "#dadce0",
    "--accent":       "#1a73e8",
    "--accent-light": "#e8f0fe",
}

def _load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    css = css_path.read_text()
    # Inject theme variables into :root
    vars_block = ":root {\n" + "\n".join(
        f"    {k}: {v};" for k, v in THEME.items()
    ) + "\n}"
    st.markdown(f"<style>{vars_block}\n{css}</style>", unsafe_allow_html=True)

_load_css()


# ─────────────────────────────────────────────────────────────────────────────
# CHART STYLE  (set once, inherited by every figure)
# ─────────────────────────────────────────────────────────────────────────────
BLUE  = "#2563EB"
RED   = "#DC2626"
GRAY  = "#6B7280"
LGRAY = "#E5E7EB"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    LGRAY,
    "axes.labelcolor":   GRAY,
    "axes.titlecolor":   "#111111",
    "axes.titlesize":    10,
    "axes.titleweight":  "semibold",
    "axes.labelsize":    8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#F3F4F6",
    "grid.linewidth":    0.8,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "legend.frameon":    True,
    "legend.framealpha": 1,
    "legend.edgecolor":  LGRAY,
    "legend.fontsize":   8,
    "font.family":       "sans-serif",
    "font.size":         9,
    "text.color":        "#111111",
    "savefig.bbox":      "tight",
    "savefig.dpi":       150,
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (all cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset(name: str, n: int, noise: float, seed: int):
    """Return X, y, feature_names for the chosen dataset."""
    rng = np.random.default_rng(seed)

    if name == "Moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
        return X, y, ["x1", "x2"]

    if name == "Circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.45, random_state=seed)
        return X, y, ["x1", "x2"]

    if name == "XOR":
        pts = []
        for cx, cy, cls in [(-1.5, 1.5, 0), (1.5, -1.5, 0),
                             (1.5,  1.5, 1), (-1.5, -1.5, 1)]:
            k = n // 4
            pts.append(np.column_stack([
                rng.normal(cx, 0.55 + noise * 2.5, k),
                rng.normal(cy, 0.55 + noise * 2.5, k),
                np.full(k, cls),
            ]))
        arr = np.vstack(pts)
        return arr[:, :2], arr[:, 2].astype(int), ["x1", "x2"]

    if name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data[:, :2], data.target
        idx  = rng.choice(len(y), min(n, len(y)), replace=False)
        return X[idx], y[idx], list(data.feature_names[:2])

    # Default — linearly separable
    X, y = make_classification(
        n_samples=n, n_features=2,
        n_informative=2, n_redundant=0,
        n_clusters_per_class=2,
        class_sep=max(0.1, 1.2 - noise * 3),
        random_state=seed,
    )
    return X, y, ["feature_a", "feature_b"]


@st.cache_data(show_spinner=False)
def fit_model(
    X_tr, y_tr,
    max_depth, min_split, min_leaf,
    max_feat, criterion, splitter, cw,
):
    """Fit and return a DecisionTreeClassifier (cached by its inputs)."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        max_features=max_feat,
        criterion=criterion,
        splitter=splitter,
        class_weight=cw,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    return clf


@st.cache_data(show_spinner=False)
def depth_sweep(_X_tr, _y_tr, _X_te, _y_te, criterion: str):
    """Compute train/test accuracy for max_depth 1–15 (cached)."""
    train_acc, test_acc = [], []
    for d in range(1, 16):
        m = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=42)
        m.fit(_X_tr, _y_tr)
        train_acc.append(accuracy_score(_y_tr, m.predict(_X_tr)) * 100)
        test_acc.append(accuracy_score(_y_te, m.predict(_X_te)) * 100)
    return train_acc, test_acc


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS  (pure — no Streamlit calls, no side effects)
# ─────────────────────────────────────────────────────────────────────────────
_C0, _C1         = RED,  BLUE          # scatter dot colours
_C0_BG, _C1_BG  = "#FEF2F2", "#EFF6FF" # boundary fill colours


def _style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)


def chart_boundary(clf, X_tr, X_te, y_tr, y_te, feat_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap_bg = ListedColormap([_C0_BG, _C1_BG])

    for ax, (X_s, y_s), title in zip(
        axes,
        [(X_tr, y_tr), (X_te, y_te)],
        ["Training set", "Test set"],
    ):
        _style_ax(ax)
        try:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.55,
                response_method="predict_proba",
                plot_method="pcolormesh", cmap=cmap_bg,
            )
        except Exception:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.55,
                response_method="predict",
                plot_method="pcolormesh", cmap=cmap_bg,
            )
        colors = [_C0 if v == 0 else _C1 for v in y_s]
        ax.scatter(X_s[:, 0], X_s[:, 1], c=colors, s=20,
                   edgecolors="white", linewidths=0.5, zorder=5)
        ax.legend(
            handles=[
                mpatches.Patch(facecolor=_C1, label="Class 1"),
                mpatches.Patch(facecolor=_C0, label="Class 0"),
            ],
            framealpha=1, loc="upper right",
        )
        ax.set_title(title)
        ax.set_xlabel(feat_names[0])
        ax.set_ylabel(feat_names[1])

    fig.tight_layout(pad=2.0)
    return fig


def chart_tree(clf, feat_names, max_show):
    depth = min(clf.get_depth(), max_show)
    w = min(max(8, 4 * (2 ** depth)), 26)
    h = max(4, 2.4 * (depth + 1))
    fig, ax = plt.subplots(figsize=(w, h))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plot_tree(
        clf,
        feature_names=feat_names,
        class_names=["Class 0", "Class 1"],
        filled=True, rounded=True,
        impurity=True, proportion=False,
        max_depth=max_show,
        ax=ax, fontsize=8, precision=3,
    )
    suffix = (f"  —  showing {depth + 1} of {clf.get_depth() + 1} levels"
              if clf.get_depth() > max_show else "")
    ax.set_title(f"Tree structure{suffix}")
    fig.tight_layout()
    return fig


def chart_depth_vs_acc(train_acc, test_acc, current_depth):
    depths = list(range(1, 16))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _style_ax(ax)
    ax.plot(depths, train_acc, color=BLUE, lw=1.8, marker="o", ms=4, label="Train")
    ax.plot(depths, test_acc,  color=GRAY, lw=1.8, marker="o", ms=4, label="Test")
    ax.fill_between(depths, train_acc, test_acc,
                    where=[t > v for t, v in zip(train_acc, test_acc)],
                    color=RED, alpha=0.07, label="Overfit gap")
    ax.axvline(current_depth, color=RED, lw=1.2, linestyle="--",
               label=f"Current  (depth = {current_depth})")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs max_depth")
    ax.set_ylim(25, 103)
    ax.set_xlim(0.5, 15.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def chart_feat_importance(clf, feat_names):
    fi = clf.feature_importances_
    order = np.argsort(fi)
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(feat_names) * 0.5 + 1)))
    _style_ax(ax)
    colors = [BLUE if i == order[-1] else LGRAY for i in range(len(fi))]
    bars = ax.barh(
        [feat_names[i] for i in order],
        fi[order],
        color=[colors[i] for i in order],
        edgecolor="none", height=0.5,
    )
    for bar, val in zip(bars, fi[order]):
        ax.text(
            val + max(fi) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            color=GRAY, fontsize=8,
        )
    ax.set_xlim(0, max(fi) * 1.3)
    ax.set_xlabel("Gini importance")
    ax.set_title("Feature importances")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def chart_confusion(clf, X_te, y_te):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    _style_ax(ax)
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, clf.predict(X_te)),
        display_labels=["Class 0", "Class 1"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix  (test set)")
    for txt in ax.texts:
        txt.set_fontsize(13)
        txt.set_fontweight("bold")
    fig.tight_layout()
    return fig


def chart_leaf_dist(clf):
    samples, impurities = [], []

    def _walk(node):
        left = clf.tree_.children_left[node]
        if left == -1:
            samples.append(clf.tree_.n_node_samples[node])
            impurities.append(clf.tree_.impurity[node])
        else:
            _walk(left)
            _walk(clf.tree_.children_right[node])

    _walk(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, data, xlabel, color in [
        (ax1, samples,     "Samples per leaf",     BLUE),
        (ax2, impurities,  "Gini impurity per leaf", GRAY),
    ]:
        _style_ax(ax)
        ax.hist(data, bins=min(30, max(5, len(data) // 3)),
                color=color, alpha=0.25, edgecolor=color, linewidth=0.7)
        ax.axvline(np.mean(data), color=RED, lw=1.3, linestyle="--",
                   label=f"Mean  {np.mean(data):.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend()
    fig.tight_layout(pad=2.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Decision Tree Explorer")
    st.write("Tune the parameters below and every chart updates live.")
    st.divider()

    # ── Dataset ──────────────────────────────────────────────
    st.markdown("**Dataset**")
    dataset_name = st.selectbox(
        "Dataset",
        ["Moons", "Circles", "XOR", "Linear", "Breast Cancer"],
        help="Moons and Circles are non-linearly separable. "
             "XOR has four clusters in a checkerboard pattern. "
             "Breast Cancer uses the first two features of the sklearn dataset.",
        label_visibility="collapsed",
    )
    n_samples = st.slider(
        "Number of samples", 200, 2000, 600, step=100,
        help="More samples = more stable accuracy estimates, slower training.",
    )
    noise = st.slider(
        "Noise", 0.00, 0.50, 0.20, step=0.05,
        help="How much random scatter to add to the data.",
    )
    col_ts, col_sd = st.columns(2)
    test_pct  = col_ts.slider("Test %",   10, 40,  25, step=5)
    rand_seed = col_sd.slider("Seed",      0, 99,  42)

    st.divider()

    # ── Model hyperparameters ─────────────────────────────────
    st.markdown("**Hyperparameters**")

    max_depth = st.slider(
        "max_depth", 1, 15, 4,
        help="Maximum depth the tree is allowed to grow. "
             "Shallow = simple / underfitting. Deep = complex / overfitting.",
    )
    min_samples_split = st.slider(
        "min_samples_split", 2, 60, 2,
        help="A node will only be split if it contains at least this many samples. "
             "Higher values prevent splits on tiny groups.",
    )
    min_samples_leaf = st.slider(
        "min_samples_leaf", 1, 40, 1,
        help="Every leaf must hold at least this many training samples. "
             "Higher values smooth the decision boundary.",
    )
    max_features = st.selectbox(
        "max_features",
        ["all", "sqrt", "log2"],
        help="Number of features to consider at each split. "
             "'all' considers every feature; 'sqrt'/'log2' add randomness.",
    )
    criterion = st.selectbox(
        "criterion",
        ["gini", "entropy", "log_loss"],
        help="The function used to measure the quality of a split.",
    )
    splitter = st.selectbox(
        "splitter",
        ["best", "random"],
        help="'best' always picks the globally optimal split. "
             "'random' samples split candidates, adding variance.",
    )
    class_weight = st.selectbox(
        "class_weight",
        ["None", "balanced"],
        help="'balanced' adjusts sample weights inversely to class frequency. "
             "Useful when one class heavily outnumbers the other.",
    )

    st.divider()

    # ── Quick presets ─────────────────────────────────────────
    st.markdown("**Presets**")
    col_a, col_b = st.columns(2)
    if col_a.button("Stump",    use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 1, 2, 1
    if col_b.button("Balanced", use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 4, 5, 3
    if col_a.button("Overfit",  use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 15, 2, 1
    if col_b.button("Pruned",   use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 5, 20, 8

# Resolve selectbox strings to sklearn values
_max_feat = None   if max_features == "all"  else max_features
_cw       = None   if class_weight == "None" else "balanced"


# ─────────────────────────────────────────────────────────────────────────────
# DATA  +  MODEL  (both cached)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    X, y, feat_names = get_dataset(dataset_name, n_samples, noise, rand_seed)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=test_pct / 100,
    random_state=rand_seed,
    stratify=y,
)

# Convert arrays to tuples so cache hash is stable
with st.spinner("Fitting model…"):
    clf = fit_model(
        X_tr, y_tr,
        max_depth, min_samples_split, min_samples_leaf,
        _max_feat, criterion, splitter, _cw,
    )

# ── Derived stats ─────────────────────────────────────────────
train_acc = accuracy_score(y_tr, clf.predict(X_tr)) * 100
test_acc  = accuracy_score(y_te, clf.predict(X_te)) * 100
gap       = train_acc - test_acc
tree_depth   = clf.get_depth()
n_leaves     = clf.get_n_leaves()
n_nodes      = clf.tree_.node_count


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Decision Tree Explorer")
st.caption(
    f"Dataset: **{dataset_name}**  ·  "
    f"{len(X_tr):,} training samples  ·  "
    f"{len(X_te):,} test samples  ·  "
    f"Features: `{feat_names[0]}`, `{feat_names[1]}`"
)
st.divider()

# ── Key metrics ───────────────────────────────────────────────
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Train accuracy",  f"{train_acc:.1f}%")
col2.metric("Test accuracy",   f"{test_acc:.1f}%",
            delta=f"{gap:+.1f}% gap",
            delta_color="inverse" if gap > 10 else "normal")
col3.metric("Tree depth",      str(tree_depth))
col4.metric("Total nodes",     str(n_nodes))
col5.metric("Leaf nodes",      str(n_leaves))
col6.metric("Train samples",   f"{len(X_tr):,}")
col7.metric("Test samples",    f"{len(X_te):,}")

# ── Diagnostic banner (single, contextual) ───────────────────
if gap > 15 and train_acc > 95:
    st.error(
        f"**Overfitting.** Train accuracy is {gap:.1f} percentage points above test. "
        "The tree is memorising the training data. "
        "Try increasing `min_samples_split`, increasing `min_samples_leaf`, "
        "or reducing `max_depth`."
    )
elif test_acc < 68:
    st.warning(
        "**Underfitting.** Test accuracy is low — the model is too simple to capture the pattern. "
        "Try increasing `max_depth` or decreasing `min_samples_split`."
    )
elif gap < 3 and test_acc > 88:
    st.success(
        f"**Good generalisation.** Train and test accuracy are close ({gap:.1f}% gap), "
        "and test accuracy is strong."
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_boundary, tab_tree, tab_bias, tab_leaves, tab_importance, tab_report = st.tabs([
    "Decision boundary",
    "Tree structure",
    "Depth vs accuracy",
    "Leaf analysis",
    "Feature importance",
    "Full report",
])


# ── Decision boundary ─────────────────────────────────────────
with tab_boundary:
    st.write(
        "The coloured regions show the class the tree predicts for every point in "
        "feature space. Because decision trees split on a single feature at a time, "
        "every boundary is a straight horizontal or vertical line."
    )
    st.pyplot(chart_boundary(clf, X_tr, X_te, y_tr, y_te, feat_names),
              use_container_width=True)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**Reading the chart**")
        st.markdown(
            "- All boundaries are axis-aligned — this is a hard constraint of decision trees.\n"
            "- A fragmented boundary with many small islands usually signals overfitting.\n"
            "- A smooth boundary with large regions typically generalises better.\n"
            "- Any point plotted inside the wrong colour region is a misclassification."
        )
    with right:
        st.markdown("**What your current settings produce**")
        if max_depth <= 2:
            st.info(
                "Depth 1–2 creates very coarse regions. "
                "The boundary cannot follow the true shape of the data."
            )
        elif gap > 12:
            st.warning(
                "The boundary is fragmented. The tree is fitting noise in the "
                "training set and will not generalise."
            )
        elif min_samples_leaf >= 8:
            st.info(
                "A large `min_samples_leaf` forces larger leaf regions, "
                "producing a smoother boundary."
            )
        else:
            st.info(
                "The boundary looks balanced. "
                "Increase `max_depth` and watch it fragment; "
                "increase `min_samples_leaf` and watch it smooth out."
            )


# ── Tree structure ────────────────────────────────────────────
with tab_tree:
    st.write(
        "Each internal node shows the split condition (`feature ≤ threshold`), "
        "the Gini impurity, and the number of training samples that reach that node. "
        "Each leaf shows the predicted class and sample count. "
        "Left branch takes the True path; right branch takes the False path."
    )
    levels_to_show = st.slider(
        "Levels to display",
        min_value=1,
        max_value=min(10, tree_depth + 1),
        value=min(4, tree_depth + 1),
        key="tree_depth_slider",
    )
    st.pyplot(chart_tree(clf, feat_names, levels_to_show),
              use_container_width=True)
    if tree_depth > levels_to_show:
        st.caption(
            f"The full tree has {tree_depth + 1} levels. "
            "Use the slider above to reveal deeper levels."
        )
    with st.expander("View raw decision rules as text"):
        rules = export_text(clf, feature_names=feat_names, max_depth=8)
        st.code(rules, language="text")


# ── Depth vs accuracy ─────────────────────────────────────────
with tab_bias:
    st.write(
        "This chart trains a fresh tree for every value of `max_depth` from 1 to 15, "
        "keeping all other hyperparameters fixed. "
        "It shows the classic bias–variance tradeoff: "
        "shallow trees underfit (high bias), deep trees overfit (high variance). "
        "The vertical dashed line marks your current setting."
    )
    with st.spinner("Computing depth sweep…"):
        tr_sweep, te_sweep = depth_sweep(X_tr, y_tr, X_te, y_te, criterion)

    st.pyplot(chart_depth_vs_acc(tr_sweep, te_sweep, max_depth),
              use_container_width=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Left of the sweet spot**")
        st.markdown(
            "Both train and test accuracy are low. "
            "The model does not have enough capacity to learn the pattern. "
            "This is high bias / underfitting."
        )
    with c2:
        st.markdown("**Sweet spot**")
        st.markdown(
            "Test accuracy is highest and the gap between train and test is small. "
            "The model generalises well. "
            "The optimal `max_depth` is usually in this region."
        )
    with c3:
        st.markdown("**Right of the sweet spot**")
        st.markdown(
            "Train accuracy approaches 100% but test accuracy plateaus or drops. "
            "The gap widens. "
            "The model is memorising training data. "
            "This is high variance / overfitting."
        )


# ── Leaf analysis ─────────────────────────────────────────────
with tab_leaves:
    st.write(
        "These histograms show how many samples each leaf node holds "
        "and how pure each leaf is. A Gini impurity of 0 means a leaf contains "
        "only one class — the tree has perfectly separated those training samples. "
        "Many pure leaves with a large train–test gap is a sign of overfitting."
    )
    st.pyplot(chart_leaf_dist(clf), use_container_width=True)

    st.divider()
    tree_ = clf.tree_
    leaf_ids = [
        i for i in range(tree_.node_count)
        if tree_.children_left[i] == -1
    ]
    rows = []
    for lid in leaf_ids:
        n    = tree_.n_node_samples[lid]
        gini = tree_.impurity[lid]
        val  = tree_.value[lid][0]
        cls  = int(np.argmax(val))
        conf = val[cls] / val.sum() * 100
        rows.append({
            "Leaf node":      lid,
            "Samples":        n,
            "Gini impurity":  round(gini, 4),
            "Predicted class": cls,
            "Confidence (%)": round(conf, 1),
        })

    df_leaves = pd.DataFrame(rows).sort_values("Samples", ascending=False)

    st.markdown("**All leaf nodes**")
    st.dataframe(
        df_leaves.style
            .background_gradient(subset=["Gini impurity"], cmap="YlOrRd")
            .background_gradient(subset=["Confidence (%)"], cmap="Blues")
            .format({"Gini impurity": "{:.4f}", "Confidence (%)": "{:.1f}"}),
        use_container_width=True,
        height=280,
    )

    pure_pct = (df_leaves["Gini impurity"] < 0.01).mean() * 100
    avg_n    = df_leaves["Samples"].mean()
    m1, m2   = st.columns(2)
    m1.metric("Pure leaves (Gini < 0.01)", f"{pure_pct:.0f}%")
    m2.metric("Mean samples per leaf",      f"{avg_n:.1f}")


# ── Feature importance ────────────────────────────────────────
with tab_importance:
    st.write(
        "Gini-based feature importance measures the total reduction in node impurity "
        "a feature contributes across all splits in the tree, weighted by the proportion "
        "of training samples that reach each node. "
        "A higher value means the feature was more useful for separating the classes."
    )
    st.pyplot(chart_feat_importance(clf, feat_names), use_container_width=True)

    fi_df = pd.DataFrame({
        "Feature":    feat_names,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_df.index += 1
    fi_df.index.name = "Rank"
    st.dataframe(
        fi_df.style
            .format({"Importance": "{:.4f}"})
            .background_gradient(subset=["Importance"], cmap="Blues"),
        use_container_width=True,
    )
    st.caption(
        "Note: these importances are computed on the two visualisation features only. "
        "On a real dataset with many features, always inspect importances across all of them."
    )


# ── Full report ───────────────────────────────────────────────
with tab_report:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Confusion matrix**")
        st.pyplot(chart_confusion(clf, X_te, y_te), use_container_width=True)

    with col_right:
        st.markdown("**Classification report**")
        report_dict = classification_report(
            y_te, clf.predict(X_te),
            target_names=["Class 0", "Class 1"],
            output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(
            report_df.style
                .format("{:.3f}", subset=["precision", "recall", "f1-score"])
                .background_gradient(subset=["f1-score"], cmap="Blues"),
            use_container_width=True,
        )

    st.divider()
    st.markdown("**Current configuration**")
    config_df = pd.DataFrame([{
        "max_depth":          max_depth,
        "min_samples_split":  min_samples_split,
        "min_samples_leaf":   min_samples_leaf,
        "max_features":       max_features,
        "criterion":          criterion,
        "splitter":           splitter,
        "class_weight":       class_weight,
    }])
    st.dataframe(config_df, use_container_width=True)

    st.markdown("**Reproducible code**")
    st.code(
        f"""\
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples={n_samples}, noise={noise:.2f}, random_state={rand_seed})

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size={test_pct / 100:.2f},
    random_state={rand_seed},
    stratify=y,
)

clf = DecisionTreeClassifier(
    max_depth         = {max_depth},
    min_samples_split = {min_samples_split},
    min_samples_leaf  = {min_samples_leaf},
    max_features      = {repr(_max_feat)},
    criterion         = "{criterion}",
    splitter          = "{splitter}",
    class_weight      = {repr(_cw)},
    random_state      = 42,
)
clf.fit(X_train, y_train)

print("Train:", accuracy_score(y_train, clf.predict(X_train)))
print("Test: ", accuracy_score(y_test,  clf.predict(X_test)))
""",
        language="python",
    )