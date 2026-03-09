"""
Decision Tree Explorer  —  Dark Editorial Edition
===================================================
Structure:
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
import matplotlib.ticker as mticker
import streamlit as st

from matplotlib.colors import ListedColormap
from sklearn.datasets import (
    make_moons, make_circles, make_classification, load_breast_cancer,
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Tree Explorer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
#  CSS  —  load stylesheet only; toml owns all colours
# ─────────────────────────────────────────────────────────────
def _load_css() -> None:
    css = (Path(__file__).parent / "assets" / "style.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_load_css()


# ─────────────────────────────────────────────────────────────
#  CHART PALETTE  (matches THEME above)
# ─────────────────────────────────────────────────────────────
INK     = "#0C0C0E"
SURFACE = "#141418"
SURF2   = "#1C1C22"
BORDER  = "#2A2A32"
TEXT    = "#E8E6E1"
MUTED   = "#7A7880"
AMBER   = "#E8A838"
RED     = "#E05252"
GREEN   = "#52B788"
BLUE    = "#5B9BD5"

# Two class colours for scatter / boundary
C0, C0_BG = RED,   "#2E0E0E"
C1, C1_BG = BLUE,  "#0D1E2E"

plt.rcParams.update({
    "figure.facecolor":  INK,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    10,
    "axes.titleweight":  "semibold",
    "axes.labelsize":    8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        SURF2,
    "grid.linewidth":    0.6,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.major.size":  0,
    "ytick.major.size":  0,
    "legend.facecolor":  SURF2,
    "legend.edgecolor":  BORDER,
    "legend.framealpha": 1,
    "legend.fontsize":   8,
    "font.family":       "sans-serif",
    "font.size":         9,
    "text.color":        TEXT,
})


# ─────────────────────────────────────────────────────────────
#  CACHED DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset(name: str, n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    if name == "Moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
        return X, y, ["x1", "x2"]
    if name == "Circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.45, random_state=seed)
        return X, y, ["x1", "x2"]
    if name == "XOR":
        pts = []
        for cx, cy, cls in [(-1.5,1.5,0),(1.5,-1.5,0),(1.5,1.5,1),(-1.5,-1.5,1)]:
            k = n // 4
            pts.append(np.column_stack([
                rng.normal(cx, 0.55 + noise * 2.5, k),
                rng.normal(cy, 0.55 + noise * 2.5, k),
                np.full(k, cls),
            ]))
        arr = np.vstack(pts)
        return arr[:, :2], arr[:, 2].astype(int), ["x1", "x2"]
    if name == "Breast Cancer":
        d = load_breast_cancer()
        X, y = d.data[:, :2], d.target
        idx = rng.choice(len(y), min(n, len(y)), replace=False)
        return X[idx], y[idx], list(d.feature_names[:2])
    # Linear
    X, y = make_classification(
        n_samples=n, n_features=2,
        n_informative=2, n_redundant=0,
        n_clusters_per_class=2,
        class_sep=max(0.1, 1.2 - noise * 3),
        random_state=seed,
    )
    return X, y, ["feature_a", "feature_b"]


@st.cache_data(show_spinner=False)
def fit_model(X_tr, y_tr, max_depth, min_split, min_leaf,
              max_feat, criterion, splitter, cw):
    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_split=min_split,
        min_samples_leaf=min_leaf, max_features=max_feat,
        criterion=criterion, splitter=splitter,
        class_weight=cw, random_state=42,
    )
    return clf.fit(X_tr, y_tr)


@st.cache_data(show_spinner=False)
def depth_sweep(X_tr, y_tr, X_te, y_te, criterion: str):
    tr_acc, te_acc = [], []
    for d in range(1, 16):
        m = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=42)
        m.fit(X_tr, y_tr)
        tr_acc.append(accuracy_score(y_tr, m.predict(X_tr)) * 100)
        te_acc.append(accuracy_score(y_te, m.predict(X_te)) * 100)
    return tr_acc, te_acc


# ─────────────────────────────────────────────────────────────
#  CHART FUNCTIONS  (pure — no st.* calls inside)
# ─────────────────────────────────────────────────────────────
def _ax(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)


def chart_boundary(clf, X_tr, X_te, y_tr, y_te, fn):
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.patch.set_facecolor(INK)
    cmap = ListedColormap([C0_BG, C1_BG])

    for ax, Xs, ys, title in [
        (ax_tr, X_tr, y_tr, "Training set"),
        (ax_te, X_te, y_te, "Test set"),
    ]:
        _ax(ax)
        try:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.7,
                response_method="predict_proba",
                plot_method="pcolormesh", cmap=cmap,
            )
        except Exception:
            DecisionBoundaryDisplay.from_estimator(
                clf, X_tr, ax=ax, alpha=0.7,
                response_method="predict",
                plot_method="pcolormesh", cmap=cmap,
            )
        colors = [C0 if v == 0 else C1 for v in ys]
        ax.scatter(Xs[:, 0], Xs[:, 1], c=colors, s=18,
                   edgecolors=INK, linewidths=0.4, zorder=5)
        ax.legend(handles=[
            mpatches.Patch(facecolor=C1, edgecolor=BORDER, label="Class 1"),
            mpatches.Patch(facecolor=C0, edgecolor=BORDER, label="Class 0"),
        ])
        ax.set_title(title)
        ax.set_xlabel(fn[0])
        ax.set_ylabel(fn[1])

    fig.tight_layout(pad=2.0)
    return fig


def chart_tree(clf, fn, max_show):
    depth = min(clf.get_depth(), max_show)
    w = min(max(9, 4 * (2 ** depth)), 26)
    h = max(4, 2.4 * (depth + 1))
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Use monospace font for the tree node labels so
    # conditions like "x1 <= 0.432" are easy to scan
    with plt.rc_context({"font.family": "monospace", "font.size": 8, "text.color": "#000000"}):
        plot_tree(
            clf, feature_names=fn,
            class_names=["Class 0", "Class 1"],
            filled=True, rounded=True,
            impurity=True, proportion=False,
            max_depth=max_show, ax=ax,
            fontsize=8, precision=3,
        )

    # After plot_tree renders, walk every artist and make
    # arrows (FancyArrowPatch) and edge labels (Text) white.
    for artist in ax.get_children():
        # Arrow lines between nodes
        if hasattr(artist, "set_color") and hasattr(artist, "get_arrowstyle"):
            artist.set_color("#FFFFFF")
        # True / False text labels on edges
        if isinstance(artist, plt.Text):
            txt = artist.get_text().strip()
            if txt in ("True", "False"):
                artist.set_color("#FFFFFF")
                artist.set_fontweight("bold")

    suffix = (f"  ·  {depth + 1} of {clf.get_depth() + 1} levels shown"
              if clf.get_depth() > max_show else "")
    ax.set_title(f"Tree structure{suffix}", color=TEXT,
                 fontfamily="sans-serif", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    return fig


def chart_depth_acc(tr_acc, te_acc, cur_depth):
    depths = list(range(1, 16))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor(INK)
    _ax(ax)

    ax.fill_between(depths, tr_acc, te_acc,
                    where=[t > v for t, v in zip(tr_acc, te_acc)],
                    color=RED, alpha=0.12, zorder=1)
    ax.plot(depths, tr_acc, color=AMBER, lw=2,
            marker="o", ms=4, label="Train accuracy", zorder=3)
    ax.plot(depths, te_acc, color=BLUE,  lw=2,
            marker="o", ms=4, label="Test accuracy", zorder=3)
    ax.axvline(cur_depth, color=RED, lw=1.2,
               linestyle="--", label=f"Current  depth = {cur_depth}", zorder=4)

    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs max_depth  —  bias–variance tradeoff")
    ax.set_ylim(25, 103)
    ax.set_xlim(0.5, 15.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(framealpha=1)
    fig.tight_layout()
    return fig


def chart_feat_imp(clf, fn):
    fi     = clf.feature_importances_
    order  = np.argsort(fi)
    labels = [fn[i] for i in order]
    vals   = fi[order]
    colors = [AMBER if i == order[-1] else MUTED for i in range(len(fi))]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(fn) * 0.55 + 1.2)))
    fig.patch.set_facecolor(INK)
    _ax(ax)

    bars = ax.barh(labels, vals,
                   color=[colors[i] for i in range(len(order))],
                   edgecolor="none", height=0.45)
    for bar, val in zip(bars, vals):
        ax.text(val + max(fi) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color=MUTED, fontsize=8)

    ax.set_xlim(0, max(fi) * 1.35)
    ax.set_xlabel("Gini importance")
    ax.set_title("Feature importances")
    ax.grid(axis="x", color=SURF2, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def chart_confusion(clf, X_te, y_te):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(BORDER)

    ConfusionMatrixDisplay(
        confusion_matrix(y_te, clf.predict(X_te)),
        display_labels=["Class 0", "Class 1"],
    ).plot(ax=ax, colorbar=False, cmap="YlOrBr")

    ax.set_title("Confusion matrix  (test set)", color=TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.tick_params(colors=MUTED)
    for txt in ax.texts:
        txt.set_fontsize(13)
        txt.set_fontweight("bold")
        txt.set_color(INK)
    fig.tight_layout()
    return fig


def chart_leaf_dist(clf):
    samples, impurities = [], []
    def _walk(n):
        lc = clf.tree_.children_left[n]
        if lc == -1:
            samples.append(clf.tree_.n_node_samples[n])
            impurities.append(clf.tree_.impurity[n])
        else:
            _walk(lc); _walk(clf.tree_.children_right[n])
    _walk(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor(INK)

    for ax, data, label, color in [
        (ax1, samples,     "Samples per leaf",       AMBER),
        (ax2, impurities,  "Gini impurity per leaf",  BLUE),
    ]:
        _ax(ax)
        ax.hist(data,
                bins=min(30, max(5, len(data) // 3)),
                color=color, alpha=0.25,
                edgecolor=color, linewidth=0.8)
        ax.axvline(np.mean(data), color=RED, lw=1.3, linestyle="--",
                   label=f"Mean  {np.mean(data):.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend()

    fig.tight_layout(pad=2.0)
    return fig


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Decision Tree Explorer")
    st.caption("Adjust any parameter — all charts update live.")
    st.divider()

    st.markdown("**Dataset**")
    dataset = st.selectbox(
        "Dataset", ["Moons", "Circles", "XOR", "Linear", "Breast Cancer"],
        label_visibility="collapsed",
    )
    n_samples = st.slider(
        "Samples", 200, 2000, 600, step=100,
        help="Total number of data points to generate.",
    )
    noise = st.slider(
        "Noise", 0.00, 0.50, 0.20, step=0.05,
        help="Scatter added to the data. Higher values make classification harder.",
    )
    col_a, col_b = st.columns(2)
    test_pct  = col_a.slider("Test %", 10, 40, 25, step=5)
    rand_seed = col_b.slider("Seed",    0, 99, 42)

    st.divider()
    st.markdown("**Model hyperparameters**")

    max_depth = st.slider(
        "max_depth", 1, 15, 4,
        help="How many levels the tree is allowed to grow. "
             "Shallow = simpler model. Deep = more complex, higher overfit risk.",
    )
    min_samples_split = st.slider(
        "min_samples_split", 2, 60, 2,
        help="Minimum samples a node must hold before it can be split further.",
    )
    min_samples_leaf = st.slider(
        "min_samples_leaf", 1, 40, 1,
        help="Every leaf must contain at least this many samples. "
             "Higher values smooth the decision boundary.",
    )
    max_features = st.selectbox(
        "max_features", ["all", "sqrt", "log2"],
        help="How many features to consider at each split candidate.",
    )
    criterion = st.selectbox(
        "criterion", ["gini", "entropy", "log_loss"],
        help="The impurity measure used to evaluate split quality.",
    )
    splitter = st.selectbox(
        "splitter", ["best", "random"],
        help="'best' always picks the optimal split; "
             "'random' introduces stochastic variation.",
    )
    class_weight = st.selectbox(
        "class_weight", ["None", "balanced"],
        help="'balanced' upweights minority-class samples proportionally.",
    )

    st.divider()
    st.markdown("**Presets**")
    col_c, col_d = st.columns(2)
    if col_c.button("Stump",    use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 1, 2, 1
    if col_c.button("Overfit",  use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 15, 2, 1
    if col_d.button("Balanced", use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 4, 5, 3
    if col_d.button("Pruned",   use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 5, 20, 8

_max_feat = None       if max_features == "all"  else max_features
_cw       = None       if class_weight == "None" else "balanced"


# ─────────────────────────────────────────────────────────────
#  FIT
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    X, y, fn = get_dataset(dataset, n_samples, noise, rand_seed)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=test_pct / 100,
    random_state=rand_seed, stratify=y,
)

with st.spinner("Fitting model…"):
    clf = fit_model(
        X_tr, y_tr,
        max_depth, min_samples_split, min_samples_leaf,
        _max_feat, criterion, splitter, _cw,
    )

tr_acc  = accuracy_score(y_tr, clf.predict(X_tr)) * 100
te_acc  = accuracy_score(y_te, clf.predict(X_te)) * 100
gap     = tr_acc - te_acc
depth   = clf.get_depth()
n_nodes = clf.tree_.node_count
n_leaves= clf.get_n_leaves()


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("## Decision Tree Explorer")
st.caption(
    f"Dataset: **{dataset}**  ·  "
    f"{len(X_tr):,} train  /  {len(X_te):,} test  ·  "
    f"Features: `{fn[0]}`, `{fn[1]}`"
)
st.divider()

# ── Metrics ──
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Train accuracy",  f"{tr_acc:.1f}%")
c2.metric("Test accuracy",   f"{te_acc:.1f}%",
          delta=f"{gap:+.1f}% gap",
          delta_color="inverse" if gap > 10 else "normal")
c3.metric("Tree depth",      str(depth))
c4.metric("Total nodes",     str(n_nodes))
c5.metric("Leaf nodes",      str(n_leaves))
c6.metric("Train n",         f"{len(X_tr):,}")
c7.metric("Test n",          f"{len(X_te):,}")

# ── Single contextual banner ──
if gap > 15 and tr_acc > 95:
    st.error(
        f"**Overfitting.** Train is {gap:.1f}pp above test. "
        "The tree is memorising the training set. "
        "Raise `min_samples_split`, raise `min_samples_leaf`, or reduce `max_depth`."
    )
elif te_acc < 68:
    st.warning(
        "**Underfitting.** Test accuracy is low — the model is too simple. "
        "Increase `max_depth` or lower `min_samples_split`."
    )
elif gap < 3 and te_acc > 88:
    st.success(
        f"**Generalising well.** Train–test gap is only {gap:.1f}pp "
        f"and test accuracy is {te_acc:.1f}%."
    )

st.divider()


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
(tab_bnd, tab_tree, tab_bias,
 tab_leaf, tab_imp, tab_report) = st.tabs([
    "Decision boundary",
    "Tree structure",
    "Depth vs accuracy",
    "Leaf analysis",
    "Feature importance",
    "Report",
])


# ── Decision boundary ────────────────────────────────────────
with tab_bnd:
    st.write(
        "Each coloured region is the class the tree predicts for every point "
        "in that area of feature space. Because trees split one feature at a time, "
        "every boundary must be a horizontal or vertical line — "
        "complex shapes are built from many such cuts stacked together."
    )
    st.pyplot(chart_boundary(clf, X_tr, X_te, y_tr, y_te, fn),
              use_container_width=True)
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**What to look for**")
        st.markdown(
            "- All splits are axis-aligned — this is a hard geometric constraint.\n"
            "- Many small islands in the training plot but smooth blobs in the test "
            "plot is a classic sign of overfitting.\n"
            "- Points plotted inside the wrong region are misclassifications."
        )
    with right:
        st.markdown("**What your current settings produce**")
        if max_depth <= 2:
            st.info("Depth 1–2 produces very coarse, blocky regions. "
                    "The boundary cannot follow the true shape of the data.")
        elif gap > 12:
            st.warning("The boundary is fragmented. The tree is fitting "
                       "noise rather than signal.")
        elif min_samples_leaf >= 8:
            st.info("A high `min_samples_leaf` forces larger leaf regions, "
                    "producing a smoother, more conservative boundary.")
        else:
            st.info("Settings look balanced. Try pushing `max_depth` to 10+ "
                    "to see the boundary fragment, or `min_samples_leaf` to 10+ "
                    "to see it smooth out.")


# ── Tree structure ───────────────────────────────────────────
with tab_tree:
    st.write(
        "Each internal node shows the split rule (`feature ≤ threshold`), "
        "Gini impurity, and sample count. "
        "Leaves show the predicted class. "
        "Left branch = condition is True, right = False."
    )
    lvl = st.slider(
        "Levels to show", 1, min(10, depth + 1), min(4, depth + 1),
        key="tree_lvl",
    )
    st.pyplot(chart_tree(clf, fn, lvl), use_container_width=True)
    if depth > lvl:
        st.caption(f"Full tree has {depth + 1} levels. Increase the slider to reveal more.")
    with st.expander("Text decision rules"):
        st.code(export_text(clf, feature_names=fn, max_depth=8), language="text")


# ── Depth vs accuracy ────────────────────────────────────────
with tab_bias:
    st.write(
        "A fresh tree is trained for every value of `max_depth` from 1 to 15, "
        "holding all other hyperparameters constant. "
        "The amber line is training accuracy; the blue line is test accuracy. "
        "The red shading between them is the overfit gap. "
        "The vertical dashed line marks your current setting."
    )
    with st.spinner("Running depth sweep…"):
        tr_sw, te_sw = depth_sweep(X_tr, y_tr, X_te, y_te, criterion)
    st.pyplot(chart_depth_acc(tr_sw, te_sw, max_depth), use_container_width=True)
    st.divider()
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**Underfitting region** (left)")
        st.markdown(
            "Both curves are low. The tree lacks the capacity to learn "
            "the pattern in the data. This is high-bias."
        )
    with cb:
        st.markdown("**Sweet spot** (middle)")
        st.markdown(
            "Test accuracy is near its peak and the gap to training is small. "
            "The model generalises well to unseen data."
        )
    with cc:
        st.markdown("**Overfitting region** (right)")
        st.markdown(
            "Training accuracy approaches 100% but test accuracy plateaus or drops. "
            "The tree is memorising noise. This is high-variance."
        )


# ── Leaf analysis ────────────────────────────────────────────
with tab_leaf:
    st.write(
        "A leaf with Gini impurity = 0 contains only one class — "
        "the tree has perfectly separated those training samples. "
        "Many pure leaves alongside a large train–test gap "
        "strongly indicates overfitting."
    )
    st.pyplot(chart_leaf_dist(clf), use_container_width=True)
    st.divider()

    tree_ = clf.tree_
    leaf_ids = [i for i in range(tree_.node_count)
                if tree_.children_left[i] == -1]
    rows = []
    for lid in leaf_ids:
        n    = tree_.n_node_samples[lid]
        gini = tree_.impurity[lid]
        val  = tree_.value[lid][0]
        cls  = int(np.argmax(val))
        conf = val[cls] / val.sum() * 100
        rows.append({
            "Leaf":           lid,
            "Samples":        n,
            "Gini impurity":  round(gini, 4),
            "Predicted class": cls,
            "Confidence (%)": round(conf, 1),
        })
    df_leaves = pd.DataFrame(rows).sort_values("Samples", ascending=False)
    st.dataframe(
        df_leaves.style
            .background_gradient(subset=["Gini impurity"],  cmap="YlOrRd")
            .background_gradient(subset=["Confidence (%)"], cmap="YlGn")
            .format({"Gini impurity": "{:.4f}", "Confidence (%)": "{:.1f}"}),
        use_container_width=True, height=280,
    )
    m1, m2 = st.columns(2)
    pure = (df_leaves["Gini impurity"] < 0.01).mean() * 100
    m1.metric("Pure leaves (Gini < 0.01)", f"{pure:.0f}%")
    m2.metric("Mean samples per leaf",      f"{df_leaves['Samples'].mean():.1f}")


# ── Feature importance ───────────────────────────────────────
with tab_imp:
    st.write(
        "Gini importance is the total reduction in node impurity a feature contributes, "
        "weighted by the fraction of training samples reaching each node. "
        "The amber bar is the most important feature."
    )
    st.pyplot(chart_feat_imp(clf, fn), use_container_width=True)
    fi_df = (
        pd.DataFrame({"Feature": fn, "Importance": clf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df.index += 1
    fi_df.index.name = "Rank"
    st.dataframe(
        fi_df.style
            .format({"Importance": "{:.4f}"})
            .background_gradient(subset=["Importance"], cmap="YlOrBr"),
        use_container_width=True,
    )


# ── Report ───────────────────────────────────────────────────
with tab_report:
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Confusion matrix**")
        st.pyplot(chart_confusion(clf, X_te, y_te), use_container_width=True)
    with cr:
        st.markdown("**Classification report**")
        rdf = pd.DataFrame(
            classification_report(
                y_te, clf.predict(X_te),
                target_names=["Class 0", "Class 1"],
                output_dict=True,
            )
        ).T
        st.dataframe(
            rdf.style
                .format("{:.3f}", subset=["precision", "recall", "f1-score"])
                .background_gradient(subset=["f1-score"], cmap="YlGn"),
            use_container_width=True,
        )

    st.divider()
    st.markdown("**Current configuration**")
    st.dataframe(pd.DataFrame([{
        "max_depth":          max_depth,
        "min_samples_split":  min_samples_split,
        "min_samples_leaf":   min_samples_leaf,
        "max_features":       max_features,
        "criterion":          criterion,
        "splitter":           splitter,
        "class_weight":       class_weight,
    }]), use_container_width=True)

    st.markdown("**Reproducible code**")
    st.code(f"""\
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples={n_samples}, noise={noise:.2f}, random_state={rand_seed})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_pct / 100:.2f}, random_state={rand_seed}, stratify=y,
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
""", language="python")