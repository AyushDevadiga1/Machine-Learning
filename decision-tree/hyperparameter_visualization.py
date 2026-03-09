"""
Decision Tree Hyperparameter Explorer
======================================
Run:   streamlit run app.py
Deps:  pip install -r requirements.txt

Structure:
  app.py               — logic only, zero inline styles
  assets/style.css     — all layout & component CSS
  .streamlit/config.toml — colour tokens & font
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
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Tree Explorer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  LOAD CSS  — one call, no inline styles anywhere else
# ─────────────────────────────────────────────────────────────
css = Path("assets/style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB DEFAULTS  — monochrome to match minimal theme
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FFFFFF",
    "axes.facecolor":    "#FAFAFA",
    "axes.edgecolor":    "#E4E4E7",
    "axes.labelcolor":   "#71717A",
    "axes.titlecolor":   "#18181B",
    "axes.titlesize":    10,
    "axes.titleweight":  "semibold",
    "axes.labelsize":    8,
    "axes.grid":         True,
    "grid.color":        "#F4F4F5",
    "grid.linewidth":    0.8,
    "xtick.color":       "#A1A1AA",
    "ytick.color":       "#A1A1AA",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.facecolor":  "#FFFFFF",
    "legend.edgecolor":  "#E4E4E7",
    "legend.framealpha": 1,
    "legend.fontsize":   8,
    "font.size":         9,
    "text.color":        "#18181B",
})

# Two fixed data colours — only these two need to be vivid
C0 = "#F43F5E"   # rose  — class 0
C1 = "#0EA5E9"   # sky   — class 1
C0_BG = "#FFF1F3"
C1_BG = "#F0F9FF"
ACCENT = "#18181B"


# ─────────────────────────────────────────────────────────────
#  DATASET LOADER
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
            pts.extend(zip(
                rng.normal(cx, .55 + noise * 2.5, n),
                rng.normal(cy, .55 + noise * 2.5, n),
                [c] * n,
            ))
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
            class_sep=max(0.1, 1.2 - noise * 3),
            random_state=seed,
        )
        fn = ["Feature A", "Feature B"]
    return X, y, fn


# ─────────────────────────────────────────────────────────────
#  PLOT FUNCTIONS  — no inline CSS, colours from constants
# ─────────────────────────────────────────────────────────────
def _ax(ax):
    ax.set_facecolor("#FAFAFA")
    for sp in ax.spines.values():
        sp.set_edgecolor("#E4E4E7")
    ax.tick_params(colors="#A1A1AA", labelsize=8)


def plot_boundary(clf, X_tr, X_te, y_tr, y_te, fn):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    cmap_bg = ListedColormap([C0_BG, C1_BG])
    for ax, (Xs, ys), title in zip(
        axes,
        [(X_tr, y_tr), (X_te, y_te)],
        ["Training set", "Test set"],
    ):
        _ax(ax)
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
        dot_c = [C0 if v == 0 else C1 for v in ys]
        ax.scatter(Xs[:, 0], Xs[:, 1], c=dot_c, s=22,
                   edgecolors="#FFFFFF", linewidths=0.6, zorder=5)
        ax.legend(handles=[
            mpatches.Patch(color=C1, label="Class 1"),
            mpatches.Patch(color=C0, label="Class 0"),
        ])
        ax.set_title(title)
        ax.set_xlabel(fn[0])
        ax.set_ylabel(fn[1])
    fig.tight_layout(pad=2)
    return fig


def plot_tree_struct(clf, fn, max_show):
    d = min(clf.get_depth(), max_show)
    w = max(10, 4 * (2 ** d))
    h = max(4, 2.5 * (d + 1))
    fig, ax = plt.subplots(figsize=(min(w, 26), h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    plot_tree(clf, feature_names=fn, class_names=["Class 0", "Class 1"],
              filled=True, rounded=True, impurity=True,
              max_depth=max_show, ax=ax, fontsize=8, precision=3)
    label = (f"Top {d+1} of {clf.get_depth()+1} levels"
             if clf.get_depth() > max_show else "Full tree")
    ax.set_title(label)
    fig.tight_layout()
    return fig


def plot_depth_acc(X_tr, y_tr, X_te, y_te, cur, criterion):
    depths = list(range(1, 16))
    tr_a, te_a = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=42)
        m.fit(X_tr, y_tr)
        tr_a.append(accuracy_score(y_tr, m.predict(X_tr)) * 100)
        te_a.append(accuracy_score(y_te, m.predict(X_te)) * 100)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#FFFFFF")
    _ax(ax)
    ax.plot(depths, tr_a, color=ACCENT,  lw=2, marker="o", ms=4, label="Train")
    ax.plot(depths, te_a, color="#71717A", lw=2, marker="o", ms=4, label="Test")
    ax.fill_between(depths, tr_a, te_a,
                    where=[t > v for t, v in zip(tr_a, te_a)],
                    alpha=0.06, color=C0)
    ax.axvline(x=cur, color=C0, lw=1.5, linestyle="--", label=f"depth={cur}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Depth vs Accuracy")
    ax.set_ylim(30, 103); ax.set_xlim(0.5, 15.5)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feat_imp(clf, fn):
    fi = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(fn) * 0.55 + 1.2)))
    fig.patch.set_facecolor("#FFFFFF")
    _ax(ax)
    colors = [ACCENT if v == max(fi) else "#D4D4D8" for v in fi]
    bars = ax.barh(fn, fi, color=colors, edgecolor="#FFFFFF", height=0.45)
    for bar, val in zip(bars, fi):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="#A1A1AA", fontsize=8)
    ax.set_xlim(0, max(fi) * 1.3)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Gini reduction")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_confusion(clf, X_te, y_te):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, clf.predict(X_te)),
        display_labels=["Class 0", "Class 1"],
    ).plot(ax=ax, colorbar=False, cmap="Greys")
    ax.set_title("Confusion Matrix")
    for txt in ax.texts:
        txt.set_color("#18181B"); txt.set_fontsize(13); txt.set_fontweight("bold")
    for sp in ax.spines.values(): sp.set_edgecolor("#E4E4E7")
    fig.tight_layout()
    return fig


def plot_leaves(clf):
    s_list, g_list = [], []
    def walk(nid):
        lc = clf.tree_.children_left[nid]
        if lc == -1:
            s_list.append(clf.tree_.n_node_samples[nid])
            g_list.append(clf.tree_.impurity[nid])
        else:
            walk(lc); walk(clf.tree_.children_right[nid])
    walk(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor("#FFFFFF")
    for ax, data, title in [
        (axes[0], s_list, "Samples per Leaf"),
        (axes[1], g_list, "Gini Impurity per Leaf"),
    ]:
        _ax(ax)
        ax.hist(data, bins=min(30, max(5, len(data) // 3)),
                color=ACCENT, alpha=0.15, edgecolor=ACCENT, linewidth=0.6)
        ax.axvline(np.mean(data), color=C0, lw=1.5, linestyle="--",
                   label=f"mean={np.mean(data):.2f}")
        ax.set_title(title); ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.legend()
    fig.tight_layout(pad=2)
    return fig


# ─────────────────────────────────────────────────────────────
#  HTML HELPERS  — use CSS classes from style.css only
# ─────────────────────────────────────────────────────────────
def hint(title, body):
    st.markdown(
        f'<div class="hint-card">'
        f'<div class="hint-title">{title}</div>'
        f'<div class="hint-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def state_badge(text):
    st.markdown(
        f'<div style="text-align:right;margin-top:20px">'
        f'<span class="state-badge">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def pill(text):
    return f'<span class="pill">{text}</span>'


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌳 Decision Tree")
    st.caption("Adjust sliders — charts update instantly.")
    st.divider()

    st.markdown('<p class="sidebar-label">Dataset</p>', unsafe_allow_html=True)
    dataset_name = st.selectbox(
        "dataset", ["Two Moons", "Two Circles", "XOR Blobs", "Linear Separable", "Breast Cancer"],
        label_visibility="collapsed",
    )
    n_samples = st.slider("Samples",      200, 2000, 600, step=100)
    noise     = st.slider("Noise",        0.00, 0.50, 0.20, step=0.05)
    test_pct  = st.slider("Test split %", 10, 40, 25, step=5)
    seed      = st.slider("Seed",         0, 99, 42)

    st.divider()
    st.markdown('<p class="sidebar-label">Hyperparameters</p>', unsafe_allow_html=True)

    max_depth         = st.slider("max_depth",         1, 15, 4)
    min_samples_split = st.slider("min_samples_split", 2, 60, 2)
    min_samples_leaf  = st.slider("min_samples_leaf",  1, 40, 1)
    max_features      = st.selectbox("max_features",   ["all", "sqrt", "log2"])
    criterion         = st.selectbox("criterion",      ["gini", "entropy", "log_loss"])
    splitter          = st.selectbox("splitter",       ["best", "random"])
    class_weight      = st.selectbox("class_weight",   ["None", "balanced"])

    st.divider()
    st.markdown('<p class="sidebar-label">Presets</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("Stump",    use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 1, 2, 1
    if c1.button("Overfit",  use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 15, 2, 1
    if c2.button("Balanced", use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 4, 5, 3
    if c2.button("Pruned",   use_container_width=True):
        max_depth, min_samples_split, min_samples_leaf = 5, 20, 8

max_features_val = None if max_features  == "all"  else max_features
class_weight_val = None if class_weight  == "None" else "balanced"


# ─────────────────────────────────────────────────────────────
#  FIT
# ─────────────────────────────────────────────────────────────
X, y, fn = load_dataset(dataset_name, n_samples, noise, seed)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=test_pct / 100, random_state=seed, stratify=y,
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

if max_depth <= 1:           _state = "Stump"
elif tr_acc > 97 and gap>15: _state = "Overfitting"
elif te_acc < 68:            _state = "Underfitting"
elif gap < 5 and te_acc>85:  _state = "Generalizing"
elif gap > 8:                _state = "Slight Overfit"
else:                        _state = "Balanced"


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
hcol, bcol = st.columns([5, 1])
with hcol:
    st.markdown(
        f"### Decision Tree Explorer\n"
        f"<span style='font-size:13px;color:#A1A1AA'>"
        f"{dataset_name} &nbsp;·&nbsp; "
        f"{len(X_tr)} train / {len(X_te)} test &nbsp;·&nbsp; "
        + pill(fn[0]) + " &nbsp;&amp;&nbsp; " + pill(fn[1]) +
        f"</span>",
        unsafe_allow_html=True,
    )
with bcol:
    state_badge(_state)

# ── Metrics ──
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Train Acc",      f"{tr_acc:.1f}%")
c2.metric("Test Acc",       f"{te_acc:.1f}%", delta=f"{gap:+.1f}%")
c3.metric("Depth",          str(depth))
c4.metric("Nodes",          str(nodes))
c5.metric("Leaves",         str(leaves))
c6.metric("Train n",        str(len(X_tr)))
c7.metric("Test n",         str(len(X_te)))

if gap > 15:
    st.warning(f"Train is {gap:.1f}% above test — likely overfitting. "
               "Raise `min_samples_split` or lower `max_depth`.")
elif te_acc < 68:
    st.info("Test accuracy low — model may be underfitting. "
            "Try increasing `max_depth`.")

st.divider()


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Boundary", "Tree", "Depth vs Acc", "Leaves", "Importance", "Report",
])

# ── Boundary ──────────────────────────────────────────────────
with tab1:
    st.caption("Coloured regions show the predicted class. All splits are axis-aligned.")
    st.pyplot(plot_boundary(clf, X_tr, X_te, y_tr, y_te, fn), use_container_width=True)
    st.divider()
    lc, rc = st.columns(2)
    with lc:
        hint("What to look for",
             "All splits are perpendicular to axes.<br>"
             "Tiny isolated islands → overfitting.<br>"
             "Smooth large regions → good generalisation.")
    with rc:
        if max_depth <= 2:
            hint("Current settings", "Shallow tree — coarse regions, can't fit complex data.")
        elif gap > 12:
            hint("Current settings", "Jagged boundary — tree is memorising noise.")
        elif min_samples_leaf >= 8:
            hint("Current settings", "Large min-leaf enforces smooth, robust regions.")
        else:
            hint("Current settings", "Balanced. Increase max_depth to watch it fragment.")

# ── Tree ──────────────────────────────────────────────────────
with tab2:
    st.caption("Node: feature ≤ threshold · Gini · samples.  Leaf: predicted class.")
    max_show = st.slider("Levels to show", 1, min(10, depth+1), min(4, depth+1), key="tlvl")
    st.pyplot(plot_tree_struct(clf, fn, max_show), use_container_width=True)
    if depth > max_show:
        st.caption(f"Tree has {depth+1} total levels.")
    with st.expander("Text decision rules"):
        st.code(export_text(clf, feature_names=fn, max_depth=6), language="text")

# ── Depth vs Acc ──────────────────────────────────────────────
with tab3:
    st.caption("Sweeps max_depth 1→15. Dashed line = current depth.")
    st.pyplot(plot_depth_acc(X_tr, y_tr, X_te, y_te, max_depth, criterion),
              use_container_width=True)
    st.divider()
    ca, cb, cc = st.columns(3)
    with ca: hint("Left — Underfitting",  "Both curves are low. Model too simple.")
    with cb: hint("Sweet Spot",           "Both curves high and close together.")
    with cc: hint("Right — Overfitting",  "Train → 100%, test starts to drop.")

# ── Leaves ────────────────────────────────────────────────────
with tab4:
    st.caption("Pure leaves (Gini = 0) mean perfect separation of training points.")
    st.pyplot(plot_leaves(clf), use_container_width=True)
    st.divider()
    tree_ = clf.tree_
    leaf_ids = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1]
    rows = []
    for lid in leaf_ids:
        n, imp = tree_.n_node_samples[lid], tree_.impurity[lid]
        val = tree_.value[lid][0]
        cls = int(np.argmax(val))
        rows.append({"Leaf": lid, "Samples": n,
                     "Gini": round(imp, 4),
                     "Class": cls,
                     "Confidence": round(val[cls] / val.sum() * 100, 1)})
    df_lv = pd.DataFrame(rows).sort_values("Samples", ascending=False)
    st.dataframe(
        df_lv.style
            .background_gradient(subset=["Gini"],       cmap="Greys")
            .background_gradient(subset=["Confidence"], cmap="Greens"),
        use_container_width=True, height=260,
    )
    pure_pct = (df_lv["Gini"] < 0.01).mean() * 100
    mA, mB = st.columns(2)
    mA.metric("Pure leaves",          f"{pure_pct:.0f}%")
    mB.metric("Avg samples per leaf", f"{df_lv['Samples'].mean():.1f}")

# ── Importance ────────────────────────────────────────────────
with tab5:
    st.caption("Total Gini impurity reduction contributed by each feature.")
    st.pyplot(plot_feat_imp(clf, fn), use_container_width=True)
    fi_df = pd.DataFrame({
        "Feature":    fn,
        "Importance": clf.feature_importances_,
        "Rank":       pd.Series(clf.feature_importances_)
                        .rank(ascending=False).astype(int).values,
    }).sort_values("Importance", ascending=False)
    st.dataframe(
        fi_df.style.background_gradient(subset=["Importance"], cmap="Greys"),
        use_container_width=True,
    )

# ── Report ────────────────────────────────────────────────────
with tab6:
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Confusion Matrix**")
        st.pyplot(plot_confusion(clf, X_te, y_te), use_container_width=True)
    with cr:
        st.markdown("**Classification Report**")
        rdf = pd.DataFrame(classification_report(
            y_te, clf.predict(X_te),
            target_names=["Class 0", "Class 1"],
            output_dict=True,
        )).T
        st.dataframe(
            rdf.style
                .format("{:.3f}", subset=["precision", "recall", "f1-score"])
                .background_gradient(subset=["f1-score"], cmap="Greens"),
            use_container_width=True,
        )
    st.divider()
    st.markdown("**sklearn equivalent**")
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
    max_depth         = {max_depth},
    min_samples_split = {min_samples_split},
    min_samples_leaf  = {min_samples_leaf},
    max_features      = {repr(max_features_val)},
    criterion         = "{criterion}",
    splitter          = "{splitter}",
    class_weight      = {repr(class_weight_val)},
    random_state      = 42,
)
clf.fit(X_train, y_train)
print(f"Train: {{accuracy_score(y_train, clf.predict(X_train)):.3f}}")
print(f"Test:  {{accuracy_score(y_test,  clf.predict(X_test)):.3f}}")
""", language="python")