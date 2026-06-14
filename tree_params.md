Here are all params in that format:

---

**HOW TO SPLIT**

`criterion` → The formula used to measure how "pure" a split is
- `gini` → Measures probability of misclassifying a random sample. Fast and works well in most cases *(classifier default)*
- `entropy` → Uses information gain. Slightly slower than gini but sometimes better on noisy data
- `log_loss` → Same as entropy mathematically. Used when you want probabilistic outputs *(classifier)*
- `squared_error` → Mean squared error. Standard choice for regression *(regressor default)*
- `friedman_mse` → Improved MSE with Friedman's correction. Better for certain splits
- `absolute_error` → Mean absolute error. More robust to outliers than squared_error
- `poisson` → For count data (e.g. number of events). Assumes Poisson distribution

`splitter` → Decides how the tree searches for the best split at each node
- `best` → Looks at every feature and picks the globally best split. Slower but more accurate *(default)*
- `random` → Picks a random feature and finds the best threshold on just that. Faster, adds randomness. ExtraTrees use this

`max_features` → How many features to consider when looking for a split
- `None` → Consider all features at every split *(standard tree default)*
- `"sqrt"` → Use √(total features). Common in Random Forests
- `"log2"` → Use log₂(total features). Even fewer features, more randomness
- `int` → Exact number of features to consider
- `float` → Fraction of total features, e.g. `0.5` = half the features

---

**WHEN TO STOP GROWING**

`max_depth` → Maximum number of levels the tree can grow
- `None` → Keep splitting until all leaves are pure or other limits hit *(default, can overfit)*
- `int` → e.g. `max_depth=5` means root + 5 levels max. Lower = simpler tree

`min_samples_split` → Minimum samples a node must have before it's allowed to split
- `2` → Default. Very permissive, almost every node can split
- `int` → e.g. `10` means a node with fewer than 10 samples becomes a leaf
- `float` → Fraction of total training samples, e.g. `0.02` = 2% of dataset

`min_samples_leaf` → Minimum samples that must land in each child after a split
- `1` → Default. Even a single sample can form a leaf
- `int` → e.g. `5` means both children must have ≥5 samples or the split is rejected
- `float` → Fraction of total training samples

`max_leaf_nodes` → Cap on the total number of leaves in the whole tree
- `None` → No cap *(default)*
- `int` → e.g. `20` means the tree stops after 20 leaves. Grows best splits first

`ccp_alpha` → Cost-complexity pruning. Prunes branches after the tree is fully grown
- `0.0` → No pruning *(default)*
- `float > 0` → Higher value = more aggressive pruning = simpler tree. Use `cost_complexity_pruning_path()` to find the best value

`min_impurity_decrease` → A split only happens if it reduces impurity by at least this amount
- `0.0` → Any decrease in impurity is enough to split *(default)*
- `float > 0` → e.g. `0.01` filters out tiny, unhelpful splits

`min_weight_fraction_leaf` → Like min_samples_leaf but based on sum of sample weights
- `0.0` → No weight-based restriction *(default)*
- `float in [0, 0.5]` → e.g. `0.1` means each leaf must carry ≥10% of total weight. Only matters when you pass `sample_weight` to `fit()`

---

**CLASS IMBALANCE**

`class_weight` → Assigns importance to classes so rare ones aren't ignored *(classifier only)*
- `None` → All classes treated equally *(default)*
- `"balanced"` → Auto-weights classes inversely by frequency. Use this when data is imbalanced
- `dict` → Manual weights, e.g. `{0:1, 1:5}` makes class 1 five times more important

---

**BUSINESS CONSTRAINTS**

`monotonic_cst` → Forces predictions to respect a known direction as a feature increases
- `None` → No constraints *(default)*
- `1` → Prediction must increase as this feature increases. e.g. more experience → higher salary
- `-1` → Prediction must decrease as this feature increases. e.g. more distance → lower price
- `0` → No constraint on this feature
- Pass as array, one value per feature: e.g. `[1, 0, -1]`

---

**REPRODUCIBILITY**

`random_state` → Controls the random seed so results are the same every run
- `None` → Different result each run *(default)*
- `int` → e.g. `42` fixes the seed. Always set this in experiments and production
- `np.random.RandomState` → Pass a numpy random state object for fine-grained control