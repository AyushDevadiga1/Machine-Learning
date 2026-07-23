# Decision Tree — Derivation at a Glance

A structured summary of all steps, evaluations, and key discoveries from the first-principles derivation.

---

## 1. Why Decision Trees Exist

**Linear models fail when patterns are decision-based, not line-based.**

| Dataset Type | Separable? | Model |
|---|---|---|
| Age & Salary with a clean linear boundary | ✅ Yes | Logistic Regression |
| Age & Salary with a checkerboard XOR pattern | ❌ No | Needs Decision Tree |

**Key discovery:** Some data cannot be separated by any single line, no matter how it's rotated or translated. You need *multiple decisions* to split the space.

---

## 2. The Decision = Nested If-Else

A prediction on XOR-style data is literally a nested if-else:

```python
if age < 30:
    if salary < 50000:  → class 0
    else:               → class 1
else:
    if salary < 50000:  → class 1
    else:               → class 0
```

**Key discovery:** Each `if` condition is a *split* — a line drawn across one region of the graph. Trees are divide-and-conquer over feature space.

---

## 3. How Splits Carve the Space

Visualised as four progressive stages:

1. **Raw data** — all points mixed together
2. **First split (age < 30)** — vertical line, 2 regions, each still mixed
3. **Sub-divide left (salary < 50k)** — horizontal line on left half only
4. **Sub-divide right (salary < 50k)** — same line extended to right half → all 4 points isolated

**Key discovery:** No single line achieves this. Multiple sequential cuts (decisions) create the correct classification regions.

---

## 4. The Desired Property of a Split

> **A good question makes children simpler than the parent.**

**Example:**

| State | Composition | Uncertainty |
|---|---|---|
| Parent | 50 Yes, 50 No | Maximum (50/50) |
| Left child (after split) | 48 No, 2 Yes | Very low |
| Right child (after split) | 2 No, 48 Yes | Very low |

**Key discovery:** We need a *numerical measure of confusion (impurity)*. If we can measure impurity before and after a split, we can choose the split that reduces it the most — then repeat recursively.

---

## 5. Choosing the Best Split — What Makes a Good Threshold?

Compared three candidate splits on 6 people (3 No, 3 Yes):

| Split | Left Child | Right Child | Confusion Level |
|---|---|---|---|
| Age < 30 | 3 No (pure) | 3 Yes (pure) | ✅ None |
| Age < 46 | 3 No + 1 Yes (mixed) | 2 Yes (pure) | ⚠️ Some |
| Age < 100 | All 6 (completely mixed) | — | ❌ Maximum |

**Key discovery:** The split at Age < 30 is best because both children are pure. We need a formula to quantify this.

---

## 6. Defining "Confusion" — Required Properties

For binary classification with `p = P(Yes)` and `1−p = P(No)`, the confusion function must:

- Equal **0** when `p = 0` (everyone is No → no uncertainty)
- Equal **0** when `p = 1` (everyone is Yes → no uncertainty)
- Reach **maximum** at `p = 0.5` (50/50 split → highest uncertainty)
- Be **symmetric** — swapping Yes/No labels shouldn't change the value

Shape required:
```
Confusion
    |      ▲
    |     / \
    |    /   \
    |___/     \___  p
        0  0.5  1
```

---

## 7. Deriving Gini Impurity

**Proposal 1 — Linear functions:** Failed.
- `C(p) = p` → gives 1 at p=1, but should give 0
- `C(p) = 1-p` → gives 1 at p=0, but should give 0

**Proposal 2 — Quadratic (Parabola):** ✅ Works.

Candidate: `C(p) = p(1−p)`

| p | p(1−p) |
|---|---|
| 0 | 0 |
| 0.5 | 0.25 |
| 1 | 0 |

Max is 0.25, not 1 → scale by 4 → `C(p) = 4p(1−p)`, but conventionally written as:

$$G(p) = 2p(1-p)$$

**Generalised to multi-class:**

$$G = 1 - \sum p_i^2$$

**Verification:** For binary case with `p_Yes = p`, `p_No = 1−p`:

$$G = 1 - (p^2 + (1-p)^2) = 2p(1-p) \checkmark$$

**Key discovery:** Gini Impurity asks — *"How mixed is this node?"*

---

## 8. Deriving Entropy (The Shannon Story)

**Intuition:** Information = Surprise.

| Event | Probability | Surprise |
|---|---|---|
| "The sun will rise tomorrow" | ~1 | 0 (you already knew) |
| "You won the lottery" | ~0 | Massive |

**Rule:** The rarer an event, the more information it carries.

**Mathematical derivation:**

Two independent events should have *additive* information, but probabilities *multiply*:
- Probabilities multiply: `p × q`
- Information should add: `I(p) + I(q)`

Only one function satisfies `f(a × b) = f(a) + f(b)` → **The Logarithm**

Add a negative sign so smaller probabilities → larger values:

$$I(p) = -\log(p)$$

**From individual surprise to average surprise (Entropy):**

$$H = -\sum p_i \log(p_i)$$

**Key discovery:** Entropy asks — *"How much information is still missing before I know the class?"* Gini and Entropy are complementary measures — both valid impurity metrics.

---

## 9. Information Gain & Weighted Conditional Entropy

**Goal:** Evaluate how good a split is by measuring the drop in impurity.

**Setup — splitting a class of 10 students:**

| Group | Students | Entropy | Weight |
|---|---|---|---|
| Group A (Child 1) | 8 | 0.10 (nearly pure) | 8/10 = 0.8 |
| Group B (Child 2) | 2 | 1.00 (50/50 chaos) | 2/10 = 0.2 |

**Why not a plain average?**

$$\frac{0.10 + 1.00}{2} = 0.55 \quad \text{(wrong — ignores group sizes)}$$

**Correct — Weighted Conditional Entropy:**

$$H_{\text{split}} = (0.8 \times 0.10) + (0.2 \times 1.00) = 0.08 + 0.20 = 0.28$$

**Information Gain = Parent Entropy − Conditional Entropy**

**Key discovery:** Larger children carry more weight in the score. A chaotic small group barely hurts if the big group is clean.

---

## 10. Multi-Class Extension

When there are more than 2 classes (e.g. Cat, Dog, Rabbit):

- Binary `p` and `1−p` no longer work — you need `p_cat`, `p_dog`, `p_rabbit`
- **Gini** generalises to: `G = 1 − Σ pᵢ²` (probability of a wrong guess)
- **Entropy** generalises to: `H = −Σ pᵢ log(pᵢ)` (same formula, more terms)

**Splitting rule for continuous features:** evaluate thresholds at the *midpoints between sorted unique values*, because only these boundaries represent meaningful class transitions.

---

## 11. Regression Trees — When the Target is Continuous

**Problem:** Price values like `$120k`, `$130k`, `$180k`… are all unique. Gini and Entropy are meaningless here because there are no "classes" to count.

**Solution:** Replace impurity metric with **spread metric**.

### Required properties of a spread function:
- **Zero minimum** — if all values are identical, spread = 0
- **Monotonic** — as values spread apart, metric increases
- **Directional invariance** — positive and negative deviations count equally
- **Mean-centred** — measure distance from the node mean (minimises total squared error)

**Best function:** Sum of Squared Errors (SSE): `Σ(yᵢ − ȳ)²`

### Worked example (house prices):

| Step | Value |
|---|---|
| Parent mean (ȳ) | (120+130+140+280+290+300)/6 = **210** |
| Parent variance | 6466.67 — massive, flat $210k prediction for all |

**Split at Size < 1300:**

| Child | Data | Local mean | Variance |
|---|---|---|---|
| Left (small houses) | 120, 130, 140 | **130** | 66.67 |
| Right (large houses) | 280, 290, 300 | **290** | 66.67 |

**Weighted variance after split:** `(3/6 × 66.67) + (3/6 × 66.67) = 66.67`

**Variance Reduction:** `6466.67 − 66.67 = 6400` ✅ Massive improvement

### Classification vs. Regression — structural comparison:

| Attribute | Classification Tree | Regression Tree |
|---|---|---|
| Target type | Categorical (Yes/No) | Continuous (Price) |
| Leaf output | Majority vote (Mode) | Localised average (Mean) |
| Impurity metric | Entropy / Gini | Variance / MSE |
| Split criterion | Maximise Information Gain | Maximise Variance Reduction |

---

## 12. The Overfitting Problem — Why Unconstrained Trees Fail

An unconstrained tree splits until every leaf contains exactly **one sample**.

**Structurally:** With 1,000 rows → 1,000 leaf nodes. Each leaf isolates one data point.

**Mathematically:** The leaf prediction = that single row's value → SSE = 0. The algorithm believes it achieved perfect performance.

**In practice:** The tree has memorised the training data, not learned from it.

> **Analogy:** A security guard who memorises every employee's exact face and outfit. When John wears a different shirt tomorrow, he gets blocked at the door.

**Key discovery:** Zero training error = maximum overfitting. The tree cannot generalise to new data. This is the motivation for:
- **Max depth** limits
- **Min samples per leaf** thresholds
- **Pruning**
- **Ensemble methods** (Random Forest, Gradient Boosting)

---

## Summary — The Full Arc

```
Linear models fail on non-linear patterns
        ↓
Splits = nested if-else decisions that carve feature space
        ↓
Need a metric for "confusion" in each region
        ↓
Derived Gini Impurity: G = 2p(1−p)  [quadratic, geometric]
Derived Entropy: H = −Σpᵢlog(pᵢ)   [logarithmic, information-theoretic]
        ↓
Information Gain = parent impurity − weighted child impurity
        ↓
For regression: swap impurity → variance; swap IG → variance reduction
        ↓
Unconstrained growth → memorisation → overfitting → need regularisation
```

---

## 13. Multiclass Decision Trees

### 13.1 Why Binary Formulas Break

The binary Gini formula `G = 2p(1−p)` silently assumes there are only two classes: Yes and No. The moment a third class appears — say Cat, Dog, Rabbit — the formula breaks structurally, not just numerically.

**Why?** Because `p` was the probability of Yes, and `1−p` was the probability of No. That subtraction only works when the two probabilities must sum to 1. With three classes:

$$p_{\text{cat}} + p_{\text{dog}} + p_{\text{rabbit}} = 1$$

...but `1 − p_{\text{cat}}` is now the probability of *not-cat*, which lumps dogs and rabbits together. That collapses information we need.

**Key discovery:** Binary formulas are special cases that assume structure (exactly 2 classes) that doesn't generalise. We need to re-derive from first principles.

---

### 13.2 Re-deriving Gini for Multiple Classes

Return to the geometric intuition: Gini measures the probability of a wrong random guess.

**Setup:** Pick a random point from a node. Guess its class according to the node's class proportions. What is the probability you are wrong?

For class $i$ with probability $p_i$:
- Probability of picking class $i$: $p_i$
- Probability of guessing class $i$ (i.e., calling it that class): $p_i$
- Probability of being **correct** for class $i$: $p_i \times p_i = p_i^2$

Total probability of being **correct** across all classes:

$$P(\text{correct}) = \sum_i p_i^2$$

Therefore:

$$G = 1 - \sum_i p_i^2$$

**Verification — binary case:**

$$G = 1 - (p^2 + (1-p)^2) = 1 - p^2 - 1 + 2p - p^2 = 2p(1-p) \checkmark$$

The binary formula falls out as a special case. The multiclass formula is the true general form.

---

### 13.3 Interpretation of Multiclass Gini

| Scenario | Gini |
|---|---|
| All points are one class (pure node) | $1 - 1^2 = 0$ |
| Uniform over 2 classes (50/50) | $1 - (0.5^2 + 0.5^2) = 0.5$ |
| Uniform over 3 classes (33/33/33) | $1 - 3(0.33)^2 ≈ 0.67$ |
| Uniform over $k$ classes | $1 - k \cdot (1/k)^2 = 1 - 1/k$ |

**Key discovery:** Maximum Gini grows with the number of classes. A 10-class uniform node reaches Gini ≈ 0.9. This means Gini scores are not directly comparable across datasets with different numbers of classes — only relative comparisons within one dataset matter.

---

### 13.4 Entropy Naturally Generalises

Entropy required no structural changes. The formula:

$$H = -\sum_i p_i \log(p_i)$$

...already sums over *all* classes. For binary it happened to have two terms. For $k$ classes it has $k$ terms. Nothing breaks.

**Why?** Because entropy was derived from a property of information (additivity), not from an assumption about the number of outcomes. Logarithms don't care how many terms you sum.

**Verification — binary:**

$$H = -(p \log p + (1-p)\log(1-p)) \checkmark$$

**Key discovery:** Entropy's derivation was more principled, so it generalised for free. Gini needed re-derivation because it was built on binary geometry.

---

### 13.5 Why Threshold Search Doesn't Change

For continuous features, recall the rule: evaluate thresholds at midpoints between consecutive sorted values. This rule was never about binary vs. multiclass — it was about where *class transitions can occur*.

**Argument:** Between two adjacent sorted values with the same class, no threshold in that gap can change any split outcome. Only at the boundary between two points of *different* classes does a new threshold create a new partition. This logic is independent of how many classes there are.

The threshold search loop remains:
```python
for each feature f:
    sort unique values of f
    for each adjacent pair (v_i, v_{i+1}):
        threshold = (v_i + v_{i+1}) / 2
        compute weighted Gini or Entropy of the split
    pick the threshold with lowest weighted impurity
```

**Key discovery:** The search strategy is about feature structure, not label structure. Multiclass adds nothing to this step.

---

### 13.6 Complete Algorithm for Multiclass Trees

```
function BUILD_TREE(data, depth):
    if stopping_condition(data, depth):
        return Leaf(majority_class(data))

    best_gain = -∞
    for each feature f:
        for each threshold t (midpoints of sorted unique values):
            left, right = split(data, f, t)
            gain = impurity(data) - weighted_impurity(left, right)
            if gain > best_gain:
                best_gain = gain
                best_split = (f, t)

    left_data, right_data = split(data, best_split)
    return Node(
        feature   = best_split.feature,
        threshold = best_split.threshold,
        left      = BUILD_TREE(left_data,  depth+1),
        right     = BUILD_TREE(right_data, depth+1)
    )
```

Where `impurity` = multiclass Gini (`1 − Σpᵢ²`) or Entropy (`−Σpᵢ log pᵢ`), and `majority_class` returns the most frequent label in the leaf.

---

## 14. Regression Trees

### 14.1 Why Classification Impurity Fails

Suppose you are predicting house prices: `$120k, $130k, $180k, $290k`. Every value is unique. There are no "classes" to count.

If you tried to compute Gini:
- What is $p_{120k}$? It's 1/4. Same for all others.
- Gini = $1 − 4 \times (0.25)^2 = 0.75$

But this tells you nothing useful. Every leaf with unique values would report the same Gini regardless of how close or far apart those values are. `$120k, $121k, $122k, $123k` and `$1, $1000, $1000000, $10000000` would score identically.

**Key discovery:** Impurity measures count class frequencies. They are blind to the *distance* between numeric values. Regression needs a metric that measures how tightly clustered the values are.

---

### 14.2 What Should a Leaf Predict?

Before asking how to split, ask: given a set of numbers in a leaf, what single number do we predict?

**Candidates:**
- Mode (most frequent value) → useless when all values are unique
- Median → minimises absolute error, but hard to optimise recursively
- **Mean** → minimises squared error, algebraically tractable

We want the prediction $\hat{y}$ that minimises $\sum_i (y_i - \hat{y})^2$.

---

### 14.3 Deriving the Mean as the Optimal Prediction

Take the derivative and set to zero:

$$\frac{d}{d\hat{y}} \sum_i (y_i - \hat{y})^2 = -2 \sum_i (y_i - \hat{y}) = 0$$

$$\sum_i y_i - n\hat{y} = 0 \implies \hat{y} = \frac{1}{n}\sum_i y_i = \bar{y}$$

**Key discovery:** The mean is not a convention — it is the mathematically optimal prediction under squared error. Regression trees predict the mean of the training values that fall into each leaf.

---

### 14.4 Searching for a Numerical Impurity Measure

We need a function that answers: *"How spread out are the values in this node?"*

Required properties (analogous to the confusion properties for classification):

| Property | Classification | Regression |
|---|---|---|
| Zero minimum | Pure node (one class) → 0 | All identical values → 0 |
| Monotonic | More mixing → higher value | More spread → higher value |
| Directional invariance | Symmetric in Yes/No labels | Positive and negative deviations count equally |
| Reference point | Proportions relative to 1 | Distances relative to the mean |

The last property is crucial: why measure distance from the mean and not, say, from zero?

**Answer:** The mean minimises total squared distance (just proved above). Any other reference point produces a larger sum. Measuring from the mean makes the metric as tight as possible while still detecting spread.

---

### 14.5 Rediscovering Sum of Squared Errors (SSE)

The natural candidate satisfying all four properties:

$$\text{SSE} = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

| Property | SSE check |
|---|---|
| Zero minimum | If all $y_i = \bar{y}$, every term is 0 ✅ |
| Monotonic | Values far from mean → large $(y_i - \bar{y})^2$ ✅ |
| Directional invariance | Squaring removes sign ✅ |
| Mean-centred | $\bar{y}$ is the reference point ✅ |

**Key discovery:** SSE is the regression counterpart of Gini/Entropy. Splitting criterion: choose the split that maximally reduces SSE in the children.

---

### 14.6 Variance as Node Impurity

SSE has a problem: it scales with the number of samples. A node with 1,000 points will always have a larger SSE than a node with 10 points, even if the data is equally spread.

**Fix:** Normalise by $n$ to get variance:

$$\text{Var} = \frac{\text{SSE}}{n} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \bar{y})^2$$

Variance is size-invariant and comparable across nodes of different sizes. This is why some implementations report variance reduction rather than SSE reduction as the impurity metric.

In practice, both lead to identical split decisions — dividing both sides of the comparison by the same $n$ doesn't change which split wins. The choice of SSE vs. Variance is cosmetic for splitting, but Variance is more interpretable as a standalone score.

---

### 14.7 Variance Reduction (Regression Information Gain)

Directly analogous to Information Gain for classification:

$$\text{Variance Reduction} = \text{Var(parent)} - \left(\frac{n_L}{n} \cdot \text{Var}(L) + \frac{n_R}{n} \cdot \text{Var}(R)\right)$$

**Worked example (from Section 11):**

| | Value |
|---|---|
| Parent Var | 6466.67 |
| Left child (small houses) Var | 66.67 |
| Right child (large houses) Var | 66.67 |
| Weighted child Var | $(0.5 × 66.67) + (0.5 × 66.67) = 66.67$ |
| **Variance Reduction** | $6466.67 − 66.67 = \mathbf{6400}$ ✅ |

**Key discovery:** The numerical machinery is identical to classification. Only the impurity measure changes: Gini/Entropy → Variance/SSE.

---

### 14.8 Complete Regression Tree Algorithm

```
function BUILD_REGRESSION_TREE(data, depth):
    if stopping_condition(data, depth):
        return Leaf(mean(data.y))          # ← mean, not majority class

    best_reduction = -∞
    for each feature f:
        for each threshold t (midpoints of sorted unique values):
            left, right = split(data, f, t)
            reduction = variance(data.y)
                        - weighted_variance(left.y, right.y)
            if reduction > best_reduction:
                best_reduction = reduction
                best_split = (f, t)

    left_data, right_data = split(data, best_split)
    return Node(
        feature   = best_split.feature,
        threshold = best_split.threshold,
        left      = BUILD_REGRESSION_TREE(left_data,  depth+1),
        right     = BUILD_REGRESSION_TREE(right_data, depth+1)
    )
```

The only structural differences from the classification algorithm: `mean` instead of `majority_class` at leaves, and `variance` instead of `Gini/Entropy` as the metric.

---

## 15. Overfitting in Trees

### 15.1 Why Splitting Always Improves Training Error

This is a mathematical certainty, not an empirical observation.

**Proof sketch:** Any split of a node into two children cannot increase impurity. In the worst case, a split separates one point from all others. The isolated leaf has zero impurity. The remaining leaf has the same impurity as the parent minus the removed point. The weighted sum can only equal or decrease.

More formally: a single-point leaf always achieves SSE = 0 (regression) or Gini = 0 (classification, since the node is pure). You can always "improve" training error by adding more splits.

**Key discovery:** The training metric gives the algorithm zero signal to stop. Left to itself, the algorithm will always split deeper.

---

### 15.2 Memorization vs Learning

An unconstrained tree on $n$ unique training rows produces exactly $n$ leaves.

**Structure of the memorised tree:**
- Every leaf contains exactly one training sample
- Leaf prediction = that sample's exact label
- Training accuracy = 100%

**What the tree has actually learned:** A lookup table. Not a pattern.

```
Seen before:     Age=27, Salary=48000 → Class 1   (memorised)
New example:     Age=28, Salary=48000 → ???        (no rule exists)
```

The tree has no mechanism to generalise the age=27 leaf to age=28. It either falls into an adjacent leaf by accident or returns an arbitrary majority class.

**Key discovery:** Perfect training performance is not a success signal for trees — it is a failure mode.

---

### 15.3 Training Error vs Test Error

| Tree Depth | Training Error | Test Error |
|---|---|---|
| 1 (stump) | High | High (underfit) |
| Optimal depth | Low | Low |
| Maximum depth | 0% | High (overfit) |

The test error curve is U-shaped. Training error is monotonically decreasing. The gap between them is the *generalisation gap* — and an unconstrained tree maximises it.

**Analogy:** A student who memorises every question from last year's exam. On a new exam: they know the questions they've seen and fail every new one.

---

### 15.4 Bias–Variance Tradeoff

| Tree Type | Bias | Variance |
|---|---|---|
| Shallow tree (stump) | High — misses patterns | Low — stable across samples |
| Deep tree (fully grown) | Low — fits training data | High — changes wildly with new data |
| Optimal tree | Balanced | Balanced |

**Bias:** How wrong the model is on average, across all possible datasets.
**Variance:** How much the model's predictions fluctuate when trained on different samples.

A fully grown decision tree has nearly zero bias (it fits everything) but extreme variance: train it on a slightly different sample and you get a completely different tree structure.

**Key discovery:** This is the root motivation for ensemble methods. Random Forest reduces variance by averaging many high-variance trees. Gradient Boosting reduces bias by correcting errors sequentially.

---

### 15.5 Pre-Pruning

Pre-pruning stops the tree *before* it overfits by setting constraints during growth.

**Common hyperparameters:**

| Parameter | What it controls |
|---|---|
| `max_depth` | Maximum number of splits from root to any leaf |
| `min_samples_split` | Minimum samples in a node before it can be split |
| `min_samples_leaf` | Minimum samples that must remain in each child |
| `min_impurity_decrease` | Minimum gain required to justify a split |

**Mechanism:** At each candidate split, before actually splitting, check: does this split violate any constraint? If yes, make the current node a leaf.

**Tradeoff:** Pre-pruning is fast but greedy. A split that looks bad at depth 3 might enable very good splits at depth 4 and 5. By blocking it early, you lose those downstream gains.

**Key discovery:** Pre-pruning solves the symptom (too many splits) rather than the cause (no global view of what each split contributes). This motivated post-pruning.

---

### 15.6 Post-Pruning

Post-pruning grows the full tree first, then removes subtrees that don't earn their complexity.

**CART's approach — Cost Complexity Pruning:**

Define a penalised score for each subtree $T$:

$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

Where:
- $R(T)$ = total training error of subtree $T$
- $|T|$ = number of leaves in $T$ (complexity term)
- $\alpha$ ≥ 0 = regularisation hyperparameter (tuned via cross-validation)

**Intuition:** Every additional leaf must "pay" a penalty of $\alpha$. A leaf that reduces error by less than $\alpha$ is not worth keeping. Collapse it into its parent.

**Algorithm:**
1. Grow the full tree.
2. For each internal node, compute the gain in $R_\alpha$ if you collapse its subtree into a single leaf.
3. Collapse the node where the gain is least (weakest link).
4. Repeat until only the root remains.
5. Pick the tree along this sequence with lowest cross-validation error.

**Key discovery:** Post-pruning considers the global contribution of each split. A split that looked locally bad might be globally valuable — and post-pruning correctly preserves it.

---

### 15.7 Why CART Prefers Post-Pruning

CART (Classification and Regression Trees) — the algorithm behind scikit-learn's `DecisionTreeClassifier` — grows fully and then prunes, rather than stopping early.

**Reason 1 — Lookahead problem:** A split with zero immediate gain can enable highly pure children two levels down. Pre-pruning blocks this permanently. Post-pruning makes this decision after seeing the full subtree.

**Reason 2 — Global optimality:** Cost complexity pruning produces a sequence of trees ordered from most complex to simplest. You cross-validate over this sequence and pick the globally best tree. Pre-pruning searches a much smaller space (it commits early and never revisits).

**Reason 3 — Single tuning knob:** $\alpha$ is one parameter with a clean interpretation (cost per leaf). Pre-pruning requires tuning multiple thresholds simultaneously with less intuitive interactions.

**Key discovery:** Post-pruning is more expensive at training time but produces better generalisation. In practice the computational cost is acceptable because you only grow one full tree.

---

## 16. Evolution of Decision Tree Algorithms

### 16.1 Why ID3 Was Revolutionary

Before ID3 (Quinlan, 1986), decision tree construction was largely heuristic and non-systematic. ID3 introduced a principled, computable objective — Information Gain — and a recursive algorithm to maximise it.

The core idea: at every node, pick the feature that reduces entropy the most. Simple, elegant, and fast on categorical data.

**ID3 limitations:**
- Handles only categorical features (no continuous-variable splits)
- No pruning mechanism
- Cannot handle missing values
- Biased toward features with many categories (the hidden problem below)

---

### 16.2 The Hidden Bias of Information Gain

Consider splitting on two candidate features:

- **Outlook:** 3 values — Sunny, Cloudy, Rain
- **Student ID:** $n$ values — one per student (S001, S002, …, Sn)

Student ID will almost always produce the highest Information Gain. Why?

**Because splitting on Student ID creates $n$ leaves, each containing exactly one student.** Every leaf is perfectly pure. Information Gain = parent entropy − 0 = maximum possible value.

The algorithm "thinks" Student ID is a perfect predictor. In reality, it's a unique identifier — it will never generalise to a new student.

**Key discovery:** Information Gain is biased toward high-cardinality features. Any feature with many distinct values can achieve near-perfect gain simply by fragmenting the data, not by finding a meaningful pattern.

---

### 16.3 Student-ID Example

| Student | Outlook | Result |
|---|---|---|
| S001 | Sunny | Play |
| S002 | Rain | Don't Play |
| S003 | Cloudy | Play |
| S004 | Sunny | Don't Play |

Split by **Outlook (3 values):** 3 children, some mixed → moderate gain.

Split by **Student ID (4 values):** 4 children, each pure → gain = parent entropy − 0 = **maximum**.

ID3 would choose Student ID. This tree has 100% training accuracy and 0% generalisation ability.

**Diagnosis:** Gain rewards fragmentation. The more pieces you cut the data into, the purer each piece becomes — regardless of whether the feature is meaningful.

---

### 16.4 Deriving Split Information

The fix is to penalise features that fragment the data too aggressively. We need a measure of *how much a feature split fragmented the dataset*.

Apply the entropy formula to the split itself (treating the resulting partitions as "classes"):

$$\text{Split Information} = -\sum_{j=1}^{k} \frac{|S_j|}{|S|} \log\left(\frac{|S_j|}{|S|}\right)$$

Where $|S_j|$ is the size of child $j$ and $|S|$ is the total parent size.

**Intuition:** This is the entropy of the *partitioning itself*, not of the labels within partitions. A feature that splits into 100 equal-size groups has very high Split Information. A feature that puts 99% of data in one group and 1% in another has low Split Information.

**For Student ID on 4 students:** Each child has 1 of 4 students → Split Information = $-4 \times (0.25 \log 0.25) = 2.0$ (maximum).

**For Outlook (3 values, unevenly distributed):** Split Information would be lower because children are unequal in size.

---

### 16.5 Gain Ratio

Divide Information Gain by Split Information:

$$\text{Gain Ratio} = \frac{\text{Information Gain}}{\text{Split Information}}$$

**Effect:** Features that achieve high gain by extreme fragmentation are penalised because their Split Information is also high. Gain is only credited when it's *efficient* — not when it comes from simply splitting into many pieces.

| Feature | Gain | Split Info | Gain Ratio |
|---|---|---|---|
| Outlook | 0.25 | 1.58 | **0.158** |
| Student ID | 1.00 | 2.00 | **0.500** |

Wait — Student ID still wins here. But notice: these numbers are illustrative. In real datasets with meaningful features, Outlook-type splits have gain ratios much closer to Student-ID-type splits because the gain is real and the Split Information is moderate. The ratio removes the free ride from cardinality inflation.

**Key discovery:** Gain Ratio normalises gain by the cost of fragmentation. High fragmentation must be earned by high genuine gain.

---

### 16.6 C4.5

C4.5 (Quinlan, 1993) extended ID3 with:

| Feature | ID3 | C4.5 |
|---|---|---|
| Continuous features | ❌ Not supported | ✅ Midpoint threshold search |
| Split criterion | Information Gain | Gain Ratio |
| Missing values | ❌ Not handled | ✅ Probabilistic assignment |
| Pruning | ❌ None | ✅ Error-based post-pruning |
| Output | Classification only | Classification only |

**C4.5's pruning approach (error-based):** Each leaf has a training error rate. Using statistical confidence intervals (binomial distribution), estimate the *true* error rate with some confidence. If collapsing a subtree to a leaf has lower estimated true error than keeping it, prune.

This is different from CART's cost-complexity pruning — C4.5 prunes based on statistical upper bounds rather than regularisation penalties.

**Key discovery:** C4.5 solved the ID3 cardinality bias and added continuous feature support, making it the dominant practical algorithm through the 1990s.

---

### 16.7 Why CART Returned to Gini

CART (Breiman et al., 1984) uses Gini impurity instead of Entropy. This was a deliberate choice, not an oversight.

**Reason 1 — Computational cost:** Entropy requires computing logarithms; Gini requires only multiplication and subtraction. At scale, across thousands of candidate splits, this gap matters.

**Reason 2 — Empirical similarity:** Gini and Entropy produce nearly identical splits in practice. Both peak at 0.5 and hit 0 at purity. Their numerical values differ but their *ranking* of candidate splits is almost always the same.

**Reason 3 — Gain Ratio adds complexity without consistent benefit:** CART's designers found Gain Ratio to be less stable — it can penalise genuinely good splits if their Split Information happens to be large for legitimate reasons (e.g., a binary feature with a very unequal split). CART addresses the cardinality problem differently: it restricts splits to *binary* (always left and right, never $k$-way), which caps fragmentation structurally.

**Key discovery:** CART eliminates the cardinality bias by design — a feature with 100 categories still only produces a binary split. No normalisation required.

---

### 16.8 Comparing ID3, C4.5 and CART

| Attribute | ID3 | C4.5 | CART |
|---|---|---|---|
| Year | 1986 | 1993 | 1984 |
| Split criterion | Information Gain | Gain Ratio | Gini (classification), Variance (regression) |
| Split type | $k$-way (all categories) | $k$-way | Binary only |
| Continuous features | ❌ | ✅ | ✅ |
| Missing values | ❌ | ✅ Probabilistic | ✅ Surrogate splits |
| Pruning | ❌ | Error-based post | Cost-complexity post |
| Regression | ❌ | ❌ | ✅ |
| Multiclass | ✅ | ✅ | ✅ |
| Still used today | Rarely (educational) | Rarely | ✅ (scikit-learn default) |

**Key discovery:** Each algorithm solved a real problem: ID3 gave us the framework, C4.5 gave us continuous features and cardinality correction, CART gave us binary splits, regression, and a principled regularisation framework. Modern tree libraries implement CART.

---

## 17. Key Takeaways

### Classification vs Regression Trees

| Aspect | Classification | Regression |
|---|---|---|
| Leaf output | Majority class (mode) | Mean of values |
| Impurity metric | Gini or Entropy | Variance (SSE/n) |
| Split criterion | Maximise Information Gain | Maximise Variance Reduction |
| Stopping output | Most common label | Average label |
| Overfitting sign | 100% training accuracy | SSE = 0 on training set |

The tree-building algorithm is structurally identical. The only variable is what you optimise.

---

### Gini vs Entropy

| Attribute | Gini | Entropy |
|---|---|---|
| Formula | $1 - \sum p_i^2$ | $-\sum p_i \log p_i$ |
| Derivation | Geometric (probability of wrong guess) | Information-theoretic (average surprise) |
| Computation | Multiplication | Logarithm (slower) |
| Range (binary) | 0 to 0.5 | 0 to 1 |
| Split preference | Slight preference for purity | Slightly more sensitive to imbalance |
| Practical difference | Nearly identical split selections |  |
| Default in scikit-learn | ✅ `criterion="gini"` | `criterion="entropy"` |

**Rule of thumb:** Either works. Try both if tuning matters; default to Gini for speed.

---

### Information Gain vs Gain Ratio

| Attribute | Information Gain | Gain Ratio |
|---|---|---|
| Formula | $H(\text{parent}) - H_{\text{weighted}}(\text{children})$ | $\text{IG} / \text{SplitInfo}$ |
| Used by | ID3 | C4.5 |
| Bias | Favours high-cardinality features | Corrects for fragmentation |
| Stability | High | Lower (can penalise good splits) |
| CART's fix | Binary splits only — cardinality problem dissolves |  |

---

### SSE vs Variance Reduction

| Attribute | SSE | Variance Reduction |
|---|---|---|
| Formula | $\sum(y_i - \bar{y})^2$ | $\text{Var(parent)} - \text{weighted Var(children)}$ |
| Scale-dependent? | Yes — grows with $n$ | No — normalised by $n$ |
| Split decisions | Identical to Variance Reduction | Same as SSE |
| Interpretability | Raw error magnitude | Comparable across nodes |

Use Variance Reduction to compare nodes. Use SSE if you need absolute error values.

---

### Complete Decision Tree Building Pipeline

```
1. START
   └── Receive training data (X, y)

2. CHOOSE TASK
   ├── Classification → impurity = Gini or Entropy
   └── Regression     → impurity = Variance (SSE/n)

3. RECURSIVE SPLIT LOOP (at each node):
   ├── Check stopping condition
   │   ├── max_depth reached?
   │   ├── min_samples_split not met?
   │   ├── all labels identical?
   │   └── If any → return Leaf (mode for classification, mean for regression)
   │
   ├── For each feature f:
   │   └── For each midpoint threshold t:
   │       ├── Split data into left (≤ t) and right (> t)
   │       └── Compute gain = parent_impurity − weighted_child_impurity
   │
   ├── Select (f*, t*) with highest gain
   └── Recurse: build left subtree and right subtree

4. PRUNING (post-growth)
   └── CART: Cost Complexity Pruning
       ├── For each α: find weakest link subtrees
       ├── Generate sequence of nested trees T_max ⊃ T_1 ⊃ … ⊃ T_root
       └── Cross-validate to select best α

5. PREDICT
   ├── Classification: traverse tree, return majority class at leaf
   └── Regression:     traverse tree, return mean at leaf
```

---

### The Full Arc (Extended)

```
Linear models fail on non-linear patterns
        ↓
Splits = nested if-else decisions that carve feature space
        ↓
Need a metric for "confusion" in each region
        ↓
Derived Gini Impurity: G = 1 − Σpᵢ²  [geometric — probability of wrong guess]
Derived Entropy: H = −Σpᵢlog(pᵢ)     [information-theoretic — average surprise]
        ↓
Information Gain = parent impurity − weighted child impurity
        ↓
For regression: swap impurity → variance; swap IG → variance reduction
        ↓
Multiclass: binary formulas generalise — Gini via re-derivation, Entropy for free
        ↓
Unconstrained growth → memorisation → overfitting
        ↓
Pre-pruning: stop early (fast, local, greedy)
Post-pruning: grow then trim (global, preferred by CART)
        ↓
Algorithm evolution: ID3 → C4.5 (Gain Ratio, continuous features)
                            → CART (binary splits, regression, cost-complexity pruning)
        ↓
Modern practice: CART (scikit-learn default)
Ensembles next: Random Forest (↓ variance), Gradient Boosting (↓ bias)
```
