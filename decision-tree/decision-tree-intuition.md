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
