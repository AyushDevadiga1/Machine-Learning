# Decision Trees: A Comprehensive Intuition Guide

This document captures the complete intuition behind decision trees, from the fundamental "why" to the mathematical derivations of Gini impurity and entropy, including the Shannon story, weighted averages, and multi-class extensions.

---

## 1. The Fundamental Question: Why Decision Trees?

### Linear Classifiers Hit a Wall

Consider a dataset where a single line can separate the two classes:

| Age | Salary | Bought |
|-----|--------|--------|
| 18  | 20000  | 0      |
| 22  | 23000  | 0      |
| 28  | 40000  | 1      |
| 35  | 60000  | 1      |
| 45  | 90000  | 1      |

**Observation:** A simple rule like `Age > 25` perfectly separates the classes. Linear models (logistic regression, perceptrons) can handle this easily.

**But real data is often messier.** Consider:

| Age | Salary | Bought |
|-----|--------|--------|
| 20  | 20000  | 0      |
| 20  | 80000  | 1      |
| 50  | 20000  | 1      |
| 50  | 80000  | 0      |

**Problem:** No single line can separate these points. No matter how you rotate or shift it, some points will always be on the wrong side.

### The Solution: Divide and Conquer

Instead of one equation, we use **nested decisions**:

```python
if age < 30:
    if salary < 50000:
        predict 0
    else:
        predict 1
else:
    if salary < 50000:
        predict 1
    else:
        predict 0
```

Each condition is a **split**—it divides the space into smaller regions. Each region then becomes easier to classify.

This is exactly the **divide and conquer** strategy: split the data, solve each part recursively.

---

## 2. Desired Property of a Good Split

A good split should make the child nodes **simpler** (more pure) than the parent.

**Imagine:** Parent has 50 Yes and 50 No — maximum confusion. After a split:
- Left child: 48 No, 2 Yes — 96% No, only 4% Yes
- Right child: 2 No, 48 Yes — 96% Yes, only 4% No

Each child is **much easier** to classify. The uncertainty dropped drastically.

**Key insight:** Everything reduces to one central question:

> **What is a good mathematical definition of "confusion" (or "impurity") in a set of labels?**

If we can measure impurity, we can:
1. Measure impurity before a split.
2. Measure impurity after a split.
3. Choose the split that reduces impurity the most.
4. Repeat recursively until the data is pure enough.

---

## 3. Deriving the Confusion Function

### Requirements for a Binary Impurity Function

Let $p = P(\text{Yes})$ and $1-p = P(\text{No})$. Our confusion function $C(p)$ must satisfy:

| Property | Meaning |
|----------|---------|
| $C(0) = 0$ | All No → no confusion |
| $C(1) = 0$ | All Yes → no confusion |
| Maximum at $p = 0.5$ | Equal split → highest uncertainty |
| Symmetry | $C(p) = C(1-p)$ — swapping labels doesn't change uncertainty |

### Candidate 1: Linear Functions Fail
- $C(p) = p$: at $p=1$, $C=1$ (wrong — should be 0)
- $C(p) = 1-p$: at $p=0$, $C=1$ (wrong — should be 0)

### Candidate 2: Quadratic (Gini)

Try $C(p) = p(1-p)$:

| $p$ | $p(1-p)$ |
|-----|----------|
| 0   | 0        |
| 0.1 | 0.09     |
| 0.2 | 0.16     |
| 0.3 | 0.21     |
| 0.4 | 0.24     |
| 0.5 | 0.25 (max) |
| 0.6 | 0.24     |
| 0.7 | 0.21     |
| 0.8 | 0.16     |
| 0.9 | 0.09     |
| 1   | 0        |

**It satisfies all properties!** To scale the maximum to 1, multiply by 4:

$$
C(p) = 4p(1-p) \quad \text{or more commonly} \quad G(p) = 2p(1-p)
$$

This is the **Gini impurity** (for binary classification).

### For Multi-class ($C$ classes)

The probability of guessing correctly (if you guess randomly according to the distribution) is:

$$
\sum_{i=1}^{C} p_i^2
$$

Thus, the probability of being wrong (impurity) is:

$$
G = 1 - \sum_{i=1}^{C} p_i^2
$$

For binary: $G = 1 - (p^2 + (1-p)^2) = 2p(1-p)$ — exactly our derived formula.

**What does this number mean?**  
Example: 90% Yes, 10% No → $G = 2(0.9)(0.1) = 0.18$. It simply says: "This node is less pure than a pure node, but purer than a 50/50 split."

---

## 4. The Entropy Alternative: Shannon's Information Theory

Gini asks: *"How mixed is this node?"*  
Entropy asks: *"How much information is still missing before I know the class?"*

Both are valid; they are just different perspectives on the same problem.

### Act I: The Million-Dollar Whisper (1948, Bell Labs)

Claude Shannon asked: **"What is information?"** Engineers thought it was the number of letters. Shannon disagreed — information is about **surprise**.

- If I say, "The sun will rise tomorrow," you learn nothing. You already knew that. Surprise = 0.
- If I say, "You just won a billion dollars," you are shocked. Surprise is enormous.

**Rule:** The rarer an event, the more information it contains.

If $p$ is the probability, the surprise should be:
- 0 when $p=1$ (certain)
- Large when $p \to 0$ (rare)

### Act II: The Mathematical Marriage

If you flip two independent coins, the total information should add:

$$
I(\text{Coin1}) + I(\text{Coin2})
$$

But the probabilities multiply:

$$
p(\text{Coin1}) \times p(\text{Coin2})
$$

We need a function that turns multiplication into addition:

$$
I(p \times q) = I(p) + I(q)
$$

**The only function that does this is the logarithm:**

$$
I(p) = -\log(p)
$$

Test it: $-\log(1) = 0$ — no surprise. Perfect.

### Act III: From Individual Surprise to Average Surprise (Entropy)

Now, we are building a decision tree. We have a node with 80% Yes, 20% No. We want the **average surprise** of picking a random sample from this node.

In probability, the average (expected value) is:

$$
\text{Average Surprise} = \sum_{\text{outcomes}} P(\text{outcome}) \times \text{Surprise}(\text{outcome})
$$

Substituting $I(p) = -\log_2(p)$:

$$
H = -\sum_{i=1}^{C} p_i \log_2(p_i)
$$

This is **Entropy**. For binary:

$$
H = -p\log_2(p) - (1-p)\log_2(1-p)
$$

Entropy is also maximized at $p=0.5$ (value = 1) and zero at purity.

---

## 5. Evaluating a Split: The Weighted Average

A split produces multiple child nodes. To compute the overall impurity after the split, we take a **weighted average**, where the weight is the proportion of samples that fall into each child.

### Example: A Classroom of 10 Students

- **Parent:** 10 students, some pass, some fail.
- **Split on Study Habit:**
  - **Group A:** 8 students (weight = 0.8) — very pure, entropy $H_A = 0.10$
  - **Group B:** 2 students (weight = 0.2) — totally mixed, entropy $H_B = 1.00$

If we took a simple average: $\frac{0.10 + 1.00}{2} = 0.55$.  
This is wrong — it gives equal importance to the tiny chaotic group and the large pure group.

Instead, we must weight by group size:

$$
\text{Conditional Entropy} = (0.8 \times 0.10) + (0.2 \times 1.00) = 0.08 + 0.20 = 0.28
$$

Because Group A represents 80% of the data, its low entropy dominates, proving this split is effective.

**General formula:**

$$
\text{Impurity}_{\text{after}} = \sum_{k} w_k \cdot I(\text{child}_k)
$$

where $w_k = \frac{\text{size of child}_k}{\text{size of parent}}$.

The **gain** is:

$$
\text{Gain} = \text{Impurity}_{\text{before}} - \text{Impurity}_{\text{after}}
$$

We choose the split with the largest gain.

---

## 6. Worked Example: Gini vs Entropy Side-by-Side

**Dataset:**

| Sample | Feature X | Target Y |
|--------|-----------|----------|
| 1      | 1         | 1        |
| 2      | 1         | 1        |
| 3      | 0         | 1        |
| 4      | 0         | 0        |

### Parent Impurities

- **Entropy:**  
  $p_1 = 3/4 = 0.75$, $p_0 = 0.25$  
  $H = -0.75\log_2(0.75) - 0.25\log_2(0.25) \approx 0.8113$

- **Gini:**  
  $G = 2 \cdot 0.75 \cdot 0.25 = 0.375$

### Split on X

- **Child 1 ($X=1$):** Two samples, both Y=1 — pure.  
  $H_1 = 0$, $G_1 = 0$

- **Child 2 ($X=0$):** One Y=1, one Y=0 — equally mixed.  
  $H_2 = 1$, $G_2 = 0.5$

- **Weights:** $w_1 = 2/4 = 0.5$, $w_2 = 2/4 = 0.5$

### Weighted After-Split Impurities

- **Entropy after:** $0.5 \times 0 + 0.5 \times 1 = 0.5$
- **Gini after:** $0.5 \times 0 + 0.5 \times 0.5 = 0.25$

### Gains

- **Information Gain:** $0.8113 - 0.5 = 0.3113$
- **Gini Gain:** $0.375 - 0.25 = 0.125$

Both metrics agree that this split provides value, though they quantify it differently.

---

## 7. Generalising to $C$ Classes

When you have more than two classes (e.g., Cat, Dog, Rabbit), the binary formulas no longer work because $p$ and $1-p$ only describe two classes.

For $C$ classes, with probabilities $p_1, p_2, \dots, p_C$ (where $\sum p_i = 1$):

| Metric | Formula |
|--------|---------|
| **Gini** | $G = 1 - \sum_{i=1}^{C} p_i^2$ |
| **Entropy** | $H = -\sum_{i=1}^{C} p_i \log_2(p_i)$ |

**Interpretation for Gini (multi-class):**  
If you randomly guess a class (according to the distribution), the probability of guessing correctly is $\sum p_i^2$. So $1 - \sum p_i^2$ is the probability of guessing **incorrectly** — i.e., the impurity.

For binary, $\sum p_i^2 = p^2 + (1-p)^2$, so $G = 1 - [p^2 + (1-p)^2] = 2p(1-p)$. Perfect match.

---

## 8. Gini vs Entropy: Which to Use?

| Aspect | Gini | Entropy |
|--------|------|---------|
| **Computation** | Faster (no logarithms) | Slightly slower |
| **Sensitivity** | Tends to favor the majority class | More balanced splits |
| **Performance** | Similar in practice | Similar in practice |
| **Common in** | CART (Classification and Regression Trees) | ID3, C4.5 |

In most cases, the difference is negligible. Choose based on your implementation or preference.

---

## 9. Extension: Regression Trees

Decision trees are not limited to classification. For **regression** (continuous target), the impurity measure is **variance** (or mean squared error).

For a node with target values $y_1, \dots, y_n$:

$$
\text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

A split that minimizes the weighted variance of the children is chosen.

$$
\text{Variance}_{\text{after}} = \sum_{k} w_k \cdot \text{Var}(\text{child}_k)
$$

The **variance reduction** is the gain. The same recursive partitioning logic applies.

---

## 10. Visual Summary of the Splitting Process

Starting from a dataset, we:

1. **Measure parent impurity** (Gini or Entropy or Variance).
2. **For each feature and each possible split value:**
   - Split the data into two children.
   - Compute each child's impurity.
   - Compute the weighted average impurity after the split.
   - Compute the gain.
3. **Select the split** with the highest gain.
4. **Repeat** recursively on each child.
5. **Stop** when a node is pure, reaches a maximum depth, or contains too few samples.

The result is a **tree** of decisions that can handle complex, non-linear relationships with clear interpretability.

---

## 11. The Big Picture

Decision trees embody a simple but powerful idea: **break a hard problem into smaller, easier problems**. By measuring "confusion" with Gini or "missing information" with Entropy, we can objectively decide which splits are best. Both metrics emerge from deep mathematical foundations—Gini from probability theory, Entropy from information theory—yet both converge on the same practical goal: building interpretable, effective models for both classification and regression.

