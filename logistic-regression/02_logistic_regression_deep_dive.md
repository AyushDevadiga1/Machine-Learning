# Logistic Regression — Deep Dive

*Companion to `01_linear_regression_deep_dive.md`. Same methodology: first-principles derivation, Socratic, worked numerical examples, connected back to prior math wherever it overlaps.*

---

## Roadmap

```
Stage 1 → Why Linear Regression breaks for classification
          (motivation, probability constraint, intro to odds/logit)
Stage 2 → The Logistic (Sigmoid) Model
          (hypothesis function, decision boundary, interpretation)
Stage 3 → Loss Function Derivation
          (Bernoulli likelihood → MLE → Binary Cross-Entropy;
           connects to MLE work from Linear Regression)
Stage 4 → Optimization
          (gradient of BCE, Gradient Descent update rule;
           connects to GD variants from Linear Regression — no closed form here)
Stage 5 → Assumptions
          (linearity in log-odds, independence, no multicollinearity,
           large sample size, no strong outliers)
Stage 6 → Diagnostics & Evaluation
          (confusion matrix, precision/recall, ROC-AUC, deviance,
           Hosmer-Lemeshow, pseudo-R², VIF)
Stage 7 → Extensions
          (regularized logistic regression, multiclass — softmax vs OvR,
           class imbalance handling)
```

Status: **Stage 3 complete → Stage 4 in progress**

---

## Stage 1: Why Linear Regression Breaks for Classification

**Setup.** Tumor size (x) vs malignant (y ∈ {0,1}), fit by OLS same as Linear Regression track.

**Failure 1 — Range violation.**
OLS fit: $\hat y = -0.2775 + 0.1866x$. At $x=1$: $\hat y = -0.09$; at $x=8$: $\hat y = 1.22$.
Violates Kolmogorov's first axiom ($0 \le P(A) \le 1$) — OLS has no mechanism to bound its output, since it's just minimizing squared distance in $\mathbb{R}$.

**Failure 2 — Boundary instability from unbounded/symmetric penalty.**
Adding one distant, unambiguous, *correctly-classified* point (x=20, y=1) shifted the fitted line enough to flip a previously-correct point (x=5) across the decision boundary.
Mechanism: MSE's only zero-loss state is *exact numerical equality* to the label. It is symmetric and unbounded — it cannot distinguish "confidently correct, far past the threshold" from "wrong by the same margin." It optimizes for **point-wise numerical closeness, not decision-boundary correctness**. (⇒ motivates cross-entropy in Stage 3, which saturates near the correct label instead of demanding exact equality.)

**Failure 3 — Step function isn't trainable by gradient descent.**
A literal step function has derivative 0 almost everywhere (no signal regardless of error magnitude) and is discontinuous at the threshold (undefined signal there). Need a function that is: bounded in (0,1), monotonic, and *differentiable everywhere*.

---

## Stage 2: The Logistic (Sigmoid) Model

**Derivation path:** odds → log-odds (logit) → invert to get sigmoid.

- Odds: $q = \frac{p}{1-p}$, range $(0,\infty)$ — bounded below, not symmetric ($p=0.9\to q=9$; $p=0.1\to q=0.111$).
- Log-odds (logit): $\log(q)$, range $(-\infty,\infty)$, additively symmetric ($\log 9 = 2.197$, $\log(1/9)=-2.197$).
- Modeling assumption: $\log\left(\frac{p}{1-p}\right) = \beta_0+\beta_1x = z$.
- Solve for $p$: $p = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}} = \sigma(z)$ — the **sigmoid**.
- Property: $\sigma(-z) = 1-\sigma(z)$ (derived algebraically; needed for Bernoulli likelihood in Stage 3, where $P(y=0|x)=1-\sigma(z)=\sigma(-z)$).
- $\sigma(0)=0.5$, so decision boundary ($p=0.5$) occurs exactly at $z=0 \Rightarrow \beta_0+\beta_1x=0 \Rightarrow x^*=-\beta_0/\beta_1$.
- **Key insight:** sigmoid is monotonically increasing, so $\sigma(z)\ge 0.5 \iff z\ge 0$. Classification depends only on the *sign* of $z$, which is linear in $x$ (a hyperplane in general). This is why logistic regression is a **linear classifier** despite $\sigma$ being nonlinear — swapping in a different monotonic squashing function would not move the decision boundary, only reshape how confidence/probability varies with distance from it.

---

## Stage 3: Loss Function Derivation (MLE on Bernoulli)

**Same procedure as Linear Regression's MLE derivation, different distribution.** Linear Regression assumed $\epsilon\sim\mathcal N(0,\sigma^2)$ and showed maximizing Gaussian log-likelihood ⟺ minimizing SSE. Here, $y\in\{0,1\}$ is Bernoulli, not Gaussian.

- Bernoulli pmf for one point: $P(y\mid p) = p^y(1-p)^{1-y}$ (verified: collapses to $p$ when $y=1$, $1-p$ when $y=0$).
- Substitute model: $p_i = \sigma(z_i)$. Joint likelihood over $n$ iid points: $L(\beta_0,\beta_1) = \prod_{i=1}^n \sigma(z_i)^{y_i}(1-\sigma(z_i))^{1-y_i}$.
- Raw product underflows numerically for large $n$ (e.g. $0.96^{10000}\approx 5\times10^{-178}$) → take log (strictly monotonic, so $\arg\max L = \arg\max \log L$):
$$\ell(\beta_0,\beta_1) = \sum_{i=1}^n\Big[y_i\log(\sigma(z_i))+(1-y_i)\log(1-\sigma(z_i))\Big]$$
- Worked numerical example ($\beta_0=-5,\beta_1=1.2$; points $x=2,y=0$; $x=5,y=1$; $x=9,y=1$): $z_i = -2.6, 1.0, 5.8$; $p_i=0.0691,0.7311,0.9970$; per-point log-lik terms $=-0.0716,-0.3133,-0.0030$; $\ell = -0.3879$.
- Flip sign (turn maximize into minimize) and average (keeps gradient magnitude independent of $n$, same reason SSE→MSE):
$$L(\beta_0,\beta_1) = -\frac{1}{n}\sum_{i=1}^n\Big[y_i\log(p_i)+(1-y_i)\log(1-p_i)\Big]$$
This is **Binary Cross-Entropy (log loss)**. For the worked example: $L = 0.3879/3 = 0.1293$.

---

## Stage 4: Optimization — Gradient of BCE, Gradient Descent

*(notes to be filled in as we work through it)*

---
