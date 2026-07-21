
---

# The Birth of Logistic Regression

## Stage 0 — The Problem

Imagine you are a doctor. Patients come to you with one key feature: the size of their tumor.

| Tumor Size | Malignant? |
| --- | --- |
| 1 | No |
| 2 | No |
| 3 | No |
| 5 | Yes |
| 6 | Yes |
| 8 | Yes |

**Question:**

> Can we build a machine that predicts whether a tumor is malignant?

Notice something crucial. The target we want to predict is binary:

* `0` $\rightarrow$ Benign
* `1` $\rightarrow$ Malignant

This is **not** a traditional regression problem; it is a **classification** problem. But suppose we don't know classification exists yet. What would we naturally try first?

---


## Stage 1 — Let's Try Linear Regression

We already know Linear Regression, so let's try to fit a straight line to this data:

$$\hat{y} = \beta_0 + \beta_1 x$$

After training our model, suppose we get the following equation:

$$\hat{y} = -0.2775 + 0.1866x$$

Now let's try to predict outcomes using this model:

* For $x = 1$, we obtain: **$-0.09$**

> *What does a probability of $-9\%$ even mean? Probabilities cannot be negative.*

* For $x = 8$, we obtain: **$1.22$**

> *What is a $122\%$ probability? That is mathematically impossible.*

### First Conclusion

Linear Regression predicts continuous values on the range $(-\infty, \infty)$, but probabilities must strictly lie within the interval $[0, 1]$.

We need a mathematical function that maps **any real number** to a **valid probability**.

---

## Stage 2 — What Should That Function Look Like?

Before searching for a complex formula, let's think about the behavior we want:

* **If the model outputs a massive negative number (e.g., $-100$):**
The probability should not be negative. It should mean *"Almost certainly class 0"* (approaching `0`).
* **If the model outputs a massive positive number (e.g., $+100$):**
The probability should not exceed 1. It should mean *"Almost certainly class 1"* (approaching `1`).
* **If the model outputs exactly $0$:**
The model isn't leaning toward either class. The most reasonable probability is `0.5` (complete uncertainty).

Therefore, our ideal function $f(z)$ must satisfy:

* As $z \to -\infty$, then $f(z) \to 0$
* If $z = 0$, then $f(z) = 0.5$
* As $z \to +\infty$, then $f(z) \to 1$

---

## Stage 3 — Why Not Use a Step Function?

A step function fits these boundaries perfectly:

$$\begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \ge 0 \end{cases}$$

This looks ideal at first glance, until we try to train the model.

Gradient Descent updates weights using derivatives: *"How should I change the weights to reduce error?"* Because a step function is completely flat everywhere except at zero, **its derivative is $0$ almost everywhere**. Gradient Descent receives a learning signal of $0$, effectively telling it: *"Don't move,"* even when the predictions are completely wrong. The model cannot learn.

### Second Conclusion

We need a function that is:

* **Smooth** and **differentiable** (so we can calculate gradients).
* **Bounded** strictly between $0$ and $1$.
* **Monotonic** (strictly increasing).

---

## Stage 4 — Where Does the Sigmoid Come From?

Instead of arbitrarily inventing a curve, can we derive one logically?

Suppose our model predicts a probability $p$. Because probabilities are bounded and linear equations are not, we cannot set them equal directly. We need to model a related quantity that is unbounded.

---

## Stage 5 — Odds

Suppose the probability of an event happening is $p = 0.8$. The probability of it not happening is $1 - p = 0.2$.

**Question:** How much more likely is Class 1 than Class 0?

$$\text{Odds} = \frac{p}{1-p} = \frac{0.8}{0.2} = 4$$

> This means Class 1 is **four times** as likely to occur as Class 0.

However, we still have a problem. The range of odds is $[0, \infty)$. While we solved the negative boundary, a linear equation can still output negative numbers, which odds cannot handle.

---

## Stage 6 — Log-Odds

To remove the lower boundary, we take the natural logarithm of the odds:

$$\log(\text{Odds}) = \log\left(\frac{p}{1-p}\right)$$

By taking the log, our range expands beautifully from $[0, \infty)$ to $(-\infty, \infty)$.

Now, we can finally set up our linear equation:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x$$

This is the **fundamental assumption of Logistic Regression**.

> We do not assume that probability is linear. We assume that the **log-odds** are linear.

---

## Stage 7 — Recover the Probability

We ultimately want to predict the actual probability $p$, not the log-odds. Let's solve for $p$.

Let $z = \beta_0 + \beta_1 x$.

$$\log\left(\frac{p}{1-p}\right) = z$$

Exponentiate both sides:

$$\frac{p}{1-p} = e^z$$

Rearranging the terms to solve for $p$:

$$p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}$$

This is the **sigmoid function**. Notice that we did not invent it out of thin air; it emerged naturally from the algebra of log-odds.

---

## Stage 8 — How Do We Train It?

Now we have a model that predicts probabilities. How do we find the optimal weights $\beta_0$ and $\beta_1$?

Our first instinct might be to use Mean Squared Error (MSE), which worked perfectly for Linear Regression.

---

## Stage 9 — Why MSE Fails Here

If we plot the squared error $(y - p)^2$, it seems fine initially. Correct predictions yield low loss, and incorrect predictions yield high loss.

However, let's look at the gradient of the loss with respect to our input $z$:

$$\frac{\partial L}{\partial z} \propto (y - p) \cdot p(1 - p)$$

This reveals a major flaw. If our model predicts $p = 0.001$ for a true label of $y = 1$, the model is confidently wrong. The error is massive, but because $p$ is so close to $0$, the term $p(1-p)$ becomes incredibly small (nearly $0$).

This term **kills the gradient**. The model learns the slowest when it is making its worst mistakes.

---

## Stage 10 — Think Probabilistically

Instead of asking, *"How far is my prediction from the target line?"* we must ask:

> *"How probable is the observed dataset given the parameters of my model?"*

This shift in perspective takes us from geometry to statistics.

---

## Stage 11 — Likelihood

For a single data point, the probability of observing the label $y$ (which can only be $0$ or $1$) is:

$$P(y\vert{}x) = p^y(1-p)^{1-y}$$

* If $y = 1$, the expression simplifies to $p$.
* If $y = 0$, the expression simplifies to $1-p$.

To calculate how likely the **entire dataset** is, we multiply the individual probabilities together (assuming the data points are independent):

$$L = \prod_{i} p_i^{y_i}(1-p_i)^{1-y_i}$$

This is the **Likelihood**. We want to find the weights that maximize this value.

---

## Stage 12 — Log-Likelihood

Multiplying thousands of probabilities together results in numbers so small that computers suffer from numerical underflow. To prevent this, we take the natural logarithm to convert the products into sums:

$$\log L = \sum_{i} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

---

## Stage 13 — Cross-Entropy

While statisticians prefer to *maximize* log-likelihood, optimization algorithms in machine learning are built to *minimize* loss.

To convert our maximization problem into a minimization problem, we simply multiply the log-likelihood by $-1$.

$$J = -\sum_{i} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

This is **Binary Cross-Entropy Loss**.

---

## Stage 14 — Deriving the Gradients

To train our model using Gradient Descent, we need the gradients of the loss with respect to our parameters.

Our computational graph flows as follows:

```text
x
│
▼
z = wᵀx + b
│
▼
p = σ(z)
│
▼
Cross-Entropy Loss

```

Rather than differentiating the loss directly with respect to the weights, we apply the **Chain Rule**.

---

### Step 1 — Gradient of the Loss with Respect to the Probability

Starting from the Binary Cross-Entropy loss:

$$L = -\left[y\log(p) + (1-y)\log(1-p)\right]$$

Differentiating with respect to $p$:

$$\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}$$

Taking a common denominator:

$$\frac{\partial L}{\partial p} = \frac{p-y}{p(1-p)}$$

---

### Step 2 — Derivative of the Sigmoid

Recall the sigmoid function:

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Differentiating with respect to $z$ yields:

$$\frac{\partial p}{\partial z} = p(1-p)$$

> *This elegant result appears repeatedly throughout machine learning.*

---

### Step 3 — The Beautiful Cancellation

Applying the chain rule:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z}$$

Substituting our previous calculations into the equation:

$$\frac{\partial L}{\partial z} = \frac{p-y}{p(1-p)} \cdot p(1-p)$$

The denominator and numerator cancel perfectly:

$$\frac{\partial L}{\partial z} = p-y$$

> *This is one of the most elegant derivations in Logistic Regression.*

---

### Why This Is Better Than MSE

Using MSE with a sigmoid gives:

$$\frac{\partial L}{\partial z} = (p-y)p(1-p)$$

The extra term $p(1-p)$ shrinks the gradient whenever the sigmoid saturates near `0` or `1`.

> *As a result, the model learns the slowest when it is most confidently wrong.*

Cross-Entropy removes this problem naturally. The sigmoid derivative is **not removed manually**. Instead, it is cancelled mathematically by the derivative of the logarithm.

The final learning signal becomes:

$$\frac{\partial L}{\partial z} = p-y$$

which remains large whenever the prediction is far from the true label.

---

### Step 4 — Gradients with Respect to the Parameters

Since our linear baseline equation is defined as:

$$z = w^Tx + b$$

The chain rule yields the following results:

* **For the bias:**

$$\frac{\partial L}{\partial b} = p-y$$



because $\frac{\partial z}{\partial b} = 1$.
* **For the weights:**

$$\frac{\partial L}{\partial w} = (p-y)x$$



because $\frac{\partial z}{\partial w} = x$.
* **For multiple features:**

$$\nabla_w L = (p-y)x$$



where $x$ is now the feature vector.

---

## Stage 15 — The Final Insight

Notice something remarkable when looking at the parameter derivatives side-by-side.

* For **Linear Regression:**

$$\frac{\partial L}{\partial w} = (\hat{y}-y)x$$


* For **Logistic Regression:**

$$\frac{\partial L}{\partial w} = (p-y)x$$



The optimization algorithm structure is almost identical! The only difference is **what the model predicts**:

* Linear Regression predicts a continuous value $\hat{y}$.
* Logistic Regression predicts a probability $p$.

Everything else—gradient descent routines, parameter updates, and optimization pathways—follows the exact same underlying principles.

---

## Logistic Regression Cheat Sheet

### Forward Pass

$$z = w^Tx + b$$

$$p = \sigma(z) = \frac{1}{1+e^{-z}}$$

---

### Loss

$$L = -\left[y\log(p) + (1-y)\log(1-p)\right]$$

---

### Gradients

$$\frac{\partial L}{\partial z} = p-y$$

$$\frac{\partial L}{\partial w} = (p-y)x$$

$$\frac{\partial L}{\partial b} = p-y$$

---

### Gradient Descent Update

$$w \leftarrow w - \eta\frac{\partial L}{\partial w}$$

$$b \leftarrow b - \eta\frac{\partial L}{\partial b}$$

---

# Assumptions of Logistic Regression

## Stage 16 — What Did We Actually Assume?

Now that the full model is derived and trained, it is worth stepping back and asking: what did we silently commit to along the way?

Every modelling decision we made — the Bernoulli distribution, the log-odds linearity, the likelihood product — carried an assumption with it. Violating any of them breaks the mathematics in a precise, traceable way.

---

## Stage 17 — Assumption 1: Bernoulli Output (and Its Variance Consequence)

The very first commitment we made was:

$$y \mid x \sim \text{Bernoulli}\big(p(x)\big)$$

This single choice has an immediate, unavoidable mathematical consequence for variance. For a Bernoulli random variable, the variance is not a free parameter — it is entirely determined by the mean:

$$\text{Var}(y \mid x) = p(x)\big(1 - p(x)\big)$$

Because $p$ changes with every value of $x$, the variance is never constant. It is small when $p$ is near $0$ or $1$ (the model is confident), and it peaks at $0.25$ exactly when $p = 0.5$ (maximum uncertainty).

### Third Conclusion

> In Linear Regression, constant variance (Homoscedasticity) must be explicitly assumed as an extra condition. In Logistic Regression, it is never assumed at all. The Bernoulli distribution forces the variance to be heteroscedastic by design — not as a flaw, but as a mathematical consequence of the output being binary.

---

## Stage 18 — Assumption 2: Linearity in the Log-Odds

This assumption is the direct consequence of the derivation in Stage 6. When we set the log-odds equal to the linear equation:

$$\log\left(\frac{p(x)}{1-p(x)}\right) = \beta_0 + \beta_1 x$$

we committed to a specific structural claim: for every one-unit increase in $x$, the log-odds shift by a constant amount $\beta_1$.

Notice what this does *not* say. The relationship between $x$ and the raw probability $p(x)$ is a curve — the sigmoid. We are not assuming that probability is linear in $x$. We are only assuming that the **log-odds** are.

### Fourth Conclusion

> Logistic Regression does not need $p$ to be linear in $x$. It only requires the logit, $\log\!\left(\frac{p}{1-p}\right)$, to be linear. This is not an extra condition bolted on from outside — it is the exact constraint that gave rise to the sigmoid function in the first place.

---

## Stage 19 — Assumption 3: Independence of Observations

When we derived the joint likelihood in Stage 11 by multiplying individual probabilities:

$$L(\beta) = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}$$

we used a fundamental rule of probability: you can only multiply individual probabilities to obtain a joint probability when the events are **independent**.

$$P(A \text{ and } B) = P(A) \times P(B) \quad \text{only if } A \perp B$$

In Logistic Regression, unlike Linear Regression, there is no explicit error term $\epsilon$. So the independence assumption shifts from errors to outcomes directly: we require that the observed labels $y_i$ and $y_j$ are completely independent of each other, conditional on the predictors.

**When this breaks:** If you collect multiple tumor biopsies from the same patient, those data points are clustered and correlated. Treating them as independent artificially inflates your effective sample size, shrinks standard errors, and invalidates the entire MLE engine.

### Fifth Conclusion

> The product structure of the likelihood is not just notation — it is a mathematical statement that every observation is independent. Violating this assumption does not produce a slightly wrong answer; it breaks the foundation the likelihood was built on.

---

## Stage 20 — Assumption 4: No Multicollinearity

This assumption carries over from Linear Regression completely unchanged, because the problem lives in the predictor matrix $X$, not in the output $y$.

At the heart of the MLE optimization is the computation of coefficient variances, which requires inverting the matrix $(X^T W X)$. When two predictors are highly correlated — say, tumor weight in grams and tumor weight in ounces — the columns of $X$ become nearly linearly dependent, and the inversion becomes numerically unstable.

The consequences cascade:

* The standard errors of $\beta$ coefficients skyrocket.
* Coefficients become wildly unstable — a small change in the data can flip a $\beta$ from $+10$ to $-10$.
* $p$-values explode, making it impossible to determine which predictors are genuinely driving the prediction.

### Sixth Conclusion

> Multicollinearity is an $X$-problem. It does not care whether your output is continuous or binary. As long as your model computes $(X^T W X)^{-1}$, collinear features will break the math in exactly the same way they did in Linear Regression.

---

## Stage 21 — Assumption 5: Gaussian Errors? They Disappear.

In Linear Regression, we explicitly assumed $\epsilon \sim \mathcal{N}(0, \sigma^2)$. This was necessary to make $t$-tests and confidence intervals exact for small sample sizes.

In Logistic Regression, this assumption **does not exist**.

Once we chose $y_i \sim \text{Bernoulli}(p_i)$, the residual for any data point is:

$$\text{residual}_i = y_i - p_i$$

Because $y_i$ can only ever be exactly $0$ or exactly $1$, the residuals can only ever land on two discrete values:

* If $y_i = 1$: the residual is $(1 - p_i)$
* If $y_i = 0$: the residual is $(-p_i)$

It is physically impossible for two discrete values to form a continuous, bell-shaped Gaussian curve.

### Seventh Conclusion

> We do not need to check for normally distributed residuals in Logistic Regression. MLE trades small-sample exactness for large-sample asymptotic normality — as $n$ grows, the Central Limit Theorem guarantees that the $\hat{\beta}$ estimates themselves become approximately normal, regardless of the binary shape of $y$.

---

## Assumptions Cheat Sheet

| # | Assumption | Status vs. Linear Regression |
|---|---|---|
| 1 | $y \mid x \sim \text{Bernoulli}(p(x))$ — variance is $p(1-p)$ | **Replaces** Homoscedasticity |
| 2 | Log-odds are linear in $x$ | **New** — replaces linearity in $y$ |
| 3 | Observations are independent (conditional on $x$) | **Carried over** — shifts from errors to outcomes |
| 4 | No multicollinearity among predictors | **Carried over** unchanged |
| 5 | Gaussian errors | **Dropped entirely** |