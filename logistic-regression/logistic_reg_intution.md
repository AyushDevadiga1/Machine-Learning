
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

We arrived here logically:

* We did not select it arbitrarily because a textbook told us to.
* We derived it step-by-step because it is the mathematically natural way to measure error when modeling probabilities.