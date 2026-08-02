# Why Optuna Exists — From Guesswork to Smart Search

> *A ground-up story of hyperparameter tuning: every dead-end, every breakthrough, and exactly why Optuna was born.*

---

## The Problem in One Line

Training a model is easy. **Choosing the right settings for it is not.**

These settings — `max_depth`, `n_estimators`, `learning_rate` — are called **hyperparameters**. You don't learn them from data. You have to *search* for them yourself.

The search is expensive. Every guess means training a full model and measuring its score. With 10 parameters and 10 values each, that's **10 billion combinations**. You need a smarter strategy than trying them all.

This notebook is the story of every strategy humans tried — and why most of them failed.

---

## Chapter 1 — The Stone Age: Manual Tuning

Before any automation existed, the search was done by hand.

```python
depths = [2, 3, 4, 5, 6, 7, 8]

for depth in depths:
    model = RandomForestClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Depth: {depth} | Score: {score}")
```

**The problem?** This is only *one* parameter. In real projects you have dozens.  
Two parameters with 7 values each = 49 combinations.  
Three parameters = 343 combinations.  
Ten parameters = millions of nested loops, days of waiting, and memory errors.

**Intuition only gets you so far.** At some point you're just guessing.

---

## Chapter 2 — Grid Search: The First Real Attempt

The first systematic fix: build a grid of all combinations and test every single point.

```
n_estimators: [50, 100, 125]
max_depth:    [3, 5, None]

Grid (all combos):
  (50, 3)   (50, 5)   (50, None)
 (100, 3)  (100, 5)  (100, None)
 (125, 3)  (125, 5)  (125, None)
```

With 3-fold cross-validation, that's **9 configs × 3 folds = 27 model trainings** — just for 2 parameters.

The evaluation function is the core of the whole system:

```python
def evaluate(algorithm, params, X_data, y_data):
    """
    The black box: f(hyperparameters) → validation score.
    The model has no idea it's being searched.
    """
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in kf.split(X_data):
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        model = algorithm(**params, random_state=42)
        model.fit(X_train, y_train)
        fold_scores.append(model.score(X_val, y_val))

    return np.mean(fold_scores)
```

Using `itertools.product` makes the grid loop clean:

```python
from itertools import product

for values in product(*param_grid.values()):
    current_params = dict(zip(param_grid.keys(), values))
    score = evaluate(RandomForestClassifier, current_params, X_train_val, y_train_val)
```

### Grid Search vs Manual Tuning

| Feature | Manual Loops | Grid Search |
|---|---|---|
| Code length | 30+ lines of nested loops | 3–4 clean lines |
| Bug-prone? | Very (easy to leak data) | No (battle-tested) |
| Speed | Single CPU, sequential | All CPUs via `n_jobs=-1` |
| Re-trains best model? | You must do it manually | Automatic (`refit=True`) |

### The Fatal Flaw

Grid Search has **zero memory**. It treats every configuration as if it's the first one it's ever seen. It doesn't care that the first 20 experiments all failed at the same depth. It ploughs forward regardless.

> Imagine testing 100 recipes and throwing away all your cooking notes between each one.

---

## Chapter 3 — Random Search: Surprisingly Better

Instead of marching through every grid point, Random Search **throws random darts** at the parameter space. Sounds worse. Is often better.

### Why? The Beach Analogy

Picture a 100m × 100m beach. Treasure is buried along a *single narrow line* — only one coordinate (dimension) really matters.

**Grid Search** walks in rigid rows, repeating the same X-values over and over:
```
□  □  □  □  □  □   ← same X positions, repeated for each row
□  □  □  □  □  □
□  □  □  □  □  □
```

**Random Search** drops random pins everywhere:
```
•      •       •    ← every pin hits a unique X position
•       •
•       •      •
      •       •
```

When only a few parameters actually matter (which is usually true), Random Search covers the important dimension far more efficiently.

### The Remaining Problem

But Random Search still doesn't *learn*. Trial 1 finds a great region. Trial 2 ignores that completely and guesses somewhere random again.

> Why keep throwing darts randomly when you already know where the bullseye is?

---

## Chapter 4 — Bayesian Optimization: Search That Learns

The core idea: **use past results to decide where to look next**.

```
Unknown Reality
      │
      ▼  (expensive to evaluate every point)
Real Objective: f(hyperparameters) → CV Score
      ▲
      │   (we only observe a few points)
      │
Build Surrogate Model  ──→  Suggest Next Experiment
(our current belief)                │
      ▲                             │
      └─────────── observe result ──┘
```

Instead of training your real model for every guess, you build a **fast, cheap mathematical stand-in** (surrogate) that *estimates* what the score would be — and importantly, tracks *how uncertain* it is about each region.

### Gaussian Processes (GP): The First Surrogate

A GP builds a continuous map of the entire search space. Near points you've tested, it's confident. Far from tested points, it admits uncertainty.

> Like predicting traffic: high confidence on busy highways you know well, low confidence on mountain roads you've never driven.

### Exploration vs Exploitation

Once you have a map of confidence, you face a dilemma:

- **Exploit**: Go where you *know* the score is high. Safe, but misses hidden peaks.
- **Explore**: Go where you're *uncertain*. Risky, but might find something better.

**UCB (Upper Confidence Bound)** formula:

```
Score = μ(x) + κ × σ(x)
```

Where `μ` = predicted score, `σ` = uncertainty, `κ` = how adventurous you want to be.

**Example — tuning `max_depth`:**

| Candidate | μ (predicted score) | σ (uncertainty) | κ=2 Score |
|---|---|---|---|
| Depth = 5 *(tested nearby)* | 91% | 1% | 91 + 2×1 = **93** |
| Depth = 20 *(never tested)* | 84% | 8% | 84 + 2×8 = **100** ✓ |

Even though Depth 5 looks safer, the algorithm picks Depth 20 — the huge uncertainty means it *might* be a goldmine. If it turns out to be terrible, uncertainty there drops to zero and the algorithm looks elsewhere.

### Why GP Failed in Practice

| Problem | What Happens |
|---|---|
| O(N³) compute cost | At 1,000+ trials, calculating the next step costs more than training the model |
| Curse of dimensionality | Breaks down past ~3 parameters |
| Categorical parameters | Can't calculate distance between `"gbtree"` and `"dart"` |
| Conditional parameters | If `booster="gblinear"`, `max_depth` doesn't exist — GP can't handle this |

---

## Chapter 5 — TPE: The Practical Breakthrough

**Tree-structured Parzen Estimator (TPE)** — Optuna's default engine.

Instead of modelling the whole function, TPE asks a simpler question:  
**"Does this new candidate look like a winner or a loser?"**

### The Cricket Scout Analogy

You're building a cricket team. You have records of 100 players.

**Step 1 — Split them:**
- Top 20 (superstars): the *Good Group*
- Bottom 80 (underperformers): the *Bad Group*

**Step 2 — Look at their habits:**
- Good players: mostly practice 4 hours/day
- Bad players: scattered — some 1 hour, some 12 hours (overtraining)

**Step 3 — Score a new candidate:**
- Candidate A (practices 1 hr): common among losers, rare among winners → **Low Score**
- Candidate B (practices 4 hrs): rare among losers, common among winners → **High Score** ✓

**Step 4 — Recruit B, test them, add the result, repeat.**

TPE never tries to predict the *exact* score. It just compares probability densities: `p(x | good) / p(x | bad)`. Higher ratio = more likely to succeed.

### GP vs TPE Side by Side

| | Gaussian Process | TPE |
|---|---|---|
| Models | The whole objective function | Good vs bad regions |
| Needs | Covariance matrices | Probability densities |
| Compute cost | Blows up with many trials | Scales well |
| Parameters | Continuous spaces only | Continuous, integer, categorical, conditional |
| Used by | Classical Bayesian opt | **Optuna (default)** |

---

## Chapter 6 — Building Optuna From Scratch

Now the full picture assembles into three clean classes.

### `RandomSampler` — The dumb baseline

```python
class RandomSampler:
    def sample_float(self, name, low, high, history):
        return random.uniform(low, high)  # Ignores history entirely
```

### `MiniTrial` — One experiment container

```python
class MiniTrial:
    def __init__(self, sampler, history):
        self.sampler = sampler   # The strategy for picking values
        self.history = history   # Everything tried so far
        self.params  = {}        # What THIS trial chose

    def suggest_float(self, name, low, high):
        value = self.sampler.sample_float(name, low, high, self.history)
        self.params[name] = value
        return value
```

### `MiniStudy` — The experiment loop

```python
class MiniStudy:
    def __init__(self, sampler):
        self.sampler     = sampler
        self.history     = []           # All (params, score) pairs
        self.best_score  = float("-inf")
        self.best_params = None

    def optimize(self, objective, n_trials):
        for _ in range(n_trials):
            trial = MiniTrial(self.sampler, self.history)
            score = objective(trial)                       # Run the user's function
            self.history.append((trial.params, score))    # Record result
            if score > self.best_score:
                self.best_score  = score
                self.best_params = trial.params
```

### `MiniTPESampler` — The smart brain

```python
class MiniTPESampler:
    def sample_float(self, name, low, high, history):
        # Phase 1: not enough data yet → explore randomly
        if len(history) < 5:
            return random.uniform(low, high)

        # Phase 2: enough data → exploit what we've learned
        sorted_history = sorted(history, key=lambda x: x[1], reverse=True)

        cutoff      = max(1, len(sorted_history) // 5)      # Top 20%
        good_trials = sorted_history[:cutoff]

        good_values = [t[0][name] for t in good_trials if name in t[0]]
        centre      = sum(good_values) / len(good_values)   # Average of winners

        value = random.gauss(centre, 0.4)                   # Sample near the winner zone
        return max(low, min(high, value))                    # Keep within bounds
```

### How a Single TPE Trial Flows

```
[ 10 Past Trials from MiniStudy ]
             │
             ▼
     [ 1. Sort by Score ]
   (highest → lowest)
             │
             ▼
    [ 2. Slice Top 20% ]
   (the "good group")
             │
             ▼
   [ 3. Find the Centre ]
   (average x of winners)
             │
             ▼
  [ 4. Gaussian Sampling ]
  (draw near that centre)
             │
             ▼
  [ 5. Clip to Bounds ]
  (stay within low…high)
             │
             ▼
  [ Next Smart Suggestion ]
```

**Concrete example:**

History: `[(x=1.2, score=4), (x=3.4, score=9), (x=5.1, score=2), (x=3.6, score=10), (x=0.5, score=1)]`

- Sort → best is `x=3.6`
- Top 20% → `[x=3.6]`
- Centre = `3.6`
- Gaussian around 3.6 → next try is `3.58` or `3.71`, almost never `1.0`

---

## The Full Timeline

```
Manual Tuning
    → slow, error-prone, doesn't scale

Grid Search
    → systematic, but no memory, wastes trials on bad regions

Random Search
    → surprisingly better when few params matter, still learns nothing

Bayesian Optimization (Gaussian Process)
    → learns from history, but breaks under high dimensions and categories

TPE (Optuna's default)
    → learns from history, scales, handles every param type
```

---

## Quick Reference

| Term | Plain English |
|---|---|
| Hyperparameter | A model setting you choose before training (e.g. `max_depth`) |
| Objective function | The black box: takes params, returns a score |
| Surrogate model | A cheap fake that estimates the real score |
| Trial | One experiment (one set of params → one score) |
| Study | The full search: many trials, tracks the best |
| TPE | Optuna's smart sampler — learns from past winners |
| Exploitation | Go near the best known region |
| Exploration | Try new, uncertain regions |
| UCB | Formula balancing exploration and exploitation: `μ + κσ` |
| Warm-up phase | First few random trials to gather baseline data before TPE kicks in |
