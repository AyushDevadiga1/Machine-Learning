# The Birth of Decision Trees

---

# Stage 0 — The Problem

Suppose we again have a dataset.

| Age | Bought Laptop? |
|------|----------------|
|18|No|
|20|No|
|23|Yes|
|25|Yes|
|40|Yes|
|45|No|

Our goal remains the same:

> Can we build a machine that predicts the class?

Unlike Logistic Regression, we do **not** assume any mathematical relationship between the features and the output.

Instead, we want the machine to ask questions.

---

# Stage 1 — A Human Thinks Like This

Imagine you are classifying fruits.

You naturally ask questions like:

```
Is it red?
        │
   Yes ─┴─ No
   │         │
Apple?    Banana?
```

Each answer reduces uncertainty.

Decision Trees simply automate this questioning process.

---

# Stage 2 — Which Question Should We Ask First?

Suppose we have many possible questions.

• Age > 20?
• Age > 30?
• Income > 50k?
• Student?

Clearly, not every question is equally useful.

Some questions almost determine the answer immediately.

Others barely help.

So we need a way to measure

> **How good is a split?**

---

# Stage 3 — Measuring Confusion

Imagine a node containing

```
Yes
Yes
Yes
No
Yes
```

Mostly "Yes".

We are already fairly certain.

Now consider

```
Yes
No
Yes
No
Yes
No
```

This node is completely mixed.

We have no confidence.

Therefore,

Decision Trees require a mathematical measure of

> **Confusion (Impurity)**

---

# Stage 4 — Properties of a Good Confusion Function

We reasoned that a confusion function should satisfy:

### Pure Node

```
Yes
Yes
Yes
Yes
```

Confusion = 0

because there is nothing left to learn.

---

### Maximum Confusion

```
Yes
No
Yes
No
```

Confusion should be maximum.

This is the hardest possible node.

---

### Symmetry

50% Yes
50% No

should have the same confusion as

50% No
50% Yes

The labels shouldn't matter.

Only the proportions should.

---

### Smoothness

A tiny change in probabilities

should produce a tiny change in confusion.

---

# Stage 5 — First Candidate : Gini Impurity

Suppose

Probability of Yes = p

Probability of No = 1-p

Randomly pick one sample.

Probability it is classified correctly

```
p² + (1-p)²
```

Therefore,

Probability of being wrong

```
1 - [p² + (1-p)²]
```

Simplifying,

```
Gini = 2p(1-p)
```

This satisfies every property we wanted.

---

# Stage 6 — Understanding Gini

Pure Node

```
p = 1

Gini = 0
```

Maximum confusion

```
p = 0.5

Gini = 0.5
```

The graph forms an upside-down parabola.

Exactly what intuition suggested.

---

# Stage 7 — Can We Do Better?

Gini measures

> "How likely am I to misclassify a random sample?"

But perhaps there is another way to think about confusion.

Instead of asking

> "How often will I be wrong?"

we ask

> "How much information do I still need?"

This leads us to Information Theory.

---

# Stage 8 — The Idea of Information

Imagine someone tells you

"The sun will rise tomorrow."

Did you learn much?

Not really.

Now imagine someone tells you

"You won the lottery."

That contains enormous information.

Observation:

Rare events carry more information than common events.

Therefore,

Information should increase as probability decreases.

---

# Stage 9 — Requirements of an Information Function

Our information measure should satisfy:

• Certain events carry zero information.

• Rare events carry large information.

• Independent events should have additive information.

These requirements naturally lead to

```
Information(x)

=
-log₂(p)
```

---

# Stage 10 — Expected Information

A node contains multiple possible outcomes.

Average information becomes

```
Entropy

=

Σ p log₂(1/p)
```

or

```
Entropy

=

-Σ p log₂(p)
```

This measures

> Average uncertainty.

---

# Stage 11 — Understanding Entropy

Pure Node

```
Entropy = 0
```

Maximum uncertainty

```
Entropy = 1
```

Entropy and Gini both measure confusion,

but from completely different viewpoints.

• Gini → Probability of mistake

• Entropy → Average missing information

---

# Stage 12 — Choosing the Best Split

Suppose a split divides the parent into two children.

Good splits greatly reduce confusion.

Bad splits barely change it.

Therefore

```
Information Gain

=

Parent Entropy

-

Weighted Child Entropy
```

The best split

is simply the one with

> Maximum Information Gain.

---

# Stage 13 — Growing the Tree

The algorithm now becomes surprisingly simple.

Repeat:

1. Compute impurity.
2. Try every possible split.
3. Compute Information Gain.
4. Choose the best split.
5. Create children.
6. Repeat recursively.

Stop when

• Node is pure.
• Maximum depth reached.
• Too few samples remain.
• Gain becomes negligible.

---

# Decision Tree Cheat Sheet

### Goal

Reduce uncertainty by asking questions.

---

### Gini

```
1 - Σp²
```

Binary

```
2p(1-p)
```

Measures

> Probability of misclassification.

---

### Information

```
-log₂(p)
```

Measures

> Surprise of an event.

---

### Entropy

```
-Σp log₂(p)
```

Measures

> Average uncertainty.

---

### Information Gain

```
Parent Entropy

-

Weighted Child Entropy
```

Measures

> Reduction in uncertainty after a split.

---

### Training Algorithm

Repeat

```
Find Best Split

↓

Split Dataset

↓

Recurse
```

until stopping criteria are met.