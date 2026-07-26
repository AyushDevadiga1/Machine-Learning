# Polars 2026 — Cheat Sheet
`import polars as pl`

---

## COLUMN 1 — I/O & Memory Management
> Core Mental Shift: *Build a plan first. Pay RAM only at .collect().*

---

### Eager Read — Data enters RAM immediately

```python
df = pl.read_csv("train.csv")

df = pl.read_parquet(
    "train.parquet",
    columns=["Age", "Fare"]
)

df = pl.DataFrame({
    "Age":  [22, 38, 26],
    "Fare": [7.25, 71.28, 7.92],
})
```
▸ Use for small datasets or when you need immediate access.

---

### Lazy Scan — Zero RAM until .collect()

```python
lf = pl.scan_csv("train.csv")

lf = pl.scan_parquet("train.parquet")
```
▸ Returns a `LazyFrame`. Nothing is loaded.
▸ Parquet enables true column-level skipping.
▸ CSV still parses all bytes; drops unwanted cols after.

---

### .collect() — The Execution Trigger

```python
df = (
    pl.scan_csv("train.csv")
    .filter(pl.col("Age") > 18)
    .select(["Age", "Fare"])
    .collect()
)
```
▸ Optimizer fires here — predicate + projection pushdown applied before any data moves.

---

### Optimizer Internals (Automatic)

**Predicate Pushdown**
```python
# Rows failing filter NEVER enter RAM
pl.scan_csv("train.csv")
  .filter(pl.col("Age") > 18)
```

**Projection Pushdown**
```python
# Only named cols are parsed
pl.scan_csv("train.csv")
  .select(["Age", "Fare"])
```

**No Arbitrary Reorder** — Optimizer only reorders when semantics are provably identical.
```python
# SAFE: pure filter, no dependencies
.filter(pl.col("Age") > 18)

# BLOCKED: filter cannot move
# past a column mutation
.with_columns(
    (pl.col("Age") * 2).alias("Age")
)
```

**CSE (Common Sub-expression)**
```python
# mean().over("Pclass") runs ONCE
# despite appearing twice in one block
df.with_columns(
    pl.col("Fare")
      .mean()
      .over("Pclass")
      .alias("Avg"),
    (
      pl.col("Fare") >
      pl.col("Fare")
        .mean()
        .over("Pclass")
    ).alias("AboveAvg"),
)
```

---

### Row Count After Each Operation

| Operation          | Rows After         |
|--------------------|--------------------|
| `filter()`         | Decreases          |
| `select()`         | Same, fewer cols   |
| `with_columns()`   | Same + new cols    |
| `group_by().agg()` | 1 row per group    |
| `.over()`          | Same, enriched     |
| `sort()`           | Same, reordered    |
| `join("inner")`    | ≤ left rows        |
| `join("left")`     | = left rows        |

---

## COLUMN 2 — Row Filtering
> Core Mental Shift: *pl.col() is an Expression — not a boolean array.*

---

### pl.col() — Expression, Not a Mask

```python
# This is an Expression recipe:
pl.col("Sex")

# Pandas equivalent (DO NOT use):
# df["Sex"] == "female"  ← boolean array
# df[df["Sex"] == "female"]  ← not valid
```
▸ Expressions are evaluated lazily inside `.filter()`, `.select()`, `.with_columns()`.

---

### Single Condition

```python
df.filter(pl.col("Age") > 18)

df.filter(pl.col("Sex") == "female")

df.filter(pl.col("Pclass").is_in([1, 2]))

df.filter(pl.col("Cabin").is_not_null())
```

---

### Multiple Conditions — AND via Comma

```python
# Comma = implicit AND
df.filter(
    pl.col("Sex") == "female",
    pl.col("Survived") == 1,
    pl.col("Pclass") == 1,
)
```
▸ All conditions must be True. No `&` operator needed.

---

### Multiple Conditions — OR via Pipe

```python
df.filter(
    (pl.col("Pclass") == 1) |
    (pl.col("Fare") > 100)
)
```
▸ Each OR clause must be wrapped in parentheses.

---

### Combined AND + OR

```python
df.filter(
    (pl.col("Sex") == "female") &
    (
        (pl.col("Pclass") == 1) |
        (pl.col("Age") < 18)
    )
)
```

---

### Lazy Filter — Predicate Pushdown

```python
(
    pl.scan_csv("train.csv")
    .filter(pl.col("Age") > 18)
    .filter(pl.col("Fare") > 50)
    .select(["Age", "Fare"])
    .collect()
)
```
▸ Chained filters are fused into one pass.
▸ Failing rows never enter RAM.

---

### Useful Filter Expressions

```python
# Null checks
pl.col("Age").is_null()
pl.col("Age").is_not_null()

# String matching
pl.col("Name").str.contains("Mr.")

# Membership
pl.col("Pclass").is_in([1, 2])

# Range
pl.col("Age").is_between(18, 60)

# Negation
~pl.col("IsChild")
```

---

### select() — Column Subsetting

```python
# By name list
df.select(["Age", "Fare", "Survived"])

# By regex pattern
df.select(pl.col("^.*Fare.*$"))

# Exclude specific columns
df.select(
    pl.exclude("Name", "Ticket", "Cabin")
)

# Compute and return one column
df.select(
    pl.col("Survived").sum()
)
```
▸ Same rows, controlled columns.

---

## COLUMN 3 — Parallel Column Creation
> Core Mental Shift: *One with_columns() block = one parallel snapshot. All expressions see the same original state.*

---

### Add a Single Column

```python
df.with_columns(
    (pl.col("SibSp") + pl.col("Parch") + 1)
    .alias("FamilySize")
)
```
▸ `.alias()` is mandatory — gives the result a column name.

---

### Add Multiple Columns — One Parallel Pass

```python
df.with_columns(
    (
        pl.col("SibSp") +
        pl.col("Parch") +
        1
    ).alias("FamilySize"),

    (pl.col("Age") < 18)
    .alias("IsChild"),

    (pl.col("Fare") > 100)
    .alias("HighFare"),
)
```
▸ All expressions execute in parallel on the **same original DataFrame**.
▸ `IsChild` sees the original `Age`, not a mutated version.

---

### Modify an Existing Column In-Place

```python
# Overwrite "Age" by reusing its name as alias
df.with_columns(
    (pl.col("Age") + 1).alias("Age")
)
```

---

### Snapshot Isolation — Critical Rule

```python
# WRONG EXPECTATION:
# You might expect "Adult" to use
# the incremented Age — it won't.
df.with_columns(
    (pl.col("Age") + 1).alias("Age"),
    (pl.col("Age") > 18).alias("Adult"),
    # ↑ reads ORIGINAL Age, not Age+1
)

# CORRECT: chain a second block
df.with_columns(
    (pl.col("Age") + 1).alias("Age")
).with_columns(
    (pl.col("Age") > 18).alias("Adult")
)
```

---

### Inline Derived Column — Avoid Chaining

```python
# Instead of chaining for FarePerPerson,
# write the formula directly:
df.with_columns(
    (
        pl.col("SibSp") +
        pl.col("Parch") +
        1
    ).alias("FamilySize"),

    (pl.col("Age") < 18)
    .alias("IsChild"),

    (
        pl.col("Fare") / (
            pl.col("SibSp") +
            pl.col("Parch") +
            1
        )
    ).alias("FarePerPerson"),
)
```
▸ Polars evaluates the sub-expression once internally.

---

### Missing Data — In-Column Handling

```python
df.with_columns(
    pl.col("Age")
    .fill_null(29.7)
    .alias("Age"),

    pl.col("Cabin")
    .fill_null("Unknown")
    .alias("Cabin"),

    pl.col("Age")
    .is_null()
    .alias("AgeMissing"),
)

# Forward-fill strategy
df.fill_null(strategy="forward")
# Other strategies:
# "backward", "mean", "zero", "one"

# Drop rows with any null
df.drop_nulls()
df.drop_nulls(subset=["Age", "Fare"])
```

---

### String Operations Inside with_columns

```python
df.with_columns(
    pl.col("Name")
    .str.to_lowercase()
    .alias("name_lower"),

    pl.col("Name")
    .str.len_chars()
    .alias("name_length"),

    pl.col("Name")
    .str.contains("Mr.")
    .alias("is_mr"),

    pl.col("Sex")
    .str.to_uppercase()
    .alias("Sex_upper"),
)
```

---

## COLUMN 4 — Aggregations & Window Functions
> Core Mental Shift: *group_by().agg() collapses rows. .over() enriches without collapsing.*

---

### group_by().agg() — Single Aggregation

```python
df.group_by("Pclass").agg(
    pl.col("Fare").mean()
)
```
▸ Always use `group_by()` — never `groupby()` (deprecated).

---

### group_by().agg() — Multiple Aggregations

```python
df.group_by("Pclass").agg(
    pl.col("Fare").mean(),
    pl.col("Fare")
      .max()
      .alias("FareMax"),
    pl.col("Age")
      .mean()
      .alias("AvgAge"),
    pl.len(),
    pl.col("Survived").sum(),
    pl.col("Age")
      .count()
      .alias("AgeNonNull"),
)
```
▸ `pl.len()` — total row count in the group.
▸ `pl.col('x').count()` — non-null count of column `x`.
▸ Must alias when same column used twice.

---

### All Aggregation Methods

```python
.mean()        # arithmetic mean
.sum()         # total
.min()         # minimum value
.max()         # maximum value
.std()         # standard deviation
.var()         # variance
.median()      # 50th percentile
.first()       # first row value
.last()        # last row value
.n_unique()    # distinct count
.count()       # non-null count
.quantile(0.75)# 75th percentile
pl.len()       # total group row count
```

---

### Aggregate + Join Back to Original Rows

```python
stats = df.group_by("Pclass").agg(
    pl.col("Fare")
      .mean()
      .alias("ClassAvgFare"),
    pl.col("Fare")
      .max()
      .alias("ClassMaxFare"),
)

df.join(stats, on="Pclass", how="left")
```
▸ Merges summary back onto each original row.

---

### .over() — Window Function (No Collapse)

```python
df.with_columns(
    pl.col("Fare")
      .mean()
      .over("Pclass")
      .alias("ClassAvgFare")
)
```
▸ 891 rows in → 891 rows out.
▸ Each row gets its group's computed value attached.
▸ Equivalent to SQL `AVG(Fare) OVER (PARTITION BY Pclass)`.

---

### .over() — Deviation from Group Mean

```python
df.with_columns(
    (
        pl.col("Fare") -
        pl.col("Fare")
          .mean()
          .over("Pclass")
    ).alias("FareDiff"),
)
```

---

### .over() — Full Feature Engineering Block

```python
df.with_columns(
    pl.col("Fare")
      .mean()
      .over("Pclass")
      .alias("ClassAvgFare"),

    (
        pl.col("Fare") -
        pl.col("Fare")
          .mean()
          .over("Pclass")
    ).alias("FareDiff"),

    (
        pl.col("Fare") >
        pl.col("Fare")
          .mean()
          .over("Pclass")
    ).alias("AboveClassAvg"),
)
```
▸ CSE: `mean().over("Pclass")` computed once.

---

### sort() — Ordering Rows

```python
# Single key
df.sort("Fare", descending=True)

# Multi-key tiebreaker
df.sort(
    ["Fare", "Age"],
    descending=[True, False],
)
# Same Fare → sort Age ascending
```

---

### rank() — Ranking Within Groups

```python
df.with_columns(
    pl.col("Fare")
      .rank(method="dense", descending=True)
      .alias("OverallFareRank"),

    pl.col("Fare")
      .rank(method="dense", descending=True)
      .over("Pclass")
      .alias("ClassFareRank"),

    (
        pl.col("Fare")
        .rank(method="dense", descending=True)
        .over("Pclass") <= 3
    ).alias("IsTop3InClass"),
)
```

**Rank Methods:**
```python
# dense:   100,100,80 → 1,1,2
# min:     100,100,80 → 1,1,3
# ordinal: 100,100,80 → 1,2,3
```

---

### Joins — All Types

```python
# Inner — matching rows only
df.join(other, on="id", how="inner")

# Left — all left; nulls for misses
df.join(other, on="id", how="left")

# Full outer — all rows both sides
df.join(other, on="id", how="full")

# Semi — left rows that match; no right cols
df.join(other, on="id", how="semi")

# Anti — left rows that DON'T match
df.join(other, on="id", how="anti")

# Different key names
df.join(
    other,
    left_on="PassengerId",
    right_on="pid",
    how="left",
)

# Multi-key join
df.join(
    other,
    on=["customer_id", "date"],
    how="left",
)

# Asof — nearest key (sorted required)
df.sort("ts").join_asof(
    other.sort("ts"),
    on="ts",
)
```

| `how=`   | Rows Kept        | Right Cols | Nulls  |
|----------|------------------|------------|--------|
| `inner`  | matching only    | yes        | N/A    |
| `left`   | all left         | yes        | yes    |
| `full`   | all both sides   | yes        | both   |
| `semi`   | left matching    | no         | N/A    |
| `anti`   | left non-match   | no         | N/A    |
| `cross`  | left × right     | yes        | N/A    |