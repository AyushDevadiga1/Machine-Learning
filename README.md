# Machine Learning Engineering Repository

This repository is a professional-grade collection of machine learning algorithms, data engineering pipelines, and statistical research. It spans the spectrum from foundational "from-scratch" implementations to advanced ensemble modeling and automated web scraping.

## Portfolio Highlights

### 1. Mathematical Implementations (From Scratch)
Understanding the mechanics behind the "black box."
*   **Linear & Multiple Regression**: Manual implementation of Gradient Descent, cost functions, and weight updates.
*   **KNN & Perceptron**: Logical flow of distance metrics and binary classification from the ground up.
*   **Ensemble Theory**: Conceptualizing Bagging and Boosting through manual iteration.

### 2. Advanced Supervised Learning
Production-ready library implementations focusing on hyperparameter tuning and model evaluation.
*   **Ensemble Excellence**: Deep dives into `XGBoost`, `AdaBoost`, and `RandomForest` for both classification and regression.
*   **Support Vector Machines**: High-dimensional decision boundaries and kernel tricks.
*   **Naïve Bayes**: Probabilistic modeling for classification tasks.

### 3. Unsupervised Learning & Clustering
Identifying hidden patterns without labels.
*   **Centroid Based**: K-Means clustering applied to demographic and biological data.
*   **Density Based**: DBSCAN for spatial data analysis and outlier detection.
*   **Hierarchical**: Agglomerative clustering for structural analysis.

### 4. Robust Data Engineering Pipeline
A significant portion of the repository is dedicated to the most critical part of ML: Data.
*   **Cleaning**: Systematic handling of missing values (CCA, Imputation), outliers (Z-Score, IQR), and duplicate resolution.
*   **Transformation**: Handling skewed data, categorical encoding (One-Hot, Ordinal), and feature scaling.
*   **Pipelining**: Using Scikit-Learn `Pipeline` and `ColumnTransformer` to automate workflows.

### 5. Automated Data Acquisition
*   **Web Scraping**: Utilizing `BeautifulSoup` and `RegEx` to Build datasets from live websites and YouTube social data.
*   **API Management**: Fetching and normalizing data from REST APIs using `RapidAPI`.

## Technical Ecosystem

### Frameworks & Libraries
*   **Numerical Computing**: `NumPy`, `SciPy`
*   **Data Manipulation**: `Pandas`
*   **Visualization**: `Matplotlib`, `Seaborn` (Advanced plotting like Bivariate/Multivariate analysis)
*   **Machine Learning**: `Scikit-Learn`, `XGBoost`, `LightGBM`
*   **Data Collection**: `Requests`, `BeautifulSoup4`

### Repository Organization
```text
├── Notebooks/          # Core algorithmic implementations and EDA
├── Datasets/           # Raw and cleaned CSV/JSON data sources
│   ├── Cleaned/        # Processed data ready for modeling
│   └── Raw/            # Original messy datasets for cleaning practice
├── Models/             # Serialized (.pkl/.joblib) trained models
└── Documentation/      # In-depth analysis and import logs
```

## Setup & Execution

### Environment Configuration
Ensure you have Python 3.8+ installed.

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd ML
   ```

2. **Dependency Management**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate # windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Explore**:
   Open individual notebooks via VS Code or `jupyter notebook`.
