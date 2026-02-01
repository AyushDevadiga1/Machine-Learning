# Machine Learning Repository

A comprehensive collection of machine learning implementations, data preprocessing techniques, and statistical analyses. This repository serves as a practical resource for various algorithms and data science workflows.

## Core Modules

### Supervised Learning
*   **Classification**: Implementations of Decision Trees, Random Forest, XGBoost, AdaBoost, Naive Bayes, K-Nearest Neighbors (KNN), and Logistic Regression.
*   **Regression**: Linear Regression, Multiple Variable Linear Regression, Ridge, Lasso, and ElasticNet implementations.
*   **Advanced Gradient Boosting**: Gradient Boosting Regressor/Classifier and HistGradientBoosting variants.

### Unsupervised Learning
*   **Clustering**: KMeans, DBSCAN, and Agglomerative Clustering algorithms.
*   **Dimensionality Reduction**: Principal Component Analysis (PCA) and related techniques.

### Data Engineering and Preprocessing
*   **Data Cleaning**: Identification and handling of missing values using various imputation methods, handling messy datasets, and removing duplicates.
*   **Feature Engineering**: Transformers, pipelining, numerical to categorical conversions, and outlier analysis.
*   **Analysis**: Univariate, bivariate, and multivariate analysis of datasets (e.g., Titanic, Housing, Diabetes).

### Web Scraping and API Integration
*   **Web Scraping**: Scripts for data extraction from websites and YouTube comments using various libraries and RegEx.
*   **API Integration**: Examples of fetching data via RapidAPI and Pandas.

## Project Structure

The repository is organized by specific notebooks and datasets corresponding to various ML tasks:
*   **Datasets (.csv, .json, .tsv)**: A wide variety of datasets used for training and testing models.
*   **Notebooks (.ipynb)**: Detailed implementations and experiments for each algorithm or preprocessing technique.
*   **Scripts (.py)**: Auxiliary Python scripts for specific tasks like animations or model definitions.
*   **Models**: Directory for saved machine learning models.

## Technologies Used

*   **Languages**: Python
*   **Libraries**: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn, XGBoost, BeautifulSoup
*   **Tools**: Jupyter Notebook House

## Getting Started

### Prerequisites

*   Python 3.x
*   A virtual environment is recommended

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebooks:
   ```bash
   jupyter notebook
   ```
