# Repository Dependency & Import Log

This document tracks all primary imports and modules utilized throughout the project notebooks and scripts.

## Core Standard Libraries
- `os`, `sys`, `re`, `json`, `math`, `time`, `datetime`, `pickle`, `joblib`, `warnings`

## Data Handling & Numerical Computing
- `numpy` (as `np`)
- `pandas` (as `pd`)
- `scipy` (stats, optimize, spatial)

## Machine Learning (Scikit-Learn)
### Preprocessing & Model Selection
- `sklearn.preprocessing` (StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler)
- `sklearn.model_selection` (train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold)
- `sklearn.pipeline` (Pipeline)
- `sklearn.compose` (ColumnTransformer)
- `sklearn.impute` (SimpleImputer, KNNImputer)

### Algorithms
- `sklearn.linear_model` (LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier)
- `sklearn.tree` (DecisionTreeClassifier, DecisionTreeRegressor, plot_tree)
- `sklearn.ensemble` (RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor)
- `sklearn.neighbors` (KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors)
- `sklearn.cluster` (KMeans, DBSCAN, AgglomerativeClustering)
- `sklearn.naive_bayes` (GaussianNB, MultinomialNB, BernoulliNB)
- `sklearn.svm` (SVC, SVR)
- `xgboost` (XGBClassifier, XGBRegressor)

### Evaluation Metrics
- `sklearn.metrics` (accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error, silhouette_score)

## Data Visualization
- `matplotlib.pyplot` (as `plt`)
- `seaborn` (as `sns`)
- `graphviz`

## Web Scraping & APIs
- `requests`
- `bs4` (BeautifulSoup)
- `rapidapi` (via standard requests)
- `pandas.read_html`

## Miscellaneous
- `dotenv` (load_dotenv)
- `regex` (re)
- `ipywidgets`
