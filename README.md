# ML Assignments: Matplotlib & Scikit-Learn

Two coursework assignments covering data visualization with **Matplotlib** and machine learning with **Scikit-Learn**. The notebooks contain executed outputs so plots and model results can be inspected directly on GitHub.

## Repository contents

All files are currently in the repository root:

- `Assignment1_Matplotlib.ipynb` — 2×2 sales dashboard, peak-month annotation, 200-DPI export, and built-in style comparison.
- `Assignment2_ScikitLearn.ipynb` — Wine classification, an `StandardScaler + SVC` pipeline tuned with `GridSearchCV`, and KMeans elbow analysis on Iris.
- `sales_dashboard.png` — Matplotlib dashboard.
- `style_whitegrid.png`, `style_ggplot.png` — style comparison outputs.
- `wine_model_comparison.png` — classifier metric comparison.
- `svc_gridsearch_heatmap.png` — SVC cross-validation grid-search heatmap.
- `iris_elbow_curve.png` — KMeans inertia for k = 2 through 8.
- `requirements.txt` — Python dependencies.

## Assignment 1: Matplotlib

The dashboard includes:
- Line chart for monthly sales
- Bar chart for sales by region
- Pie chart for category share
- Histogram for order values
- Titles, axis labels, legends and a custom color palette
- Peak-month annotation using `ax.annotate()` and a highlighted marker
- PNG export at 200 DPI
- Bonus comparison between `seaborn-v0_8-whitegrid` and `ggplot` styles

**Data note:** The sales data is synthetic and generated with a fixed random seed. It is intended for demonstrating plotting and is not real business data.

## Assignment 2: Scikit-Learn

### 1. Wine classification

The notebook trains Logistic Regression, KNN, and Random Forest using a stratified 75/25 train-test split. Scaling is included in pipelines for scale-sensitive models. It compares accuracy and macro F1.

| Model | Accuracy | F1 (macro) |
|---|---:|---:|
| Logistic Regression | 1.000 | 1.000 |
| KNN (k=5) | 0.933 | 0.931 |
| Random Forest | 1.000 | 1.000 |

### 2. SVC pipeline and hyperparameter tuning

A pipeline combines `StandardScaler` and an RBF-kernel SVC. `GridSearchCV` tunes `C` and `gamma` with 5-fold cross-validation.

- Best parameters: `C = 1`, `gamma = 0.1`
- Mean cross-validation accuracy: approximately 0.992
- Holdout test accuracy: approximately 0.978

### 3. Iris clustering

KMeans is evaluated for `k = 2` through `8` using inertia (within-cluster sum of squares). The elbow curve suggests a bend around `k = 3`; this is a heuristic rather than proof that three clusters are always optimal.

## Setup and run

Python 3.10+ is recommended.

```bash
git clone https://github.com/ayush2459/ml-assignment.git
cd ml-assignment
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install packages and start Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Open either notebook and use **Run → Run All Cells** to reproduce the analysis.

## Key findings

1. Logistic Regression and Random Forest reached 1.000 accuracy and macro F1 on this particular Wine holdout split; KNN scored lower.
2. Scaling is important for distance- and margin-based models such as KNN and SVC.
3. Grid search selected an SVC configuration with about 0.992 mean cross-validation accuracy and 0.978 holdout test accuracy.
4. The Iris elbow plot indicates diminishing inertia reductions around k = 3.
5. These metrics depend on the data split and are not a guarantee of performance on new data.

## Tech stack

Python · NumPy · pandas · Matplotlib · scikit-learn · Jupyter

## Author

**Ayush** · [GitHub profile](https://github.com/ayush2459)
