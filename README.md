# ML Assignments: Matplotlib & Scikit-Learn

Two coursework assignments covering data visualisation with **Matplotlib** and machine learning with **Scikit-Learn**. Each notebook is fully executed, so code, plots and outputs are visible without re-running.

## Repository structure

```
ml-assignment/
├── assignment1_matplotlib/
│   ├── Assignment1_Matplotlib.ipynb
│   ├── sales_dashboard.png
│   ├── style_whitegrid.png
│   └── style_ggplot.png
├── assignment2_scikit_learn/
│   ├── Assignment2_ScikitLearn.ipynb
│   ├── wine_model_comparison.png
│   ├── svc_gridsearch_heatmap.png
│   └── iris_elbow_curve.png
├── requirements.txt
└── README.md
```

---

## Assignment 1: Matplotlib

**Goal:** build and style a sales dashboard.

- 2×2 subplot figure: line (monthly sales), bar (sales by region), pie (category share), histogram (order values)
- Titles, axis labels, legends and a custom colour palette; exported as PNG at 200 dpi
- Peak sales month detected in code and marked with `ax.annotate()` and a highlighted marker
- **Bonus:** same data plotted in two built-in styles (`seaborn-v0_8-whitegrid`, `ggplot`) with a readability comparison

![Sales dashboard](assignment1_matplotlib/sales_dashboard.png)

> The dataset is synthetic and generated with a fixed seed, so results are reproducible.

---

## Assignment 2: Scikit-Learn

### 1. Classification (Wine dataset)
Stratified 75/25 train/test split. Logistic Regression and KNN use a `StandardScaler` pipeline because they are scale-sensitive.

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Logistic Regression | 1.000 | 1.000 |
| KNN (k=5) | 0.933 | 0.931 |
| Random Forest | 1.000 | 1.000 |

### 2. Pipeline + GridSearchCV
`StandardScaler` + `SVC` (RBF), tuning `C` and `gamma` with 5-fold cross-validation.

- Best parameters: **C = 1, gamma = 0.1**
- Cross-validation accuracy: **0.992**
- Test accuracy: **0.978**

![GridSearchCV heatmap](assignment2_scikit_learn/svc_gridsearch_heatmap.png)

### 3. Clustering (Iris, KMeans, k = 2 to 8)
The elbow curve shows a sharp drop in inertia from k = 2 to k = 3 and a flat decline afterwards, so **k = 3** is the best choice, matching the three iris species.

![Elbow curve](assignment2_scikit_learn/iris_elbow_curve.png)

### Key findings
1. Logistic Regression and Random Forest were perfect on the held-out Wine test set; KNN trailed slightly.
2. Feature scaling matters for distance- and margin-based models.
3. Tuned SVC reached about 98% test accuracy.
4. The Iris elbow supports k = 3.
5. Wine has only 178 samples, so a single split is noisy; cross-validation is safer for ranking close models.

---

## Setup

```bash
git clone https://github.com/ayush2459/ml-assignment.git
cd ml-assignment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
jupyter notebook
```

Open either notebook and choose **Run → Run All Cells**.

## Tech stack
Python · NumPy · pandas · Matplotlib · Scikit-Learn · Jupyter

## Author
**Ayush**: [github.com/ayush2459](https://github.com/ayush2459)
