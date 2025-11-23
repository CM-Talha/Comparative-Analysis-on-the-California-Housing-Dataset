# Multimodal Regression Analysis on California Housing Prices

## Overview

This repository provides a complete regression workflow using the California Housing Dataset from the UCI Machine Learning Repository. The project implements and compares multiple regression methods such as OLS, SVD-based regression, Gradient Descent, Adam, and PCA-based dimensionality reduction. The goal is to understand numerical behavior, stability, convergence, and performance across different regression methods.

## Features

* End-to-end regression pipeline
* Detailed preprocessing
* Analytical and iterative solvers
* Automatic visualization generation
* PCA-based dimensionality reduction
* Comparison of multiple regression methods
* IEEE-style report included
* Notebook version included

## Dataset

**California Housing Dataset (UCI ML Repository)**
Features include:

* Median income
* Average rooms / bedrooms
* Latitude & longitude
* Population & households
* Median house value (target)

Loaded via `sklearn.datasets.fetch_california_housing`.

## Installation & Environment Setup

### Option 1 — Run in Google Colab (Recommended)

1. Open Colab
2. Upload the notebook
3. Run all cells — dependencies auto-install

### Option 2 — Run Locally

#### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

#### 2. Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Run notebook

```bash
jupyter notebook project.ipynb
```

### Or run scripts directly

```bash
python src/train_regression.py
```

## Directory Structure

```
📦 california-regression/
 ┣ README.md
 ┣ requirements.txt
 ┣ project.ipynb
 ┣ src/
 │   ┣ data_loader.py
 │   ┣ preprocessing.py
 │   ┣ ols_solver.py
 │   ┣ svd_solver.py
 │   ┣ gradient_descent.py
 │   ┣ pca_module.py
 │   ┗ utils.py
 ┣ reports/
 │   ┣ IEEE_report.pdf
 │   ┗ Figures/
 ┗ results/
     ┗ regression_metrics.json
```

## Methods Implemented

### 1. OLS (Normal Equation)

```math
\hat{\beta} = (X^TX)^{-1}X^Ty
```

Fast, closed-form, but unstable for ill-conditioned matrices.

### 2. SVD-Based Pseudoinverse Regression

```math
X = U \Sigma V^T,
\hat{\beta} = V \Sigma^+ U^T y
```

Numerically stable even when (X^TX) is singular.

### 3. Batch Gradient Descent

```math
\beta_{k+1} = \beta_k - \eta \nabla L(\beta_k)
```

Includes:

* Loss curve
* Learning rate comparison
* Convergence analysis

### 4. Adam Optimizer

Adaptive learning method:

```math
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
```

```math
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
```

Often provides fastest convergence.

### 5. PCA + Regression

* PCA computed via SVD
* Variance explained plot
* Regression repeated for k = 2…8
* Error vs. k curve

## Results Summary

| Method           | Train MSE | Test MSE |
| ---------------- | --------- | -------- |
| OLS (Normal Eq)  | 0.5559    | 0.5758   |
| SVD (Pseudo‑inv) | 0.5559    | 0.5758   |
| Batch GD         | 0.5558    | 0.5759   |
| Adam             | 0.5559    | 0.5758   |

## Generated Visualizations

* MSE comparison chart
* R² comparison chart
* Residual plot
* Gradient descent loss curve
* Learning rate sensitivity
* PCA scree plot
* PCA 2D projection
* Regression after PCA

## Experiments & Observations

* OLS and SVD provide identical results when data is well-conditioned.
* SVD is more stable for near-singular matrices.
* Gradient descent requires tuning but converges.
* Adam converges faster.
* PCA reduces noise; but too much reduction harms accuracy.
* Optimal PCA dimension ≈ 6.

## Key Takeaways

* All solvers converge to similar solutions.
* SVD is the most numerically reliable.
* PCA enhances stability.
* Gradient-based methods are essential for large datasets.

## References

1. UCI Machine Learning Repository — California Housing
2. Hastie, Tibshirani, Friedman — *Elements of Statistical Learning*
3. Golub & Van Loan — *Matrix Computations*
4. Kingma & Ba — *Adam: A Method for Stochastic Optimization*

## License

MIT License. Free to use, modify, and distribute.
