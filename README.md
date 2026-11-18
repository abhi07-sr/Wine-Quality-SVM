# 🍷 Red Wine Quality Prediction using Support Vector Machines (SVM)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-v2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-v1.24%2B-013243.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

An end-to-end Machine Learning project that classifies and predicts the quality of red wine using physicochemical laboratory test measurements and **Support Vector Machines (SVM)** with hyperparameter optimization.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset & Clinical Features](#-dataset--clinical-features)
- [Exploratory Data Analysis (EDA) & Preprocessing](#-exploratory-data-analysis-eda--preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Modeling: Support Vector Machines (SVM)](#-modeling-support-vector-machines-svm)
- [Performance & Benchmarks](#-performance--benchmarks)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Model Inference](#-model-inference)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Project Overview

Wine quality certification traditionally relies on expensive, subjective human sensory assessments by professional sommeliers. This project utilizes objective physicochemical laboratory analyses to predict wine quality grades automatically.

### Key Highlights:
1. **Data Cleaning & Deduplication:** Identified and safely eliminated 125 exact duplicate rows to prevent train-test data leakage.
2. **Outlier Treatment:** Handled ~25% outlier distribution using **IQR Capping (Winsorization)** to protect rare, high-quality wine samples from deletion.
3. **Distance Standardization:** Applied `StandardScaler` to ensure balanced margin optimization across features with vastly different scales.
4. **Target Formulation & Benchmarking:** Evaluated Multi-Class (6-class), 3-Tier Quality Grouping (**86.76%**), and Binary Quality Classification (**89.22%**).
5. **Hyperparameter Tuning:** Tuned regularization ($C$), kernel coefficient ($\gamma$), and kernels (`rbf`, `linear`, `poly`) via `GridSearchCV`.

---

## 📊 Dataset & Clinical Features

The dataset used is `WineQT.csv`, containing **1,143 wine samples** and **11 physicochemical attributes**:

| Feature | Type | Measurement / Role | Description |
| :--- | :--- | :--- | :--- |
| `fixed acidity` | Continuous | $g(tartaric\ acid)/dm^3$ | Non-volatile acids that do not evaporate readily |
| `volatile acidity` | Continuous | $g(acetic\ acid)/dm^3$ | Amount of acetic acid; high levels give an unpleasant vinegar taste |
| `citric acid` | Continuous | $g/dm^3$ | Adds freshness and crisp flavor |
| `residual sugar` | Continuous | $g/dm^3$ | Amount of sugar remaining after fermentation ceases |
| `chlorides` | Continuous | $g(sodium\ chloride)/dm^3$ | Amount of salt in the wine |
| `free sulfur dioxide` | Continuous | $mg/dm^3$ | Free form of $SO_2$ that prevents microbial growth and oxidation |
| `total sulfur dioxide` | Continuous | $mg/dm^3$ | Sum of free and bound forms of $SO_2$ |
| `density` | Continuous | $g/cm^3$ | Density of wine close to that of water depending on alcohol/sugar |
| `pH` | Continuous | Scale (0–14) | Describes acidity level on a scale from 0 (very acid) to 14 (basic) |
| `sulphates` | Continuous | $g(potassium\ sulphate)/dm^3$ | Antimicrobial and antioxidant wine additive |
| `alcohol` | Continuous | % by volume | Percent alcohol content of the wine |
| **`quality`** | **Discrete** | **Target (3 to 8)** | **Sensory quality score graded by wine experts** |

---

## 🛠️ Exploratory Data Analysis (EDA) & Preprocessing

### 1. Deduplication
- Raw dataset contained 1,143 records.
- Identified and dropped **125 duplicate records** (`df.drop_duplicates()`), preserving **1,018 clean, independent instances** and preventing data leakage.

### 2. Missing Value Check
- Checked for null or empty values (`df.isnull().sum()`); verified **0 missing entries**.
- Dropped irrelevant database identifier column `Id`.

### 3. Outlier Handling via IQR Capping
Extreme outliers were observed in `residual sugar`, `chlorides`, `sulphates`, and `total sulfur dioxide`. 
* **Why not drop?** Deleting outlier rows would destroy 262 records (~25.7% of the dataset), disproportionately eliminating rare premium wines (ratings 3 and 8).
* **Solution:** Capped values outside the $1.5 \times \text{IQR}$ envelope:
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

### 4. Feature Scaling (Mandatory for SVM)
Because SVM maximizes the geometric margin between support vectors using Euclidean distance:
$$d(x_1, x_2) = \sqrt{\sum (x_{1i} - x_{2i})^2}$$
Features with large ranges (e.g., `total sulfur dioxide` up to 160+) would otherwise completely dominate small-scale features (e.g., `chlorides` $\sim 0.05$). Applied `StandardScaler` to normalize all features to $\mu = 0, \sigma = 1$.

---

## 🧪 Feature Engineering

Domain-specific chemical interaction features were synthesized to improve hyperplane separability:
- **`total_acidity`** = `fixed acidity` + `volatile acidity`
- **`bound_sulfur_dioxide`** = `total sulfur dioxide` - `free sulfur dioxide`
- **`sulfur_ratio`** = $\frac{\text{free sulfur dioxide}}{\text{total sulfur dioxide} + 10^{-5}}$

---

## 🧠 Modeling: Support Vector Machines (SVM)

### Why SVM for Wine Classification?
1. **Effective in High-Dimensional Spaces:** Robust when ratio of samples to features is moderate.
2. **Kernel Trick:** Maps non-linear chemical interactions into higher-dimensional linear feature spaces where distinct qualities become cleanly separable.
3. **Margin Maximization:** Focuses on the hardest boundary samples (support vectors) rather than the entire distribution.

### Mathematical Formulation (RBF Kernel):
$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$
Hyperparameters tuned:
- **$C$ (Regularization):** Controls penalty for margin misclassifications (tested: $0.1, 1, 5, 10, 20$).
- **$\gamma$ (Kernel Coefficient):** Defines influence radius of support vectors (tested: `'scale'`, `'auto'`, $0.01, 0.1$).

---

## 📈 Performance & Benchmarks

| Formulation | Classes | Model / Kernel | Test Accuracy | Macro F1-Score | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multi-Class (Raw)** | 6 (3, 4, 5, 6, 7, 8) | SVC (RBF, $C=1.0$) | ~58.3% – 60.3% | 0.48 | Subjective sensory overlap between scores 5 & 6 (83% of data) |
| **3-Tier Classification** 🌟 | 3 (Low, Medium, High) | SVC (RBF, $C=5.0$) | **86.76%** | **0.84** | High clinical & commercial distinction |
| **Binary Classification** 🏆 | 2 (Good vs Regular) | SVC (RBF, $C=10.0$) | **89.22%** | **0.88** | Kaggle / Industry benchmark standard ($\ge 7$ = Good) |

---

## 📁 Project Structure

```text
SVM/
│
├── README.md                          <- Project documentation & portfolio summary
├── WineQT.csv                         <- Dataset of red wine physicochemical tests
└── Implementation Of SVM.ipynb        <- End-to-end Jupyter Notebook (EDA, Scaling, Tuning, Evaluation)
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/abhi07-sr/Wine-Quality-SVM.git
cd Wine-Quality-SVM
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook "Implementation Of SVM.ipynb"
```

---

## 💻 Model Inference

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load trained SVM and scaler
# Sample wine input: [fixed acidity, volatile acidity, citric acid, residual sugar,
#                      chlorides, free SO2, total SO2, density, pH, sulphates, alcohol]
sample_wine = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

# Scale input with the fitted scaler
sample_wine_scaled = scaler.transform(sample_wine)

# Predict quality tier (0 = Low, 1 = Medium, 2 = Premium)
prediction = final_svm.predict(sample_wine_scaled)
tier_map = {0: "Low Quality", 1: "Medium / Standard", 2: "Premium / High Quality"}

print(f"Predicted Wine Quality: {tier_map[prediction[0]]}")
```

---

## 👤 Author & Acknowledgments
- **Developer:** [Abhinandan Naik](https://github.com/abhi07-sr)
- **Dataset:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 2009.
- **Tools:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.
