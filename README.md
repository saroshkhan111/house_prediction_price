# House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6F00?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-217346?style=flat-square)](https://lightgbm.readthedocs.io/)

An end-to-end regression project that predicts residential sale prices from 79 property
features using the Ames, Iowa housing dataset. The notebook covers the full machine learning
lifecycle — exploratory data analysis, domain-aware missing value imputation, feature
encoding, cross-validated benchmarking of 10 regressors, hyperparameter tuning, and two
ensemble strategies — and finishes with a Kaggle-compatible submission file.

**Headline result:** a weighted voting ensemble of tuned Random Forest, XGBoost, and
LightGBM models reaches a best 5-fold cross-validated **R² of 0.894** on 232 engineered
features.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Results at a Glance](#results-at-a-glance)
- [Inside the Notebook](#inside-the-notebook)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Takeaways](#key-takeaways)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Acknowledgements](#acknowledgements)

---

## Overview

The goal is straightforward: given everything you can observe about a house — its overall
quality, living area, neighborhood, year built, garage, basement, and so on — estimate what
it will sell for. What makes the problem interesting is that the data is messy in the way
real data usually is: 19 of the 81 columns in the training set contain missing values (35
once train and test are combined), many features are ordinal categories with a natural
order (Poor → Excellent), and the target is noticeably right-skewed.

This repository is the complete, reproducible solution. It demonstrates:

- **Exploratory data analysis** — distributions, correlations, and a null-value heatmap to
  understand the data before touching a model
- **Data cleaning and imputation** — per-feature strategies chosen from the meaning of the
  data, not blind mean-filling
- **Feature engineering** — ordinal encoding with domain mappings, one-hot encoding, and
  type conversions (81 raw columns → 232 model-ready features)
- **Model benchmarking** — 10 regressors scored under identical 5-fold cross-validation
- **Hyperparameter tuning** — `RandomizedSearchCV` over Random Forest, XGBoost, and LightGBM
- **Ensembling** — weighted voting and stacking regressors with a Ridge meta-model
- **Reproducibility** — fixed `random_state` everywhere, relative data paths, and an
  incremental commit history

---

## Dataset

The project uses the classic **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**
competition dataset from Kaggle (the Ames housing dataset).

| File | Description |
| --- | --- |
| `data/train.csv` | 1,460 rows × 81 columns — includes `SalePrice` (the target) |
| `data/test.csv` | 1,459 rows × 80 columns — no `SalePrice`, used for final predictions |
| `data/data_description.txt` | Column-by-column dictionary for all 79 explanatory variables |
| `data/sample_submission.csv` | Reference submission format (`Id`, `SalePrice`) |

Key features the models rely on most: `OverallQual` (material and finish quality),
`GrLivArea` (above-grade living area), `YearBuilt`, `Neighborhood`, `GarageCars`,
`FullBath`, `TotalBsmtSF`, and `KitchenQual`.

---

## Results at a Glance

All scores are **R² under 5-fold cross-validation** on the training set (mean ± std).
`random_state=42` is fixed throughout.

**Benchmark — 10 regressors with default hyperparameters:**

| Model | CV R² | CV std |
| --- | ---: | ---: |
| RandomForestRegressor | 0.842 | ± 0.092 |
| GradientBoostingRegressor | 0.837 | ± 0.138 |
| XGBRegressor | 0.822 | ± 0.084 |
| KNeighborsRegressor | 0.716 | ± 0.045 |
| DecisionTreeRegressor | 0.706 | ± 0.062 |
| LinearRegression | 0.658 | ± 0.156 |
| SVR | -0.054 | ± 0.021 |
| MLPRegressor | -4.972 | ± 0.678 |
| GaussianProcessRegressor | -5.286 | ± 0.709 |
| SGDRegressor | -387.369 | ± 313.736 |

The bottom rows are intentionally kept in the notebook: they show that most of the gain in
this project came from picking the right *family* of models, not from tuning the wrong one.

**After tuning and ensembling:**

| Model | Tuning | CV R² |
| --- | --- | ---: |
| Random Forest | `RandomizedSearchCV`, 30 candidates, 3-fold | 0.858 |
| LightGBM | `RandomizedSearchCV`, 50 candidates, 5-fold | 0.889 |
| XGBoost | `RandomizedSearchCV`, 50 candidates, 5-fold | 0.893 |
| **Weighted Voting (RF + XGB + LGBM)** | — | **0.894 ± 0.024** |
| Stacking (RF + XGB + LGBM → Ridge) | — | 0.893 ± 0.026 |

The final model writes `submission.csv` with 1,459 predictions in the exact format Kaggle
expects.

---

## Inside the Notebook

`house_price_prediction_1.ipynb` is a single, fully commented notebook that takes you from
raw CSV to submission. Here is the flow:

**1. Load and integrate.** Train and test sets are concatenated (2,919 rows × 81 columns)
so that every preprocessing step treats both sets identically, then split back apart after
encoding — a common and effective pattern for competition-style data.

**2. Exploratory data analysis.** Structural inspection (`shape`, `info`, `describe`),
`SalePrice` distribution and summary statistics, feature-versus-target correlations, and a
heatmap of null counts per column (`EDA_img/heatmap_DF_of_null_values.png`) to decide
where cleaning effort should go.

**3. Missing value imputation.** Every column with gaps was imputed according to what the
gap actually *means*:

- `LotFrontage` → mean of the feature (verified column-by-column after imputation)
- `MSZoning`, `Electrical`, `Exterior1st/2nd` → mode (most frequent category)
- `Alley`, basement (`BsmtQual`, `BsmtExposure`, `BsmtFinType1/2`, …), and garage columns
  (`GarageType`, `GarageFinish`, …) → an explicit "no alley / no basement / no garage"
  category, because `NaN` here means *absence*, not *unknown*
- Small remaining gaps in `MasVnrType`, `Utilities`, `KitchenQual`, and `Functional` →
  filled with safe defaults, then verified with a final zero-null check

No columns were dropped — the decision (and the reasoning behind it) is documented in the
notebook rather than done silently.

**4. Feature engineering.** 81 raw columns become 232 model-ready features:

- **Ordinal encoding** with domain-informed mappings, e.g.
  `ExterQual: {Po:1, Fa:2, TA:3, Gd:4, Ex:5}` and graded scales for basement quality,
  basement finish, heating quality, kitchen quality, garage finish, and fence quality
- **One-hot encoding** (`pd.get_dummies`, `drop_first=True`) for nominal features such as
  `MSSubClass`, `MoSold`, and `Fence`
- Year columns (`YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold`) cast to numeric
- An explicit check that no ordinal feature was accidentally mapped to `0`

**5. Scaling.** A `StandardScaler` is **fit on the training data only** and then applied to
both splits, so test-set statistics never leak into training. The notebook also inspects
the fitted scaler's attributes (`mean_`, `scale_`, `var_`, `n_features_in_`) — the exact
state you would need to persist for consistent inference in production.

**6. Benchmark.** A small reusable `test_model()` helper runs 5-fold CV (`KFold`,
shuffled) and scores 10 regressors on identical folds, so the comparison is fair. Results
are collected into a DataFrame and sorted by R².

**7. Tune.** `RandomizedSearchCV` explores realistic parameter distributions for Random
Forest, XGBoost, and LightGBM — hundreds of candidate combinations evaluated in parallel
(`n_jobs=-1`) — and reports the best parameters and scores for each.

**8. Ensemble.** Two combinations of the tuned models are trained and compared:

- a **weighted `VotingRegressor`** (higher weight on the gradient boosting models) — the
  final choice, CV R² **0.894**
- a **`StackingRegressor`** with Ridge as the meta-model, CV R² 0.893

**9. Submit.** The best ensemble predicts on the held-out test set and writes
`submission.csv` (`Id`, `SalePrice`) — ready for upload to Kaggle.

---

## Project Structure

```
house_prediction_price/
├── data/
│   ├── data_description.txt          # Feature dictionary (79 explanatory variables)
│   ├── train.csv                     # 1,460 rows, includes SalePrice
│   ├── test.csv                      # 1,459 rows, no SalePrice
│   └── sample_submission.csv         # Kaggle submission format reference
├── EDA_img/
│   └── heatmap_DF_of_null_values.png # Null-value heatmap from the EDA phase
├── house_price_prediction_1.ipynb    # The full pipeline, end to end
└── README.md
```

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/saroshkhan111/house_prediction_price.git
cd house_prediction_price
```

**2. Create an environment and install dependencies**

Using [uv](https://github.com/astral-sh/uv) (the tool used to build this project):

```bash
uv venv
uv pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm scipy jupyter
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm scipy jupyter
```

**3. Run the notebook**

```bash
jupyter notebook house_price_prediction_1.ipynb
```

> **Note:** the notebook was developed on Google Colab, and its loading cells point at
> `/content/train.csv` and `/content/test.csv`. When running locally, change those two
> paths to `data/train.csv` and `data/test.csv` — everything else works as-is. The tuning
> cells run several hundred cross-validation fits in parallel, so allow a few minutes for
> a full top-to-bottom execution.

---

## Key Takeaways

What I took away from building this project end to end:

1. **A missing value is information.** `NaN` in `Alley` or `GarageType` means "this house
   has no alley / no garage" — a meaningful category — not "data lost". Imputing with an
   explicit `None`/`NA` level preserved that signal; blanket mean-filling would have
   destroyed it.
2. **Benchmark before you tune.** A single cross-validated sweep across 10 models told me
   immediately where the signal lived (tree ensembles) and where it didn't (linear, kernel,
   and neural baselines). Tuning a bad model choice is wasted compute.
3. **`RandomizedSearchCV` beats grid search for wide spaces.** 50 random candidates
   consistently found near-optimal configurations for XGBoost and LightGBM at a fraction of
   the fits a full grid would need.
4. **Ensembling has diminishing returns.** The voting ensemble beat a single tuned XGBoost
   by ~0.001 R². Worth reporting honestly: sometimes the simpler model is the right
   production choice.
5. **Watch the gap between train and cross-validated scores.** The ensembles hit 0.99 train
   R² versus ~0.89 CV R² — variance is still in the models, and that is exactly what
   regularization and feature selection should target next.
6. **Reproducibility is not optional.** Fixed seeds, relative paths, and a commit history
   that shows each preprocessing decision mean anyone (including future me) can rerun and
   trust the results.

---

## Limitations and Next Steps

Honest assessment of where the project stands and what I would do next:

- **Metric alignment.** The Kaggle competition is scored on RMSLE (root-mean-squared log
  error), while this notebook tracks R². The obvious next step is a `log1p` transform of
  `SalePrice` during training and `expm1` on predictions — on a right-skewed target this
  typically improves RMSLE substantially.
- **Pipeline packaging.** The preprocessing chain is written as explicit notebook cells.
  Refactoring it into a single scikit-learn `ColumnTransformer` + `Pipeline` would make the
  whole transform serializable with the model — one object to save, one object to serve.
- **Interpretability.** Adding residual plots, actual-vs-predicted charts, and SHAP values
  would make the model's behavior explainable to non-technical stakeholders.
- **Deployment.** A lightweight Streamlit app or FastAPI endpoint would turn this into a
  live tool where anyone can enter house details and get a price estimate.
- **Strong linear baselines.** Ridge, Lasso, and ElasticNet on the encoded matrix are
  classic strong performers on this dataset and are worth adding to the benchmark table.

---

## Acknowledgements

- Data courtesy of the **[Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**
  competition, built on Dean De Cock's Ames, Iowa housing dataset.
- Models and tooling by the scikit-learn, XGBoost, and LightGBM communities.

---

## Author

**saroshkhan111** — [github.com/saroshkhan111](https://github.com/saroshkhan111)

If you found this useful or have feedback, feel free to open an issue or reach out — I am
always happy to talk about data and machine learning.

