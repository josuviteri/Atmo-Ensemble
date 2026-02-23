# ATMO-ENSEMBLE


# Atmo-Ensemble: Researching Ensemble Models for Air Quality

**Josu Viteri, Gotzon Viteri, Iker Dominguez**

**2026**   

## 1. Project Abstract

The objective of this initiative is to research ensemble machine learning models and compare implementation strategies for a multi-class classification problem . specifically focused on air quality and pollution. Unlike standard challenges that prioritize only accuracy, this project aims to understand the underlying mechanics of ensemble learning, exploring trade-offs between variance and bias, handling ordinal targets, and optimizing for edge-computing efficiency.

## 2. The Dataset

The dataset consists of 5000 samples capturing critical environmental and demographic factors.

**Input Features:** Temperature, Humidity, , , , , CO, Proximity to Industrial Areas, and Population Density.


**Target Variable:** Air Quality Levels (Ordinal).
* Classes: `Good` < `Moderate` < `Poor` < `Hazardous` .

**Constraint:** The dataset is imbalanced and the target is ordinal rather than nominal.





## 3. Research Proposals & Experiments

This repository implements seven specific research lines defined in the proposal:

### 3.1 PCA and Feature Extraction

We investigate if dimensionality reduction (PCA) improves robustness compared to using raw, interpretable sensor data, particularly given the high multicollinearity between particulate matters and industrial gases.

### 3.2 Explainable AI (XAI)

We utilize SHAP dependence plots to identify non-linear interactions and understand the weight of specific features in decision-making.

### 3.3 Bagging vs. Boosting (Noise Resilience)

We compare parallel ensembles (Random Forest) against sequential ensembles (Gradient Boosting) by introducing artificial noise to simulate sensor error. The goal is to determine which architecture suffers faster metric degradation.

### 3.4 Handling Ordinality

Since the target has a rank (), standard multiclass classification effectively ignores the severity of errors. We compare:

**Standard Multiclass:** Treating classes as nominal.

**Frank and Hall Method:** Decomposing classes into binary classifiers.

**Regressor Ensemble:** Training XGBoost Regressor to predict continuous scores mapped to classes.



### 3.5 Imbalanced Learning & Cost Sensitivity

Given that "Hazardous" air events are rare but critical, we test:


**Hybrid Sampling:** SMOTE-Boost vs. RUSBoost vs. Balanced Random Forest.



**Cost-Sensitive Learning:** Implementing a custom loss function to penalize dangerous misclassifications (e.g., predicting "Good" when it is "Hazardous").



**Robustness to Missing Data:** Comparing imputation (KNN/Mean) vs. native NaN handling in XGBoost/LightGBM.



### 3.6 Semi-Supervised Learning

We simulate real-world labeling constraints by partially deleting target labels and running active and passive semi-supervised learning processes.

### 3.7 Efficiency vs. Accuracy (Edge Computing)

To address the feasibility of running models on low-power edge sensors, we compare:

**Heavy Models:** 3-layer Stacking Classifiers.

**Light Models:** Pruned Decision Trees or LightGBM.

**Metrics:** We measure Inference Time (ms) and Model Size (MB) alongside F1-scores.



## 4. Current State of the Art

Previous community approaches utilized Logistic Regression, SVM, and GBM, with Decision Trees achieving the best results (~0.95 F1 score). This project aims to surpass these results using advanced ensemble techniques while providing deeper research insights.

---


### Project Structure

```text
atmo-ensemble/
├── README.md                  # Project overview and research goals
├── requirements.txt           # Python dependencies 
├── data/
│   ├── raw/                   # Original dataset (5000 samples)
│   ├── processed/             # Cleaned data, imputed versions, and splits
│   └── external/              # Synthetic data generated via SMOTE/RUS
├── notebooks/                 # Jupyter notebooks for specific research proposals
│   ├── 01_eda_and_pca.ipynb            # Feature extraction & correlation analysis
│   ├── 02_bagging_vs_boosting.ipynb    # Noise simulation & architecture comparison
│   ├── 03_ordinal_ensembles.ipynb      # Frank & Hall and Regressor approaches
│   ├── 04_imbalanced_learning.ipynb    # SMOTE-Boost, RUSBoost, Cost-Sensitive
│   ├── 05_semi_supervised.ipynb        # Active learning experiments
│   ├── 06_edge_efficiency.ipynb        # Light vs. Heavy models (inference time)
│   └── 07_xai_interpretability.ipynb   # SHAP plots and feature interactions
├── src/                       # Source code for reproducible experiments
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py    # Scripts to download and clean data
│   │   └── corruption.py      # Simulating sensor failure/noise (Proposal 1.3 & 1.5.3)
│   ├── features/
│   │   ├── pca_reducer.py     # PCA implementation
│   │   └── ordinal_encoder.py # Custom ordinal mappings
│   ├── models/
│   │   ├── custom_loss.py     # Cost-sensitive loss functions (Proposal 1.5.2)
│   │   ├── ordinal_wrapper.py # Wrapper for Regressor-based classification
│   │   └── ensemble_builder.py# Helper to build Stacking/Voting classifiers
│   └── visualization/
│       └── plot_metrics.py    # Confusion matrices and SHAP visualization
├── results/                   # Generated artifacts
│   ├── figures/               # SHAP plots, degradation curves, confusion matrices
│   └── metrics/               # CSVs comparing F1-score, Inference Time, Model Size
└── reports/
    └── team10_project_proposal.pdf  # Original proposal document

```
