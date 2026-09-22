# 🏦 Bank Customer Churn Prediction

An end-to-end Machine Learning project to predict bank customer churn using binary classification techniques. This repository is built around the [Kaggle Playground Series - Season 4, Episode 1](https://www.kaggle.com/competitions/binary-classification-with-bank-churn-dataset?utm_source=gemini) dataset.

## 📌 Table of Contents

* [Overview](#overview)

* [Dataset Description](#dataset-description)

* [Evaluation Metric](#evaluation-metric)

* [Project Structure](#project-structure)

* [Methodology](#methodology)

  * [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)

  * [2. Feature Engineering](#2-feature-engineering)

  * [3. Modeling & Validation](#3-modeling--validation)

  * [4. Ensembling & Blending](#4-ensembling--blending)

* [Installation & Setup](#installation--setup)

* [Usage](#usage)

* [Results](#results)

* [License & Acknowledgments](#license--acknowledgments)

## ℹ️ Overview

Customer churn is a critical metric for financial institutions. Retaining existing customers is significantly more cost-effective than acquiring new ones.

The goal of this project is to predict whether a bank customer will churn (`Exited = 1`) or stay (`Exited = 0`) based on demographic data, account attributes, and financial metrics. The dataset was generated from a deep learning model trained on the original [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shrutimehta/a-real-time-execution-of-bank-churn-prediction?utm_source=gemini).

## 📊 Dataset Description

The dataset consists of tabular records containing demographic and account details:

| 

| **Feature** | **Type** | **Description** | 
| `id` | Integer | Unique identifier for each row | 
| `CustomerId` | Integer | Unique identifier for each customer | 
| `Surname` | Categorical | Customer's last name | 
| `CreditScore` | Numerical | Credit score of the customer | 
| `Geography` | Categorical | Country of residence (e.g., France, Spain, Germany) | 
| `Gender` | Categorical | Gender of the customer (Male/Female) | 
| `Age` | Numerical | Age of the customer | 
| `Tenure` | Numerical | Number of years the customer has been with the bank | 
| `Balance` | Numerical | Account balance of the customer | 
| `NumOfProducts` | Numerical | Number of bank products the customer uses | 
| `HasCrCard` | Binary | Whether the customer has a credit card (1 = Yes, 0 = No) | 
| `IsActiveMember` | Binary | Whether the customer is an active member (1 = Yes, 0 = No) | 
| `EstimatedSalary` | Numerical | Estimated annual salary of the customer | 
| **`Exited`** | **Binary (Target)** | Whether the customer churned (1 = Yes, 0 = No) | 

## 📐 Evaluation Metric

Submissions are evaluated using the **Area Under the ROC Curve (ROC AUC)** between the predicted probabilities and the observed target values (`Exited`).

$$
\text{ROC AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt
$$

## 📁 Project Structure

```
.
├── data/
│   ├── train.csv              # Training set
│   ├── test.csv               # Test set
│   └── sample_submission.csv  # Baseline submission format
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb      # Model training & hyperparameter tuning
├── src/
│   ├── utils.py               # Helper functions
│   ├── features.py            # Feature extraction and encoding scripts
│   └── train.py               # Model training script
├── models/                    # Saved model artifacts (.pkl, .bin)
├── submissions/               # Submission files
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```

## ⚙️ Methodology

### 1. Exploratory Data Analysis (EDA)

* Analyzing distributions of numerical variables (`Age`, `CreditScore`, `Balance`, `EstimatedSalary`).

* Investigating target class imbalance (`Exited`).

* Cross-tabulating categorical variables (`Geography`, `Gender`, `NumOfProducts`, `IsActiveMember`) against the target.

* Combining original dataset records with Kaggle generated data for richer feature distributions.

### 2. Feature Engineering

* **Ratios & Interactions**:

  * `BalanceToSalaryRatio = Balance / EstimatedSalary`

  * `AgeToTenureRatio = Tenure / Age`

  * `CreditScoreToAge = CreditScore / Age`

* **Categorical Encodings**:

  * Frequency encoding for high-cardinality attributes like `Surname`.

  * Target Encoding and One-Hot Encoding for categorical features (`Geography`, `Gender`).

* **Group Aggregations**:

  * Grouping by `Geography` and `Gender` to compute mean/std statistics for `Balance` and `CreditScore`.

### 3. Modeling & Validation

* **Validation Strategy**: 5-Fold Stratified K-Fold Cross-Validation to prevent data leakage and handle target distribution across folds.

* **Algorithms Evaluated**:

  * **Gradient Boosting**: LightGBM, XGBoost, CatBoost

  * **Neural Networks**: TabNet / MLP

  * **Baselines**: Logistic Regression, Random Forest

### 4. Ensembling & Blending

* Rank-averaging and probability-weighted blending of top-performing GBDT models (LightGBM + XGBoost + CatBoost).

* Stacking ensemble using Meta-Learner (Logistic Regression / Ridge) for optimal predictions.

## 🚀 Installation & Setup

1. **Clone the repository:**

   ```
   git clone https://github.com/your-username/bank-churn-prediction.git
   cd bank-churn-prediction
   
   ```

2. **Create a virtual environment:**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   
   ```

4. **Kaggle API Setup (Optional):** Download data directly using the Kaggle CLI:

   ```
   kaggle competitions download -c binary-classification-with-bank-churn-dataset -p data/
   unzip data/*.zip -d data/
   
   ```

## 💻 Usage

To train the models and generate predictions:

```
python src/train.py --model lgb --folds 5

```

Or open and run the Jupyter Notebooks step-by-step:

```
jupyter notebook notebooks/

```

## 📈 Results

| **Model** | **CV ROC AUC** | **Public LB Score** | 
| Baseline LightGBM | 0.8872 | \- | 
| XGBoost + Engineered Features | 0.8891 | \- | 
| CatBoost | 0.8885 | \- | 
| **Weighted Ensemble (LGBM + XGB + CatBoost)** | **0.8912** | **0.8908** | 

*(Update table values with your actual model evaluation scores)*

## 📄 License & Acknowledgments

* **Dataset Source**: [Kaggle Playground Series s4e1](https://www.kaggle.com/competitions/binary-classification-with-bank-churn-dataset?utm_source=gemini)

* **Original Dataset**: Walter Reade and Ashley Chow, *Binary Classification with Bank Churn Dataset*, Kaggle (2024).

* **License**: MIT License