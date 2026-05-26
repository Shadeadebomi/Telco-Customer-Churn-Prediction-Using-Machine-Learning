# 📊 Telco Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

The Telco Customer Churn Prediction project is a Machine Learning classification project developed to predict whether a customer is likely to leave a telecommunications company (Churn) or remain a customer.

Customer churn is one of the most important business challenges for subscription-based companies because retaining existing customers is usually less expensive than acquiring new ones.

This project uses customer demographic information, service subscriptions, billing details, and account information to build a predictive model that identifies customers at risk of churning.

---

# 🎯 Business Problem

Telecommunication companies lose revenue when customers discontinue their services.

The objective of this project is to:

- Identify customers likely to churn.
- Understand factors contributing to churn.
- Build a predictive classification model.
- Help businesses implement customer retention strategies.

---

# 📂 Dataset Information

The dataset contains customer information including:

| Feature Category | Examples |
|-----------------|----------|
| Demographic Information | Gender, SeniorCitizen, Partner, Dependents |
| Service Information | OnlineSecurity, TechSupport, StreamingTV |
| Account Information | Contract, PaymentMethod, PaperlessBilling |
| Billing Information | MonthlyCharges, TotalCharges |
| Target Variable | Churn |

### Target Variable

```text
Churn
```

| Value | Meaning |
|---------|---------|
| 0 | Customer Stayed |
| 1 | Customer Churned |

---

# 🛠️ Project Workflow

The project follows a complete Machine Learning pipeline:

```text
Data Collection
↓
Data Cleaning
↓
Feature Engineering
↓
Feature Encoding
↓
Exploratory Data Analysis
↓
Feature Scaling
↓
Train-Test Split
↓
Model Building
↓
Model Evaluation
↓
Prediction
```

---

# 📥 Import Required Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
```

---

# 🧹 Data Cleaning

## Remove Unnecessary Column

The `customerID` column does not contribute to prediction and is removed.

```python
df.drop('customerID', axis=1, inplace=True)
```

---

## Handle TotalCharges

Convert the column to numeric values.

```python
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)
```

Replace missing values with the median.

```python
df['TotalCharges'].fillna(
    df['TotalCharges'].median(),
    inplace=True
)
```

---

# ⚙️ Feature Engineering

Additional features were created to improve model performance.

### FamilyCustomer

```python
df['FamilyCustomer'] = (
    df['Partner'] + df['Dependents']
)
```

### CustomerLifetimeValue

```python
df['CustomerLifetimeValue'] = (
    df['MonthlyCharges'] * df['tenure']
)
```

### TotalServices

```python
df['TotalServices'] = (
    df[['OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies']]
    .sum(axis=1)
)
```

---

# 🔄 Feature Encoding

Machine Learning algorithms require numerical input.

## Binary Encoding

```python
Yes → 1
No → 0
```

Applied to:

```text
Partner
Dependents
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
PaperlessBilling
Churn
```

Example:

```python
df['Partner'] = df['Partner'].replace({
    'Yes':1,
    'No':0
})
```

---

## One-Hot Encoding

Applied to:

```text
InternetService
Contract
PaymentMethod
```

```python
df = pd.get_dummies(
    df,
    columns=[
        'InternetService',
        'Contract',
        'PaymentMethod'
    ],
    drop_first=True
)
```

---

# 📈 Exploratory Data Analysis (EDA)

EDA was performed to understand customer behavior and churn patterns.

### Areas Explored

- Churn distribution
- Contract type analysis
- Payment methods
- Monthly charges
- Tenure distribution
- Service usage patterns
- Correlation analysis

Key findings:

- Customers on month-to-month contracts churn more frequently.
- Customers with higher monthly charges are more likely to churn.
- Customers without Online Security and Tech Support have higher churn rates.
- Long-term customers are less likely to churn.

---

# 📏 Feature Scaling

Numerical features were standardized.

```python
scaler = StandardScaler()

scale_cols = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'CustomerLifetimeValue'
]

X_train[scale_cols] = scaler.fit_transform(
    X_train[scale_cols]
)

X_test[scale_cols] = scaler.transform(
    X_test[scale_cols]
)
```

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets.

```python
X = df.drop('Churn', axis=1)

y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Split Ratio

| Dataset | Percentage |
|----------|----------|
| Training | 80% |
| Testing | 20% |

---

# 🤖 Model Building

## Logistic Regression

A Logistic Regression classifier was selected because:

- It performs well on binary classification problems.
- It provides interpretable coefficients.
- It produces probability estimates.
- It serves as a strong baseline model.

```python
lr = LogisticRegression(
    max_iter=1000
)

lr.fit(
    X_train,
    y_train
)
```

---

# 🔮 Churn Prediction

Generate predictions:

```python
y_pred = lr.predict(X_test)
```

Generate churn probabilities:

```python
y_prob = lr.predict_proba(X_test)
```

Example:

```text
[[0.18, 0.82]]
```

Interpretation:

```text
18% probability customer stays
82% probability customer churns
```

---

# 📊 Model Evaluation

## Accuracy Score

```python
accuracy_score(y_test, y_pred)
```

Measures overall prediction accuracy.

---

## Classification Report

```python
classification_report(
    y_test,
    y_pred
)
```

Provides:

- Precision
- Recall
- F1-Score
- Support

---

## Confusion Matrix

```python
confusion_matrix(
    y_test,
    y_pred
)
```

Helps evaluate:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

## ROC-AUC Score

```python
roc_auc_score(
    y_test,
    lr.predict_proba(X_test)[:,1]
)
```

Measures the model's ability to distinguish between churners and non-churners.

---

# 📌 Key Churn Indicators

The model identified several important factors influencing churn:

### High Churn Risk

- Month-to-Month Contract
- Fiber Optic Internet
- Electronic Check Payment
- High Monthly Charges
- Short Tenure
- No Tech Support
- No Online Security

### Low Churn Risk

- Long-Term Contracts
- Longer Tenure
- Online Security Subscription
- Tech Support Subscription
- Lower Monthly Charges

---

# 📈 Business Impact

This model helps businesses:

- Predict customer churn early.
- Reduce customer loss.
- Improve customer retention strategies.
- Increase revenue.
- Target high-risk customers with personalized offers.

---

# 🚀 Future Improvements

Potential enhancements include:

- Random Forest Classifier
- XGBoost Classifier
- Hyperparameter Tuning
- Feature Selection
- Cross Validation
- Model Deployment with Flask or Streamlit
- Real-Time Prediction API

---

# 🏆 Conclusion

The Telco Customer Churn Prediction model successfully classifies customers into churn and non-churn categories using Logistic Regression.

The model provides valuable business insights and enables proactive customer retention strategies by identifying customers who are likely to leave before churn occurs.

This project demonstrates a complete end-to-end Machine Learning workflow, from data preprocessing and feature engineering to model training, evaluation, and prediction.
