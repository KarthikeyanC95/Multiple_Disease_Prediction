# Multiple Disease Prediction — Kidney, Liver & Parkinson's

## 📌 Project Overview

**Multiple Disease Prediction** is a Machine Learning project designed to predict the presence of multiple diseases using patient and clinical data.

The project focuses on three major disease prediction tasks:

* 🫘 **Kidney Disease Prediction**
* 🫀 **Liver Disease Prediction**
* 🧠 **Parkinson's Disease Prediction**

The objective is to develop machine learning models that can learn patterns from medical datasets and provide predictions based on patient attributes and diagnostic measurements.

This project follows a complete machine learning workflow:

> **Data Collection → Exploratory Data Analysis → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Model Selection → Prediction**

The project is intended as an **educational machine learning application** and should not be considered a substitute for professional medical diagnosis.

---

# 🎯 Problem Statement

Early identification of diseases can play an important role in healthcare by helping identify patients who may require further medical evaluation.

However, medical datasets often contain:

* Multiple numerical health indicators
* Categorical patient information
* Missing values
* Imbalanced target classes
* Features with significantly different scales
* Highly correlated clinical measurements

The problem addressed in this project is:

> **Can machine learning classification algorithms learn meaningful patterns from patient health data and accurately predict whether a patient is likely to have a particular disease?**

Three separate classification problems are considered:

### 1. Kidney Disease Prediction

Build a classification model that predicts whether a patient is likely to have kidney disease based on clinical and laboratory measurements.

### 2. Liver Disease Prediction

Build a classification model that predicts whether a patient is likely to have liver disease using demographic information and liver-function-related measurements.

### 3. Parkinson's Disease Prediction

Build a classification model that predicts whether a patient is likely to have Parkinson's disease using biomedical voice measurements.

---

# 🧠 Machine Learning Approach

The project uses supervised machine learning classification algorithms.

For each disease, the general workflow is:

```text
                Medical Dataset
                      │
                      ▼
              Data Understanding
                      │
                      ▼
          Exploratory Data Analysis
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    Numerical Features      Categorical Features
          │                       │
          └───────────┬───────────┘
                      ▼
             Data Preprocessing
                      │
                      ▼
             Feature Engineering
                      │
                      ▼
                Train/Test Split
                      │
                      ▼
                 Feature Scaling
                      │
                      ▼
              Model Development
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Logistic    Tree-Based    Distance/
       Regression    Models      Kernel Models
          │           │           │
          └───────────┼───────────┘
                      ▼
                Model Evaluation
                      │
                      ▼
                Best Model
                      │
                      ▼
                  Prediction
```

---

# 🔬 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure and distribution of the datasets.

The analysis included:

* Dataset dimensions
* Column names and data types
* Descriptive statistics
* Missing-value analysis
* Class distribution
* Numerical feature distributions
* Categorical feature distributions
* Correlation analysis
* Correlation heatmaps
* Pair plots

EDA helps identify patterns, outliers, class imbalance, and relationships between features before training machine learning models.

---

# 🫀 Liver Disease Prediction

## Dataset

The Liver Disease dataset contains **583 patient records** and **11 columns** before preprocessing.

The target variable is:

```text
Dataset
```

The original target values were:

```text
1 → Liver Disease
2 → No Liver Disease
```

For machine learning, the target was transformed into:

```text
0 → No Liver Disease
1 → Liver Disease
```

### Target Distribution

| Class     | Meaning          |   Count |
| --------- | ---------------- | ------: |
| 0         | No Liver Disease |     416 |
| 1         | Liver Disease    |     167 |
| **Total** |                  | **583** |

The dataset is therefore somewhat imbalanced, with more observations belonging to class `0`.

---

## Liver Features

The model uses the following features:

| Feature                    | Description                |
| -------------------------- | -------------------------- |
| Age                        | Patient age                |
| Gender_Male                | Encoded gender             |
| Total_Bilirubin            | Total bilirubin level      |
| Direct_Bilirubin           | Direct bilirubin level     |
| Alkaline_Phosphotase       | Alkaline phosphatase level |
| Alamine_Aminotransferase   | ALT measurement            |
| Aspartate_Aminotransferase | AST measurement            |
| Total_Protiens             | Total protein level        |
| Albumin                    | Albumin level              |
| Albumin_and_Globulin_Ratio | Albumin/globulin ratio     |

---

# 🛠️ Liver Data Preprocessing

### 1. Missing Value Treatment

The dataset contained **4 missing values** in:

```text
Albumin_and_Globulin_Ratio
```

These missing values were replaced using the mean of the column.

```python
df['Albumin_and_Globulin_Ratio'] = (
    df['Albumin_and_Globulin_Ratio']
    .fillna(df['Albumin_and_Globulin_Ratio'].mean())
)
```

After preprocessing, there were no missing values.

### 2. Target Encoding

The original target labels were converted from:

```text
1, 2
```

to:

```text
0, 1
```

### 3. Categorical Encoding

The `Gender` column was converted into a numerical feature using one-hot encoding:

```python
pd.get_dummies(df, columns=['Gender'], drop_first=True)
```

This produced:

```text
Gender_Male
```

where:

```text
0 → Female
1 → Male
```

### 4. Feature Scaling

Because the numerical features have very different ranges, `StandardScaler` was applied.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This transforms the features approximately to:

```text
Mean = 0
Standard Deviation = 1
```

### 5. Train/Test Split

The dataset was divided into:

```text
80% → Training data
20% → Testing data
```

using:

```python
train_test_split(
    X_scaled_df,
    y,
    test_size=0.2,
    random_state=0
)
```

The test set contained **117 observations**.

---

# 🤖 Models Evaluated — Liver Disease

The following classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. K-Nearest Neighbors (KNN)
6. Gradient Boosting
7. Support Vector Machine (SVM)

The models were evaluated using the same train/test split.

---

# 📊 Liver Disease Model Performance

The testing accuracy obtained from the experiments is:

| Rank | Model                   | Training Accuracy | Testing Accuracy |
| ---: | ----------------------- | ----------------: | ---------------: |
| 🥇 1 | **Logistic Regression** |        **74.03%** |       **70.09%** |
| 🥈 2 | **Gradient Boosting**   |            92.92% |       **69.23%** |
|    3 | Decision Tree           |           100.00% |           66.67% |
|    4 | XGBoost                 |            72.53% |           66.67% |
|    5 | SVM                     |            72.53% |           66.67% |
|    6 | Random Forest           |            96.78% |           65.81% |
|    7 | KNN                     |            78.97% |           65.81% |

### Best Performing Model

Based on **test accuracy**, Logistic Regression performed best among the evaluated models:

```text
Testing Accuracy = 70.09%
```

The trained Logistic Regression model was subsequently saved using Pickle:

```python
pickle.dump(model, open("liver_model.sav", "wb"))
```

---

# 📈 Liver Model Evaluation Metrics

## Logistic Regression

### Confusion Matrix

```text
[[74,  4],
 [31,  8]]
```

Represented as:

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------: | ----------: |
| **Actual 0** |          74 |           4 |
| **Actual 1** |          31 |           8 |

Therefore:

* True Negative = **74**
* False Positive = **4**
* False Negative = **31**
* True Positive = **8**

### Classification Report

| Class            | Precision |   Recall | F1-Score |
| ---------------- | --------: | -------: | -------: |
| 0                |      0.70 | **0.95** |     0.81 |
| 1                |      0.67 |     0.21 |     0.31 |
| **Macro Avg**    |  **0.69** | **0.58** | **0.56** |
| **Weighted Avg** |  **0.69** | **0.70** | **0.64** |

### Important Observation

Although Logistic Regression achieved the highest overall test accuracy, the recall for the positive class was relatively low:

```text
Positive-class recall = 21%
```

This means the model missed a considerable number of actual positive cases.

For a medical prediction problem, **accuracy alone is not sufficient**. Recall, precision, F1-score, ROC-AUC, and especially the cost of false negatives should also be considered.

Therefore, the current model should be viewed as a baseline rather than a clinically reliable diagnostic model.

---

# 🧪 Comparison of Liver Models

### Logistic Regression

```text
Training Accuracy : 74.03%
Testing Accuracy  : 70.09%
```

The model has relatively similar training and testing performance, suggesting less overfitting compared with some tree-based models.

### Gradient Boosting

```text
Training Accuracy : 92.92%
Testing Accuracy  : 69.23%
```

The relatively large difference between training and testing accuracy indicates some degree of overfitting.

### Decision Tree

```text
Training Accuracy : 100.00%
Testing Accuracy  : 66.67%
```

The perfect training accuracy combined with lower test accuracy is a strong indication of overfitting.

### Random Forest

```text
Training Accuracy : 96.78%
Testing Accuracy  : 65.81%
```

Despite strong training performance, the test performance was lower than Logistic Regression.

### XGBoost

```text
Training Accuracy : 72.53%
Testing Accuracy  : 66.67%
```

With the parameters used in the experiment, XGBoost did not outperform Logistic Regression.

### KNN

```text
Training Accuracy : 78.97%
Testing Accuracy  : 65.81%
```

KNN provided reasonable baseline performance but did not achieve the best test accuracy.

### SVM

```text
Training Accuracy : 72.53%
Testing Accuracy  : 66.67%
```

The selected SVM configuration did not outperform the other models.

---

# 🧠 Parkinson's Disease Prediction

## Dataset

The Parkinson's dataset contains:

```text
195 observations
24 columns
```

The original dataset includes a patient identifier:

```text
name
```

which was removed before model development.

The target variable is:

```text
status
```

where:

```text
0 → Healthy
1 → Parkinson's disease
```

### Target Distribution

| Class     | Meaning     |   Count |
| --------- | ----------- | ------: |
| 0         | Healthy     |      48 |
| 1         | Parkinson's |     147 |
| **Total** |             | **195** |

The dataset is significantly imbalanced toward the positive Parkinson's class.

---

# 🧬 Parkinson's Features

The dataset contains biomedical voice measurements including:

* Fundamental frequency measurements
* Jitter measurements
* Shimmer measurements
* Noise-to-Harmonics Ratio (NHR)
* Harmonics-to-Noise Ratio (HNR)
* RPDE
* DFA
* Spread measures
* D2
* PPE

Examples include:

```text
MDVP:Fo(Hz)
MDVP:Fhi(Hz)
MDVP:Flo(Hz)
MDVP:Jitter(%)
MDVP:Jitter(Abs)
MDVP:RAP
MDVP:PPQ
Jitter:DDP
MDVP:Shimmer
MDVP:Shimmer(dB)
NHR
HNR
RPDE
DFA
spread1
spread2
D2
PPE
```

These measurements describe different characteristics of vocal frequency, variation, noise, and nonlinear dynamics.

---

# 🔎 Parkinson's EDA Findings

Correlation analysis showed several strong relationships among the voice-related features.

For example:

```text
MDVP:Jitter(%) ↔ MDVP:RAP ≈ 0.99
MDVP:Jitter(%) ↔ Jitter:DDP ≈ 0.99
MDVP:Shimmer ↔ Shimmer:DDA ≈ 0.99
spread1 ↔ PPE ≈ 0.96
```

The target variable also showed notable relationships with several features.

For example:

```text
status ↔ spread1 ≈ 0.56
status ↔ PPE ≈ 0.53
status ↔ spread2 ≈ 0.45
status ↔ MDVP:Shimmer ≈ 0.37
status ↔ HNR ≈ -0.36
status ↔ MDVP:Fo(Hz) ≈ -0.38
```

These correlations indicate that several voice characteristics may contain useful predictive information.

> **Note:** Correlation does not establish causation, and highly correlated predictors may introduce redundancy into a machine learning model.

---

# 📊 Parkinson's Model Performance

The provided Parkinson's notebook currently contains the data exploration, preprocessing and correlation analysis, but **does not include the final model-training and evaluation results**.

Therefore, model performance should be added after training the selected classifiers.

Recommended metrics:

| Metric             | Purpose                                    |
| ------------------ | ------------------------------------------ |
| Accuracy           | Overall percentage of correct predictions  |
| Precision          | Reliability of positive predictions        |
| Recall/Sensitivity | Ability to identify positive cases         |
| F1-Score           | Balance between precision and recall       |
| ROC-AUC            | Overall ranking/discrimination performance |
| Confusion Matrix   | Detailed prediction breakdown              |

### Recommended section after model training

```text
| Model | Training Accuracy | Testing Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | XX% | XX% | XX | XX | XX | XX |
| Decision Tree | XX% | XX% | XX | XX | XX | XX |
| Random Forest | XX% | XX% | XX | XX | XX | XX |
| XGBoost | XX% | XX% | XX | XX | XX | XX |
| SVM | XX% | XX% | XX | XX | XX | XX |
| KNN | XX% | XX% | XX | XX | XX | XX |
```

---

# 🫘 Kidney Disease Prediction

The Kidney Disease module follows the same overall machine learning philosophy:

```text
Kidney Dataset
      ↓
Data Cleaning
      ↓
Missing Value Treatment
      ↓
Categorical Encoding
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Multiple Classification Models
      ↓
Performance Evaluation
      ↓
Best Model Selection
```

The objective is to classify patients according to their kidney-disease status using relevant clinical and laboratory measurements.

### Kidney Model Performance

The final Kidney Disease model results can be reported using:

| Model               | Training Accuracy | Testing Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | ----------------: | ---------------: | --------: | -----: | -------: | ------: |
| Logistic Regression |                 — |                — |         — |      — |        — |       — |
| Decision Tree       |                 — |                — |         — |      — |        — |       — |
| Random Forest       |                 — |                — |         — |      — |        — |       — |
| XGBoost             |                 — |                — |         — |      — |        — |       — |
| KNN                 |                 — |                — |         — |      — |        — |       — |
| Gradient Boosting   |                 — |                — |         — |      — |        — |       — |
| SVM                 |                 — |                — |         — |      — |        — |       — |

> Replace the `—` values with the actual results from the Kidney Disease model experiments.

---

# 📊 Overall Model Comparison

The final project can be summarized using a consolidated performance table.

| Disease        | Best Model              | Test Accuracy | Precision |    Recall |  F1-Score | ROC-AUC |
| -------------- | ----------------------- | ------------: | --------: | --------: | --------: | ------: |
| 🫘 Kidney      | To be added             |             — |         — |         — |         — |       — |
| 🫀 Liver       | **Logistic Regression** |    **70.09%** | **0.67**¹ | **0.21**¹ | **0.31**¹ |       — |
| 🧠 Parkinson's | To be added             |             — |         — |         — |         — |       — |

¹ Positive-class (`1`) metrics for the Liver Disease Logistic Regression model.

---

# 🏆 Current Best Result

Based on the results currently available in this project:

### 🫀 Liver Disease

**Best Model: Logistic Regression**

```text
Training Accuracy : 74.03%
Testing Accuracy  : 70.09%
```

Positive-class metrics:

```text
Precision : 0.67
Recall    : 0.21
F1-Score  : 0.31
```

The Logistic Regression model achieved the highest testing accuracy among the seven evaluated liver-disease models.

However, the low positive-class recall demonstrates why **medical ML models should not be evaluated using accuracy alone**.

---

# 📏 Evaluation Metrics

The project uses several classification metrics.

## Accuracy

Accuracy measures the proportion of all predictions that are correct.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

It is useful for obtaining an overall performance measure but can be misleading when classes are imbalanced.

---

## Precision

Precision measures how many predicted positive cases are actually positive.

```text
Precision = TP / (TP + FP)
```

Higher precision means fewer false-positive predictions.

---

## Recall

Recall measures how many actual positive cases are correctly identified.

```text
Recall = TP / (TP + FN)
```

For disease prediction, recall can be particularly important because false negatives may represent patients who actually have the disease but are predicted as healthy.

---

## F1-Score

F1-score combines precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It is particularly useful when there is a class imbalance.

---

## Confusion Matrix

The confusion matrix provides four important values:

```text
True Positive
True Negative
False Positive
False Negative
```

This gives a more detailed view of model behavior than accuracy alone.

---

# ⚠️ Important Model Evaluation Considerations

The datasets used in this project are not perfectly balanced.

For example, the Liver dataset contains:

```text
416 → Class 0
167 → Class 1
```

while the Parkinson's dataset contains:

```text
48  → Class 0
147 → Class 1
```

Because of this, accuracy alone may not adequately represent model quality.

For a production-level healthcare application, the following should be considered:

* Recall/Sensitivity
* Specificity
* Precision
* F1-score
* ROC-AUC
* PR-AUC
* Confusion matrix
* Cross-validation
* Class weighting
* Threshold optimization
* Calibration
* External validation

---

# 🔧 Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Pickle

---

# 📁 Project Structure

A recommended repository structure is:

```text
Multiple_Disease_Prediction/
│
├── README.md
│
├── datasets/
│   ├── kidney_disease.csv
│   ├── liver_disease.csv
│   └── parkinsons.csv
│
├── notebooks/
│   ├── Kidney_Disease_Prediction.ipynb
│   ├── Liver_Disease_Prediction.ipynb
│   └── Parkinsons_Disease_Prediction.ipynb
│
├── models/
│   ├── kidney_model.sav
│   ├── liver_model.sav
│   └── parkinsons_model.sav
│
├── requirements.txt
│
└── app/
    └── app.py
```

---

# 🚀 Future Improvements

Several improvements can be made to increase the reliability and generalization of the models.

### 1. Cross-Validation

Instead of relying on a single train/test split, use stratified k-fold cross-validation to obtain more reliable estimates of model performance.

### 2. Hyperparameter Optimization

Apply techniques such as:

* GridSearchCV
* RandomizedSearchCV
* Bayesian optimization

to identify better model configurations.

### 3. Handle Class Imbalance

Techniques such as:

* Class weights
* SMOTE
* Random oversampling
* Random undersampling

can be investigated where appropriate.

### 4. Feature Selection

Highly correlated features may contain redundant information.

Feature selection techniques could be used to:

* Remove redundant variables
* Reduce model complexity
* Improve generalization
* Improve interpretability

### 5. Better Evaluation

Future experiments should include:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
PR-AUC
Specificity
Sensitivity
```

rather than relying primarily on accuracy.

### 6. Model Explainability

Explainable AI techniques such as SHAP or permutation importance could be used to understand which clinical features contribute most to predictions.

### 7. External Validation

A model trained on one dataset should ideally be evaluated on an independent dataset before being considered for real-world use.

---

# 💡 Key Learnings

This project demonstrates several important machine learning concepts:

* Medical data preprocessing
* Exploratory Data Analysis
* Missing-value treatment
* Categorical encoding
* Feature scaling
* Correlation analysis
* Binary classification
* Model comparison
* Confusion matrix analysis
* Precision/recall trade-offs
* Class imbalance
* Model persistence using Pickle

One of the key findings from the Liver Disease experiment is that the model with the highest training accuracy is **not necessarily the best model for unseen data**.

For example:

```text
Decision Tree
Training Accuracy → 100%
Testing Accuracy  → 66.67%
```

while:

```text
Logistic Regression
Training Accuracy → 74.03%
Testing Accuracy  → 70.09%
```

This illustrates the importance of evaluating models on unseen data and monitoring overfitting.

---

# ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The predictions generated by these machine learning models should **not be used as a medical diagnosis or as a replacement for consultation with a qualified healthcare professional**.

The datasets are limited in size and may not represent the broader population. Model performance on these datasets does not guarantee equivalent performance in real-world clinical settings.

---

# 👨‍💻 Project Summary

**Multiple Disease Prediction** demonstrates how machine learning can be applied to different healthcare classification problems using structured clinical and biomedical data.

The project covers three disease domains:

```text
🫘 Kidney Disease
       +
🫀 Liver Disease
       +
🧠 Parkinson's Disease
       ↓
Machine Learning Classification
```

The current Liver Disease experiments evaluated seven classification algorithms, with **Logistic Regression achieving the highest test accuracy of 70.09%**.

The project also highlights an important lesson for healthcare machine learning:

> **A high accuracy score does not necessarily mean that a model is clinically useful.**

Metrics such as recall, precision, F1-score, specificity, ROC-AUC, and the consequences of false-negative predictions must also be considered.

---
