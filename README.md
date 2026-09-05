# Multiple Disease Prediction (Kidney, Liver & Parkinson's)

A machine learning project that predicts the likelihood of **Chronic Kidney Disease (CKD)**, **Liver Disease**, and **Parkinson's Disease** from routine clinical/lab measurements. Each disease has its own dedicated data pipeline and classification model, trained and evaluated independently, with the best-performing model for each condition serialized for downstream use (e.g. a Streamlit/Flask web app).

> This README documents all three modules — **Kidney Disease**, **Liver Disease**, and **Parkinson's Disease** — including data preparation, modeling approach, and real benchmark results for each.

---

## 📌 Problem Statement

Early detection of chronic diseases like kidney disease, liver disease, and Parkinson's disease significantly improves patient outcomes, but diagnosis often depends on interpreting multiple lab values together — something that's easy to miss in a busy clinical setting or entirely inaccessible for self-screening.

The goal of this project is to build **supervised binary classification models** that take a patient's clinical and laboratory measurements as input and predict whether they are likely to have the disease in question, so that:

- At-risk patients can be flagged earlier for confirmatory testing.
- The models can be wrapped into a simple prediction interface (single combined app: "Multiple Disease Prediction System").
- Each disease's best model is chosen on the basis of a fair, multi-metric comparison rather than accuracy alone.

**Target variables:**
- **Kidney module:** `classification` → `1` = CKD (chronic kidney disease), `0` = not CKD.
- **Liver module:** `Dataset` → after remapping, `0` = liver disease patient, `1` = not a liver disease patient. ⚠️ **Note the inverted convention** here versus the kidney module — see the caveat under [Known Limitations](#️-known-limitations--honest-caveats).
- **Parkinson's module:** `status` → `1` = Parkinson's, `0` = healthy. Notably imbalanced: 147 positive vs. 48 negative (~75% / 25%) — more skewed than either the kidney or liver datasets.

---

## 🗂️ Dataset

| Disease | Source | Records | Features | Target |
|---|---|---|---|---|
| Kidney Disease | UCI CKD Dataset (`kidney_disease.csv`) | 400 | 24 clinical/lab features | `classification` (ckd / notckd) |
| Liver Disease | Indian Liver Patient Dataset (`liver_disease.csv`) | 583 | 10 clinical/lab features | `Dataset` (liver patient / not) |
| Parkinson's Disease | UCI Parkinson's Dataset (`parkinsons.csv`) | 195 | 22 voice-measurement features | `status` (Parkinson's / healthy) |

**Kidney dataset features** include: age, blood pressure (`bp`), specific gravity (`sg`), albumin (`al`), sugar (`su`), red/white blood cell counts, blood glucose (`bgr`), blood urea (`bu`), serum creatinine (`sc`), sodium/potassium, hemoglobin, packed cell volume, and comorbidity flags (`htn`, `dm`, `cad`, `appet`, `pe`, `ane`).

**Liver dataset features** include: age, gender, total/direct bilirubin, alkaline phosphotase, alamine/aspartate aminotransferase (liver enzymes), total proteins, albumin, and the albumin-to-globulin ratio.

**Parkinson's dataset features** are all derived from sustained-vowel voice recordings: fundamental frequency measures (`MDVP:Fo/Fhi/Flo(Hz)`), several jitter and shimmer variants (frequency/amplitude perturbation), noise-to-harmonics ratios (`NHR`, `HNR`), and nonlinear dynamical measures (`RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`). The `name` column (a recording ID) was dropped as non-predictive.

---

## 🔍 Approach — Kidney Disease Module

The pipeline follows a standard, reproducible ML workflow:

### 1. Data Loading & Initial Inspection
- Loaded the raw CSV, inspected shape, dtypes, and class balance (`ckd`: 250 vs `notckd`: 150 — a mild class imbalance).
- Dropped the non-predictive `id` column.

### 2. Data Type Correction
- Several numeric-looking columns (`pcv`, `wc`, `rc`) were stored as `object` due to stray non-numeric entries. Converted these with `pd.to_numeric(..., errors='coerce')` so bad values become `NaN` instead of silently corrupting the model.

### 3. Exploratory Data Analysis (EDA)
- Plotted distributions for all numerical columns (`sns.distplot`) to check skew and spread.
- Plotted count plots for all categorical columns (`sns.countplot`) to check category balance.
- Computed a full correlation matrix and visualized it as a heatmap to understand which features move together with `classification` (e.g. `sg`, `hemo`, `pcv`, `htn`, `dm`, and `al` showed the strongest correlations with CKD status).

### 4. Missing Value Treatment
The dataset had substantial missingness across many columns (`rbc`: 152 missing, `rc`: 131, `wc`: 106, `pot`: 88, `sod`: 87, etc.). Two imputation strategies were used depending on column type:
- **Numerical columns:** random sampling imputation — missing values are filled by randomly sampling from the column's own observed (non-null) distribution, preserving the original variance/shape better than a simple mean/median fill.
- **Categorical columns:** mode imputation (with `rbc` and `pc` also using random sampling first, since they had the highest missingness).

After this step, all 25 columns had zero missing values.

### 5. Feature Encoding
- Binary categorical columns (`rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`) were label-encoded to `0`/`1` via a manual mapping dictionary.
- The target `classification` was mapped from `{ckd, notckd}` to `{1, 0}`.

### 6. Train/Test Split
- `train_test_split` with `test_size=0.2`, `random_state=0`, giving 320 training rows and 80 test rows.

### 7. Model Benchmarking
Seven classification algorithms were trained on identical train/test splits and compared on accuracy, confusion matrix, precision/recall/F1, and ROC-AUC:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier (`max_depth=10`, `n_estimators=400`, `min_samples_split=7`)
- XGBoost Classifier
- K-Nearest Neighbors
- Gradient Boosting Classifier
- Support Vector Machine (RBF kernel, `C=15`, `gamma=0.0001`)

### 8. Model Selection & Persistence
- Compared all models on test accuracy and ROC-AUC (see results below).
- The selected model was serialized with `pickle` (`kidney_model.sav`) for reuse in a prediction app.

---

## 📊 Model Performance & Metrics (Kidney Disease)

### Accuracy & ROC-AUC comparison

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| **Gradient Boosting** | **98.75%** | **98.21%** |
| Random Forest | 97.50% | 96.43% |
| Decision Tree | 96.25% | 96.29% |
| Logistic Regression | 88.75% | 88.05% |
| SVM | 72.50% | 68.96% |
| KNN | 66.25% | 65.80% |
| XGBoost | 65.00% | 50.00% |

> **Note on XGBoost:** its accuracy (65%) matches the no-information baseline of always predicting the majority class in the test split, and its ROC-AUC of exactly 50% confirms it wasn't discriminating between classes at all. This is a symptom of the learning rate being set far too low (`learning_rate=0.001`) for only 100 boosting rounds — the model was effectively undertrained, not a fair reflection of what XGBoost can do on this data. Retuning it (higher learning rate and/or more estimators) would very likely bring it in line with Random Forest/Gradient Boosting.

### Best model: Gradient Boosting Classifier

**Test set confusion matrix:**

| | Predicted: notckd | Predicted: ckd |
|---|---|---|
| **Actual: notckd** | 27 | 1 |
| **Actual: ckd** | 0 | 52 |

**Classification report:**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| notckd (0) | 1.00 | 0.96 | 0.98 |
| ckd (1) | 0.98 | 1.00 | 0.99 |
| **Accuracy** | | | **0.9875** |

**Why Gradient Boosting was chosen:**
- Highest test accuracy (98.75%) and highest ROC-AUC (98.21%) among all seven models.
- Zero false negatives on the test set (every actual CKD case was correctly caught) — clinically, this is the more important error to avoid, since missing a real CKD case is costlier than a false alarm.
- Training accuracy of 100% alongside strong test performance suggests some overfitting risk; this is flagged as a follow-up item (see below) rather than treated as a fully closed case.

### Visualizations produced
- `roc_kidney.jpeg` — ROC curves for all 7 models overlaid, with AUC in the legend.
- `PE_kidney.jpeg` — grouped bar chart comparing Accuracy (%) vs ROC-AUC (%) across all models.

---

## 🔍 Approach — Liver Disease Module

### 1. Data Loading & Initial Inspection
- Loaded `liver_disease.csv` (583 rows, 11 columns) and checked class balance: `Dataset == 1` (liver patient): 416 rows, `Dataset == 2` (not a patient): 167 rows — a more pronounced imbalance (~71% / 29%) than the kidney dataset.

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for all numerical columns and a count plot for the single categorical column (`Gender`).
- Correlation heatmap: `Alamine_Aminotransferase` and `Aspartate_Aminotransferase` (liver enzymes) are strongly correlated with each other (0.79), as are `Total_Protiens` and `Albumin` (0.78), and `Albumin` with `Albumin_and_Globulin_Ratio` (0.69) — expected, since these are physiologically linked lab values. No single feature is strongly correlated with the target on its own; the strongest are `Direct_Bilirubin` (-0.25) and `Albumin_and_Globulin_Ratio` (0.16), suggesting the disease signal here is more diffuse across features than in the kidney dataset.
- A pairplot colored by `Dataset` to visually inspect class separability.

### 3. Missing Value Treatment
- Only one column had missing values: `Albumin_and_Globulin_Ratio` (4 rows, <1%), imputed with the column mean.

### 4. Target Remapping
- The raw `Dataset` column uses `1` = liver patient, `2` = not a patient. This was remapped to `0` = liver patient, `1` = not a patient via `replace([2,1],[1,0])`.
- **This is the opposite convention from the kidney module** (where `1` = disease-positive). It's functionally fine for training, but anyone reusing these labels downstream (e.g. in a combined app) needs to explicitly handle the fact that "positive class" doesn't mean the same numeric label across modules.

### 5. Feature Encoding
- `Gender` (the only categorical column) was one-hot encoded via `pd.get_dummies(..., drop_first=True)`, producing a single `Gender_Male` column, cast to `int32`.

### 6. Feature Scaling
- All features (including the binary `Gender_Male` column) were standardized with `StandardScaler` before modeling. Scaling a binary 0/1 column alongside continuous lab values is not strictly necessary and is called out below as a minor inconsistency rather than a correctness issue, since scaling a binary variable doesn't change tree-based model behavior but does change coefficient magnitude for linear/SVM models.

### 7. Train/Test Split
- `train_test_split` with `test_size=0.2`, `random_state=0`, giving 466 training rows and 117 test rows.

### 8. Model Benchmarking
The same seven algorithms as the kidney module were trained and compared on the scaled features: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Gradient Boosting, and SVM.

### 9. Model Selection & Persistence
- The best-performing model (Logistic Regression) was serialized with `pickle` (`liver_model.sav`).

---

## 📊 Model Performance & Metrics (Liver Disease)

### Accuracy comparison

| Model | Test Accuracy |
|---|---|
| **Logistic Regression** | **70.09%** |
| Gradient Boosting | 69.23% |
| Decision Tree | 66.67% |
| XGBoost | 66.67% |
| SVM | 66.67% |
| Random Forest | 65.81% |
| KNN | 65.81% |

> **Reality check on these numbers:** the majority class (`Dataset == 0`, liver patients) makes up 78 of the 117 test rows — a "predict-the-majority-class-always" baseline already scores **66.7% accuracy**. Decision Tree, XGBoost, and SVM all land almost exactly on that number, which is a strong signal that they learned little beyond the class imbalance rather than genuine liver-disease signal. Only Logistic Regression and Gradient Boosting show any real lift above that floor, and even then it's modest (+3–4 points). **Accuracy alone is a misleading metric on this dataset** — precision/recall per class (below) tell a more honest story.

### Best model: Logistic Regression

**Test set confusion matrix** (`0` = liver disease patient, `1` = not a patient):

| | Predicted: disease (0) | Predicted: healthy (1) |
|---|---|---|
| **Actual: disease (0)** | 74 | 4 |
| **Actual: healthy (1)** | 31 | 8 |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| disease (0) | 0.70 | 0.95 | 0.81 | 78 |
| healthy (1) | 0.67 | 0.21 | 0.31 | 39 |
| **Accuracy** | | | **0.7009** | 117 |

**Reading these numbers honestly:**
- The model is good at not missing disease cases: **95% recall on the disease class** means only 4 of 78 actual liver-disease patients were missed — clinically the more important error to minimize.
- But it's **poor at correctly identifying healthy patients**: only 21% recall on the healthy class means 31 of 39 truly healthy people were misclassified as having liver disease. In a real screening tool this would mean a lot of unnecessary follow-up testing for healthy people.
- This pattern — high recall on the majority class, low recall on the minority class — is the classic fingerprint of class imbalance rather than a model that has learned a clean decision boundary. It was chosen as "best" here only because it edges out the others on accuracy; it would **not** be the honest choice if minimizing false positives on healthy patients mattered more than raw accuracy.

---

## 🔍 Approach — Parkinson's Disease Module

### 1. Data Loading & Initial Inspection
- Loaded `parkinsons.csv` (195 rows, 24 columns) and checked class balance: `status == 1` (Parkinson's): 147 rows, `status == 0` (healthy): 48 rows — the most imbalanced of the three datasets (~75% / 25%).
- Dropped the `name` column (a per-recording identifier, e.g. `phon_R01_S01_1`), leaving 22 numeric features plus the target.
- Unlike the kidney and liver datasets, this one is **already fully numeric and has zero missing values** (`df.info()` shows 195 non-null entries in every column) — no type coercion or imputation is needed here.

### 2. Exploratory Data Analysis (EDA)
- Distribution plots (`sns.distplot`) across the numeric voice-measurement features.
- A full correlation matrix and heatmap. Notable patterns:
  - **Very strong internal collinearity** among the jitter measures (`MDVP:Jitter(%)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` are all pairwise correlated above 0.90, with `MDVP:RAP` and `Jitter:DDP` at a near-perfect 1.00) and similarly among the shimmer measures (`MDVP:Shimmer`, `Shimmer:APQ3`, `Shimmer:DDA` also near 0.99–1.00). These are, in several cases, near-duplicate derived features rather than independent signals.
  - **`spread1` (0.56) and `PPE` (0.53) show the strongest individual correlation with `status`**, followed by `spread2` (0.45) — all three are nonlinear/dynamical voice measures, consistent with the published literature on Parkinson's voice biomarkers.
  - `HNR` (harmonics-to-noise ratio) is **negatively** correlated with `status` (-0.36), i.e. lower voice-signal "cleanliness" is associated with Parkinson's, which is clinically intuitive.
- A pairplot colored by `status` to visually inspect class separability across feature pairs.

### 3. Missing Value Check
- Confirmed 0 missing values across all 22 features — no imputation needed.

### 4. Feature Scaling — computed but not actually used
- `StandardScaler` is fit and applied to `X`, producing `x_scaled`.
- **However, the train/test split and every model below are trained on the original unscaled `X`, not `x_scaled`.** This is a real bug in the notebook, not a stylistic choice: scaling was clearly intended (it's the only module of the three that scales, and it's the module where features vary the most in magnitude — e.g. `MDVP:Fo(Hz)` is in the hundreds while `MDVP:Jitter(Abs)` is in the ten-thousandths), but the scaled frame is never passed into `train_test_split`. This particularly affects distance-based models (KNN, SVM), which are sensitive to feature scale.

### 5. Train/Test Split
- `train_test_split` with `test_size=0.2`, `random_state=0`, giving 156 training rows and 39 test rows.

### 6. Model Benchmarking
The same seven algorithms as the other two modules were trained: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Gradient Boosting, and SVM.

### 7. Model Selection & Persistence
- As in the kidney and liver modules, the pickled model (`parkinson_model.sav`) is **hardcoded to Logistic Regression** (`model = lr`) regardless of which model actually scored best in the comparison table — see the caveat below, since this pattern repeats across all three modules.
- The ROC curve and performance-comparison plots reuse code copied from the kidney module without updating the labels: the plot titles still read *"ROC - Kidney Disease Prediction"* and *"Performance Evaluation - Kidney Disease Prediction"*, and the files are saved to `roc_kidney.jpeg` / `PE_kidney.jpeg` — meaning, if run in the same working directory as the kidney notebook, **this would silently overwrite the kidney module's visualizations** rather than creating separate Parkinson's ones.

---

## 📊 Model Performance & Metrics (Parkinson's Disease)

*(Computed by running the notebook's exact code against the real UCI Parkinson's dataset, since this module's code included final print statements. Note: the notebook does not set `random_state` on `RandomForestClassifier`, so its exact numbers will vary slightly between runs — the values below are from one run.)*

### Accuracy & ROC-AUC comparison

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| **Gradient Boosting** | **94.87%** | **93.28%** |
| Random Forest | 92.31% | 88.28% |
| Logistic Regression | 89.74% | 83.28% |
| SVM | 89.74% | 83.28% |
| KNN | 89.74% | 83.28% |
| Decision Tree | 84.62% | 86.55% |
| XGBoost | 74.36% | 50.00% |

> **XGBoost shows the exact same failure mode as in the kidney and liver modules**: its 74.36% accuracy matches the majority-class baseline (29 of 39 test rows are `status == 1`), and its ROC-AUC of 50.00% confirms it isn't discriminating at all. Its confusion matrix (`[[0, 10], [0, 29]]`) shows it predicted "Parkinson's" for every single test row. This is the third module in a row where `learning_rate=0.001` with only 100 rounds has left XGBoost undertrained — a clear, fixable, repo-wide pattern rather than three unrelated issues.

### Best model: Gradient Boosting Classifier

**Test set confusion matrix** (`0` = healthy, `1` = Parkinson's):

| | Predicted: healthy (0) | Predicted: Parkinson's (1) |
|---|---|---|
| **Actual: healthy (0)** | 9 | 1 |
| **Actual: Parkinson's (1)** | 1 | 28 |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| healthy (0) | 0.90 | 0.90 | 0.90 | 10 |
| Parkinson's (1) | 0.97 | 0.97 | 0.97 | 29 |
| **Accuracy** | | | **0.9487** | 39 |

**Why this looks like a genuinely good result (with a caveat):**
- Unlike the liver module, Gradient Boosting here shows **balanced performance across both classes** (90% recall on healthy, 97% recall on Parkinson's) rather than just riding the majority class — a real signal that the model learned something meaningful from the voice features.
- That said, the **test set is only 39 rows**, so a single misclassification shifts accuracy by ~2.5 points; treat this as an encouraging result on a small sample, not a validated clinical-grade number. Cross-validation across the full 195 rows would give a much more trustworthy estimate.
- As noted above, the model that actually gets saved to disk (`parkinson_model.sav`) is Logistic Regression (89.74%), not the better-performing Gradient Boosting — worth fixing before this feeds into a prediction app.

---

## 🔁 Cross-Module Observations

Three separate notebooks, three separate diseases — but the same handful of issues show up in each, which suggests these are copy-paste patterns worth fixing once, centrally, rather than three times:

1. **The saved model is always hardcoded to Logistic Regression** (`model = lr; pickle.dump(...)`), regardless of which model actually won the accuracy comparison in that module. In the kidney and Parkinson's modules, Gradient Boosting scored higher; in the liver module, Logistic Regression genuinely was the best, so that one happens to be correct by coincidence.
2. **XGBoost is undertrained in all three modules** (`learning_rate=0.001`, `n_estimators=100`), and in all three cases its accuracy lands almost exactly on the majority-class baseline with a ROC-AUC near 50% — i.e., it isn't learning anything. The fix is the same in each: raise the learning rate and/or number of estimators.
3. **Accuracy is reported as the headline metric everywhere, but is misleading on two of the three imbalanced datasets** (liver and Parkinson's) where several models are simply riding the majority class. A shared model-selection function that ranks by F1 or balanced accuracy instead of raw accuracy would catch this automatically across all three modules.
4. **No `random_state`** is set for `RandomForestClassifier` in the Parkinson's module (unlike the other two), making its reported numbers non-reproducible run to run.

---

## ⚠️ Known Limitations / Honest Caveats

**Kidney module:**
- **Small dataset (400 rows):** an 80-row test set means each misclassified sample shifts accuracy by ~1.25 percentage point — treat the exact numbers above as indicative, not precise.
- **Random-sampling imputation** for columns with heavy missingness (e.g. `rbc` at 38% missing) can understate how much the model is relying on imputed values.
- **XGBoost was undertrained** in this run (`learning_rate=0.001`, only 100 rounds — see note above) and needs re-tuning before it can be fairly compared to the other ensemble models.
- **Class imbalance** (250 CKD vs 150 not-CKD) wasn't explicitly addressed (e.g. via class weighting or resampling) — worth testing whether it changes model ranking.
- Some models (Decision Tree, Gradient Boosting) show **train accuracy of 100%**, a classic overfitting signal on a small dataset; cross-validation (not just a single train/test split) would give a more reliable performance estimate.

**Liver module:**
- **Accuracy is a misleading metric here.** With a 78/39 class split in the test set, "always predict liver disease" already scores 66.7% — three of the seven models (DT, XGBoost, SVM) essentially collapsed to that baseline rather than learning real signal, and even the "best" model (Logistic Regression at 70.9%) only modestly beats it.
- **Same undertrained-XGBoost issue** as the kidney module (`learning_rate=0.001`): its confusion matrix (`[[78, 0], [39, 0]]`) shows it predicted the majority class for every single test row.
- **Inverted target convention** (`0` = disease, `1` = healthy) versus the kidney module's (`1` = disease) — a real risk of confusion if these models are combined into one app without relabeling for consistency.
- **Standardizing the one-hot `Gender_Male` column** alongside continuous features is a minor inconsistency; it doesn't affect tree-based models but changes how its coefficient reads for Logistic Regression/SVM.
- **Class imbalance was not addressed** (no class weighting, no resampling) — given how much it appears to be driving the accuracy numbers, this is the single highest-value fix for this module.
- The chosen "best" model has only **21% recall on the minority (healthy) class** — worth explicitly deciding whether accuracy or a cost-sensitive metric (e.g. balanced accuracy, F1 on the minority class) should actually drive model selection here.

**Parkinson's module:**
- **Feature scaling was computed but never used** — `StandardScaler` is fit on `X` but the unscaled `X` is what actually goes into `train_test_split` and every model. This is a genuine bug, not a design choice, and most likely to affect KNN and SVM (both distance/margin-based and sensitive to feature scale).
- **`RandomForestClassifier` has no `random_state` set**, so its reported accuracy will vary somewhat between runs (I saw 92.3–94.9% across two runs myself while producing these numbers) — worth pinning down for reproducibility.
- **Very small test set (39 rows)** — even more than the other two modules, treat these percentages as indicative rather than precise; a single flipped prediction moves accuracy by ~2.5 points.
- **The ROC/performance-comparison plotting code was copied from the kidney module without updating labels or filenames** — running it as-is will save over `roc_kidney.jpeg` and `PE_kidney.jpeg` with titles that still say "Kidney Disease Prediction," rather than producing dedicated Parkinson's plots.
- Same **undertrained XGBoost** and **hardcoded Logistic-Regression-gets-saved** issues as the other two modules — see [Cross-Module Observations](#-cross-module-observations) above.

---

## 🚀 Next Steps

- [ ] **Fix the hardcoded model-save bug** in all three modules — save whichever model actually wins the comparison table, not always Logistic Regression.
- [ ] **Fix the Parkinson's scaling bug** — actually use `x_scaled` in the train/test split instead of computing it and discarding it.
- [ ] **Fix the Parkinson's plotting bug** — the ROC/performance-comparison code still has kidney-module titles and filenames; give it its own labels and filenames so it doesn't overwrite the kidney plots.
- [ ] Set `random_state` on `RandomForestClassifier` in the Parkinson's module for reproducible results.
- [ ] Replace the single train/test split with k-fold (or stratified k-fold, given the class imbalance in all three datasets) cross-validation for more robust metrics.
- [ ] Re-tune XGBoost hyperparameters (learning rate, n_estimators) in all three modules — it's currently undertrained everywhere.
- [ ] Address class imbalance across all three modules (e.g. `class_weight='balanced'`, SMOTE) and re-evaluate — this looks especially important for the liver and Parkinson's datasets.
- [ ] Standardize the target-label convention across modules (`1` = disease-positive everywhere) before combining into one app.
- [ ] Add ROC-AUC and a precision/recall comparison chart for the liver module (currently only computed for kidney and Parkinson's).
- [ ] Investigate dropping or combining the highly collinear jitter/shimmer features in the Parkinson's dataset before modeling.
- [ ] Build a unified Streamlit/Flask app that loads all three pickled models and lets a user pick a disease and enter their values for prediction.
- [ ] Add feature importance plots (not just accuracy/ROC) so predictions are explainable to a clinician.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `plotly` · `scikit-learn` · `XGBoost` · `pickle`

---

## 📁 Repository Structure

```
Multiple-Disease-Prediction/
├── kidney/
│   ├── kidney_disease.csv
│   ├── kidney_data1.csv          # cleaned/encoded dataset
│   ├── kidney_disease_prediction.ipynb
│   ├── kidney_model.sav          # pickled Gradient Boosting model
│   ├── roc_kidney.jpeg
│   └── PE_kidney.jpeg
├── liver/
│   ├── liver_disease.csv
│   ├── liver_data1.csv           # cleaned/encoded dataset
│   ├── liver_disease_prediction.ipynb
│   └── liver_model.sav           # pickled Logistic Regression model
├── parkinsons/
│   ├── parkinsons.csv
│   ├── parkinso_data1.csv         # cleaned dataset (typo in original filename kept as-is)
│   ├── parkinsons_disease_prediction.ipynb
│   └── parkinson_model.sav        # pickled model (currently Logistic Regression — see Next Steps)
├── app.py                        # combined prediction app (planned)
├── requirements.txt
└── README.md

## 👥 Authors

- Your Name - Karthikeyan C, Aspiring Data Scientist

## 🙏 Acknowledgments

- Streamlit community

## 📞 Contact

Project Link: [https://github.com/KarthikeyanC95/Multiple_Disease_Prediction](https://github.com/KarthikeyanC95/Multiple_Disease_Prediction)

---
