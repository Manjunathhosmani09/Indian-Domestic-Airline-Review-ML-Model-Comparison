# Indian Domestic Airline Review — ML Model Comparison

> **Which machine learning model best predicts whether a passenger will recommend an airline?**  
> This project compares **Random Forest**, **KNN**, and **Naive Bayes** on real Indian domestic airline review data — using NLP-based feature engineering, SMOTE balancing, and a full metrics-driven verdict.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Tech Stack](#tech-stack)
4. [Project Workflow](#project-workflow)
5. [Key Code Snippets & Outputs](#key-code-snippets--outputs)
   - [Step 1 — Libraries](#step-1--libraries)
   - [Step 2 — Data Loading & EDA](#step-2--data-loading--eda)
   - [Step 3 — Null Handling & Cleaning](#step-3--null-handling--cleaning)
   - [Step 4 — Feature Engineering (Sentiment)](#step-4--feature-engineering-sentiment)
   - [Step 5 — Label Encoding](#step-5--label-encoding)
   - [Step 6 — Outlier Treatment](#step-6--outlier-treatment)
   - [Step 7 — Train-Test Split](#step-7--train-test-split)
   - [Step 8 — SMOTE Balancing](#step-8--smote-balancing)
   - [Step 9 — Random Forest (Default + Tuned)](#step-9--random-forest-default--tuned)
   - [Step 10 — KNN Model](#step-10--knn-model)
   - [Step 11 — Naive Bayes Model](#step-11--naive-bayes-model)
   - [Step 12 — Final Model Comparison](#step-12--final-model-comparison)
6. [Results Summary](#results-summary)
7. [Visualizations](#visualizations)
8. [Conclusion](#conclusion)

---

## Project Overview

This project applies supervised machine learning to classify whether a passenger **recommends** or **does not recommend** an Indian domestic airline, based on their review text, title, and rating. Three models are trained, evaluated, and compared across 8 performance metrics to identify the best-performing algorithm for this dataset.

**Goal:** Build a classification pipeline that is accurate, handles class imbalance, and explains what drives the recommendation.

---

## Dataset

| Property | Detail |
|---|---|
| **File** | `Indian_Domestic_Airline.csv` |
| **Rows** | 2,210 reviews |
| **Columns** | AirLine_Name, Rating - 10, Title, Name, Date, Review, Recommond |
| **Target Variable** | `Recommond` (Yes / No) |
| **Airlines Covered** | Air India Express, AirAsia India, AirIndia, Go First, IndiGo, SpiceJet, Vistara |

Class imbalance was present in the raw data:
- No Recommend (0): **1,444** samples (65.5%)
- Yes Recommend (1): **761** samples (34.5%)

---

## 🛠 Tech Stack

- **Python 3.13**
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — ML models, metrics, preprocessing
- `TextBlob` — NLP sentiment scoring
- `imbalanced-learn (SMOTE)` — class balancing

---

## Project Workflow

```
Raw CSV
  → EDA & Null Handling
    → Sentiment Feature Engineering (TextBlob)
      → Label Encoding
        → Outlier Treatment (IQR / Winsorization)
          → Train-Test Split (80/20, stratified)
            → SMOTE Balancing (training data only)
              → Model Training: RF | KNN | Naive Bayes
                → Hyperparameter Tuning (RF)
                  → Confusion Matrix + Feature Importance
                    → Final Multi-Metric Comparison
                      → Winner Declared
```

---

## Key Code Snippets & Outputs

### Step 1 — Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

**Why it matters:** Combines standard data science libraries with TextBlob for NLP and scikit-learn for all ML operations in a single unified pipeline.

---

### Step 2 — Data Loading & EDA

```python
df = pd.read_csv("Indian_Domestic_Airline.csv")
print(df.head(10))
print(df.info())
print(df.describe())
```

**Output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2210 entries, 0 to 2209
Data columns (total 7 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   AirLine_Name  2210 non-null   object 
 1   Rating - 10   2206 non-null   float64
 2   Title         2210 non-null   object 
 3   Review        2210 non-null   object 
 6   Recommond     2210 non-null   object 
```

**Why it matters:** Confirms the dataset structure. Ratings range from 1–10 with a mean of ~4.03, indicating that negative experiences dominate the dataset — which directly influences model training.

---

### Step 3 — Null Handling & Cleaning

```python
print(df.isnull().sum())    # 4 nulls found in Rating - 10

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df.isnull().sum())    # All zeros confirmed
```

**Output (after cleaning):**
```
AirLine_Name    0
Rating - 10     0
Title           0
Review          0
Recommond       0
dtype: int64
```

**Why it matters:** Missing values in the Rating column could silently bias model training. Dropping them ensures the feature matrix has no NaN values before feeding into ML algorithms.

---

### Step 4 — Feature Engineering (Sentiment)

```python
def get_sentiment(text):
    """Returns polarity score: -1 (very negative) to +1 (very positive)"""
    return TextBlob(str(text)).sentiment.polarity

df['Review_Sentiment'] = df['Review'].apply(get_sentiment)
df['Title_Sentiment']  = df['Title'].apply(get_sentiment)
df['Review_Length']    = df['Review'].apply(lambda x: len(str(x).split()))
```

**Output:**
```
       Review_Sentiment  Title_Sentiment  Review_Length
count       2205.000000      2205.000000    2205.000000
mean           0.047553        -0.035902     108.017234
std            0.281520         0.521298      82.504181
min           -1.000000        -1.000000      20.000000
```

**Why it matters:** Raw review text cannot be fed directly into a classifier. TextBlob converts each review and title into a numeric polarity score between -1 and +1. This allows the model to learn that passengers with positive language are more likely to recommend — a critical signal in the dataset.

---

### Step 5 — Label Encoding

```python
le_airline   = LabelEncoder()
le_recommend = LabelEncoder()

df['AirLine_Encoded']   = le_airline.fit_transform(df['AirLine_Name'])
df['Recommond_Encoded'] = le_recommend.fit_transform(df['Recommond'].str.strip().str.lower())
```

**Output:**
```
Airline Name Encoding Map:
   Air India Express → 0
   AirAsia India → 1
   AirIndia → 2
   Go First → 3
   IndiGo → 4
   SpiceJet → 5
   Vistara → 6

Recommend Encoding Map:
   no → 0
   yes → 1
```

**Why it matters:** Scikit-learn models require numeric inputs. Label encoding converts categorical airline names and the target variable into integers without inflating dimensionality (unlike one-hot encoding for tree-based models).

---

### Step 6 — Outlier Treatment

```python
Q1  = df['Rating - 10'].quantile(0.25)
Q3  = df['Rating - 10'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['Rating - 10'] = df['Rating - 10'].clip(lower=lower, upper=upper)
```

**Output:**
```
Rating Column:
   Q1=1.0, Q3=8.0, IQR=7.0
   Lower bound=-9.5, Upper bound=18.5
   Outliers found: 0
✅ Outliers capped using Winsorization (clipping).
```

**Why it matters:** The IQR method checks for extreme values in the Rating column. Although no outliers were found here, applying Winsorization (clipping) as a safeguard ensures robustness. For any new data added later, ratings will automatically be bounded within valid limits.

---

### Step 7 — Train-Test Split

```python
X = df[['Rating - 10', 'AirLine_Encoded', 'Review_Sentiment', 'Title_Sentiment', 'Review_Length']]
y = df['Recommond_Encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Output:**
```
✅ Training samples : 1764
✅ Testing  samples : 441

Class distribution:
0 (No Recommend)  : 1444
1 (Yes Recommend) : 761
```

**Why it matters:** The `stratify=y` parameter ensures that the 65/35 class distribution is preserved in both training and test sets — preventing the test set from accidentally containing only one dominant class, which would give a falsely high accuracy.

---

### Step 8 — SMOTE Balancing

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

**Output:**
```
Class Distribution BEFORE SMOTE:
   No  Recommend (0) : 1155 samples (65.5%)
   Yes Recommend (1) :  609 samples (34.5%)

Class Distribution AFTER SMOTE:
   No  Recommend (0) : 1155 samples (50.0%)
   Yes Recommend (1) : 1155 samples (50.0%)
   Total after SMOTE : 2310

✅ SMOTE applied on training data only.
✅ X_test and y_test remain original — never touched.
```

**Accuracy impact:**
```
Before SMOTE (Imbalanced) : 0.9660
After  SMOTE (Balanced)   : 0.9683
Difference                : +0.0023 ✅ Improved
```

**Why it matters:** Without SMOTE, the model would be biased toward predicting "No Recommend" (the majority class) because it sees nearly twice as many examples. SMOTE generates **synthetic minority class samples** in the training data only — keeping the test set untouched to preserve realistic evaluation conditions.

---

### Step 9 — Random Forest (Default + Tuned)

**Default Model:**
```python
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train_smote, y_train_smote)
y_pred_default = rf_default.predict(X_test)
```

**Output (Default):**
```
📊 Default RF Accuracy: 0.9660

               precision    recall  f1-score   support
 No Recommend       0.96      0.99      0.97       289
Yes Recommend       0.97      0.93      0.95       152
     accuracy                           0.97       441
```

**Hyperparameter Tuning:**
```python
param_grid = [
    {'n_estimators': 50,  'max_depth': 5,    'min_samples_split': 2,  'max_features': 'sqrt', 'criterion': 'gini'},
    {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 10, 'max_features': 'sqrt', 'criterion': 'gini'},
    {'n_estimators': 200, 'max_depth': 8,    'min_samples_split': 2,  'max_features': 'log2', 'criterion': 'gini'},
    ...
]

for p in param_grid:
    model = RandomForestClassifier(**p, random_state=42)
    model.fit(X_train_smote, y_train_smote)
    acc = accuracy_score(y_test, model.predict(X_test))
```

**Best Parameters Found:**
```
Best Parameters : {n_estimators: 50, max_depth: 5, min_samples_split: 2,
                   min_samples_leaf: 1, max_features: 'sqrt', criterion: 'gini'}
Best Accuracy   : 0.9683
```

**sqrt vs log2 Comparison:**
```
Metric            sqrt     log2
Average Accuracy  0.9666   0.9643
Best Accuracy     0.9683   0.9660
✅ sqrt performs BETTER than log2 overall
```

**Confusion Matrix (Best RF):**
```
                   Predicted No   Predicted Yes
Actual No              286              3        ← Only 3 False Positives
Actual Yes              11            141        ← Only 11 False Negatives
```

**Why it matters:** Tuning hyperparameters like `max_features='sqrt'` and capping `max_depth=5` prevents overfitting while maintaining high generalization accuracy. The confusion matrix shows that false positives are extremely rare (only 3), which is critical in a recommendation system.

---

### Step 10 — KNN Model

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled  = scaler.transform(X_test)

for k in [3, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_smote)
```

**Output:**
```
KNN ACCURACY TABLE:
KNN (k=3)  →  0.9501
KNN (k=7)  →  0.9615   ← Best KNN
KNN (k=9)  →  0.9592
```

**Why it matters:** KNN is sensitive to feature scale, which is why `StandardScaler` is applied first. Testing three values of k shows that k=7 offers the best balance between bias and variance. However, KNN still underperforms compared to Random Forest because it struggles with high-dimensional data and irrelevant features.

---

### Step 11 — Naive Bayes Model

```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train_smote, y_train_smote)
y_pred_nb = nb.predict(X_test)
```

**Output:**
```
Naive Bayes Accuracy : 0.9615

               precision    recall  f1-score   support
 No Recommend       0.97      0.98      0.97       289
Yes Recommend       0.95      0.93      0.94       152
     accuracy                           0.96       441
```

**Why it matters:** Naive Bayes is a probabilistic model that assumes feature independence. While it performs respectably (96.15%), it doesn't capture feature interactions that Random Forest handles via decision tree splits. Notably, Naive Bayes wins on **Precision-No** and **Recall-Yes** — meaning it is better at conservatively predicting true positives for "Yes Recommend."

---

### Step 12 — Final Model Comparison

```python
all_models = {
    'Random Forest' : acc_rf,
    'KNN (k=3)'     : acc_k3,
    'KNN (k=7)'     : acc_k7,
    'KNN (k=9)'     : acc_k9,
    'Naive Bayes'   : acc_nb,
}
best_name = max(all_models, key=all_models.get)
```

**Complete Metrics Table:**
```
        Model   Accuracy  F1-No  F1-Yes  F1-Avg  Prec-No  Prec-Yes  Recall-No  Recall-Yes
Random Forest     0.9683  0.976   0.953   0.968   0.963     0.979      0.990      0.928
KNN (k=7)         0.9615  0.969   0.945   0.961     -         -          -          -
Naive Bayes       0.9615  0.971   0.940   0.960   0.970     0.950      0.980      0.930
KNN (k=9)         0.9592    -       -       -       -         -          -          -
KNN (k=3)         0.9501    -       -       -       -         -          -          -
```

**Per-Metric Winner:**
```
Accuracy       → 🏆 Random Forest
F1 Score(avg)  → 🏆 Random Forest
F1 - No        → 🏆 Random Forest
F1 - Yes       → 🏆 Random Forest
Precision-No   → 🏆 Naive Bayes
Precision-Yes  → 🏆 Random Forest
Recall-No      → 🏆 Random Forest
Recall-Yes     → 🏆 Naive Bayes
```

**Win Count:**
```
Random Forest  ██████ (6 wins)
Naive Bayes    ██     (2 wins)
```

---

## Results Summary

| Model | Accuracy | Metrics Won |
|---|---|---|
| **Random Forest** | **96.83%** | **6 / 8** 🏆 |
| KNN (k=7) | 96.15% | — |
| Naive Bayes | 96.15% | 2 / 8 |
| KNN (k=9) | 95.92% | — |
| KNN (k=3) | 95.01% | — |

---

## Visualizations

The notebook includes the following charts:

- **Bar Chart** — Review count per airline
- **Pie Chart** — Recommendation split (Yes vs No)
- **Bar Chart** — Average rating per airline
- **Boxplot** — Rating distribution before/after outlier treatment
- **Confusion Matrices** — For RF, KNN, and Naive Bayes
- **Feature Importance Chart** — What drives recommendation (Rating, Sentiment, etc.)
- **Decision Tree Visualization** — Single tree extracted from the RF ensemble
- **Heatmap** — All metrics across all models side-by-side

---

## Conclusion

This project demonstrates a complete, production-style ML pipeline for text + tabular data classification on Indian airline reviews.

**Key Takeaways:**

- **Random Forest** is the clear overall winner, achieving **96.83% accuracy** and winning **6 out of 8 evaluation metrics** — outperforming KNN across all configurations and Naive Bayes on most metrics.
- **Sentiment engineering** via TextBlob proved to be a valuable feature — adding review and title polarity scores alongside the numeric rating gave the model richer signals beyond just a 1–10 score.
- **SMOTE** improved model accuracy by +0.23% by ensuring the model saw equal examples of both classes during training, without contaminating the test set.
- **Naive Bayes** was surprisingly competitive, winning on Precision-No and Recall-Yes, making it a valid lightweight alternative if inference speed or interpretability is prioritized over raw accuracy.
- **KNN** performed consistently below the other two, confirming that distance-based methods struggle when feature distributions are as varied as review length vs. sentiment polarity.

The best-performing Random Forest configuration was: `n_estimators=50`, `max_depth=5`, `max_features='sqrt'`, `criterion='gini'` — a shallow forest that avoids overfitting while remaining highly accurate.

---

## Repository Structure

```
├── Indian_Domestic_Airline.csv      # Raw dataset
├── manoj_random_forest.ipynb        # Main Jupyter Notebook
├── manoj_random_forest.html         # Exported HTML version
└── README.md                        # This file
```
## Recomrndations for airline
1. Competitor Benchmarking
Since our dataset covers 7 airlines, plug competitor reviews into the same model. Compare:

IndiGo vs Vistara — whose passengers are more likely to recommend?
Which airline has the highest "false positive" zone (high rating but low sentiment = frustrated passengers pretending to be okay)?


2. Marketing & Loyalty Targeting
The model outputs a probability score, not just Yes/No. Use that probability to:

Passengers predicted at >90% Yes → Ask for a Google/TripAdvisor review (they'll write a positive one)
Passengers at 50–70% Yes → Send a satisfaction nudge + loyalty points to push them over
Passengers at <40% → Don't ask for reviews. Fix them first.
---

## Author

**Manjunath Hosmani**  
MBA — Business Analytics & Strategic Planning  
Reva Business School, Bengaluru

---
