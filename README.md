<div align="center">

# 🔥 XGBRegressor Calories Burnt Prediction

### *Predicting How Many Calories You Burn — Powered by Gradient Boosting Magic* ✨

<br>

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

<br>

> 🏋️ *"Your workout data holds the secret to your calorie burn. XGBoost knows how to read it."*

<br>

---

### 🏆 Model Highlights at a Glance

| 🗃️ Total Records | 📐 Input Features | 🎯 Target Variable | 🤖 Algorithm | 📉 Mean Absolute Error |
|:---:|:---:|:---:|:---:|:---:|
| **15,000** | **6** | **Calories Burnt** | **XGBRegressor** | **~1.5 Calories** |

---

</div>

<br>

## 📚 Table of Contents

| # | 📌 Section | 🔍 What's Inside |
|:---:|:---|:---|
| 1 | [📁 Repository Structure](#-repository-structure) | Folder tree with all file names |
| 2 | [🎯 What Is This Project?](#-what-is-this-project) | Problem statement & prediction goal |
| 3 | [✨ Features](#-features) | Key capabilities of this project |
| 4 | [📊 Dataset Deep Dive](#-dataset-deep-dive--two-files-one-powerful-dataset) | Two CSVs explained — columns, types & roles |
| 5 | [🔢 Combined Dataset Stats](#-combined-dataset-stats) | Record counts, ranges & distributions |
| 6 | [🔬 End-to-End ML Pipeline](#-end-to-end-ml-pipeline) | Visual flow from raw data to prediction |
| 7 | [🧪 Step-by-Step Notebook Breakdown](#-step-by-step-notebook-breakdown) | All 8 pipeline steps with code |
| 8 | [⚡ Why XGBoost?](#-why-xgboost-the-secret-weapon-explained) | Algorithm strengths & justification |
| 9 | [🔥 What Burns the Most Calories?](#-what-actually-burns-the-most-calories) | Feature importance insights |
| 10 | [🆚 XGBoost vs Linear Regression](#-why-not-linear-regression) | Head-to-head algorithm comparison |
| 11 | [🚀 Getting Started](#-getting-started) | Clone, install & run instructions |
| 12 | [📋 Requirements](#-requirements) | All dependencies listed |
| 13 | [📂 File Reference](#-file-reference) | What each file contains |
| 14 | [🌍 Real-World Applications](#-real-world-applications) | Industry use cases |
| 15 | [📌 Key Takeaways](#-key-takeaways) | Project summary & highlights |
| 16 | [📜 License](#-license) | MIT License |

---

## 📁 Repository Structure

```
🗂️ XGBRegressor_Calories_Burnt_Prediction/
│
├── 📄 README.md                               ← You are here!!
│
└── 📂 Gradient Boosting Linear Regression Project - Calories Burnt Prediction/
    │
    ├── 📓 Gradient_Boosting_Linear_Regression_Project_-_Calories_Burnt_Prediction.ipynb
    ├── 📊 exercise.csv        ← Exercise & physiological data (15,000 records)
    └── 📊 calories.csv        ← Corresponding calories burned (15,000 records)
```

---

## 🎯 What Is This Project?

This project uses the **XGBoost Regressor** — the gold standard of gradient boosting algorithms — to **predict the exact number of calories a person burns** during a workout session, based on their physiological measurements and exercise data.

> 💡 Unlike classification (yes/no answers), this is a **Regression** problem — the model predicts a **continuous numerical value** (calories burned).

**The Core Prediction Question:**

```
🤔  "Given a person's age, gender, weight, heart rate, and workout duration...
      exactly how many calories did they burn?" 🔥
```

---

## ✨ Features

> Everything this project brings to the table — from data to deployment-ready insights:

```
🔥  PREDICTION POWER
    └── Predicts exact calories burned per workout session using XGBRegressor

📂  DUAL DATASET MERGING
    └── Seamlessly combines exercise.csv + calories.csv into one analysis-ready DataFrame

🔍  RICH EXPLORATORY ANALYSIS
    ├── Univariate: Histograms, Boxplots, Count Plots, Pie Charts
    └── Bivariate: Scatter Plots + Pearson Correlation Matrix

🛠️  SMART FEATURE ENGINEERING
    ├── Gender label encoding (text → binary: male=0, female=1)
    └── User_ID dropped — non-predictive IDs excluded from training

🌳  SCALE-INVARIANT MODELING
    └── XGBoost is tree-based — no StandardScaler needed, saving a preprocessing step

⚡  GRADIENT BOOSTING ENSEMBLE
    └── 100 sequential decision trees, each correcting the errors of the last

📉  RIGOROUS EVALUATION
    └── Mean Absolute Error (MAE) used — intuitive, same unit as the target (calories)

🏆  EXCEPTIONAL ACCURACY
    └── MAE ≈ 1.5 calories on a 1–314 range — less than 2% average error!

♻️  FULLY REPRODUCIBLE
    └── Fixed random_state=2 ensures identical results every single run

📓  WELL-DOCUMENTED NOTEBOOK
    └── 55 cells with inline comments explaining every step of the pipeline
```

---

## 📊 Dataset Deep Dive — Two Files, One Powerful Dataset

This project uniquely uses **two separate CSV files** that are merged into a single DataFrame:

### 📋 File 1 — `exercise.csv` (Input Features)

<div align="center">

| # | 🔢 Feature | 📋 Description | 🔬 Type | 🎯 Role |
|:---:|:---|:---|:---:|:---:|
| 1 | `User_ID` | Unique identifier per person | Integer | ❌ Dropped |
| 2 | `Gender` | Male / Female | Categorical | ⚙️ Encoded 0/1 |
| 3 | `Age` | Age of the individual (years) | Continuous | ✅ Feature |
| 4 | `Height` | Height in centimeters | Continuous | ✅ Feature |
| 5 | `Weight` | Weight in kilograms | Continuous | ✅ Feature |
| 6 | `Duration` | Workout duration in minutes | Continuous | ✅ Feature |
| 7 | `Heart_Rate` | Heart rate during exercise (bpm) | Continuous | ✅ Feature |
| 8 | `Body_Temp` | Body temperature during exercise (°C) | Continuous | ✅ Feature |

</div>

### 📋 File 2 — `calories.csv` (Target Variable)

| 🔢 Column | 📋 Description | 🎯 Role |
|:---|:---|:---:|
| `User_ID` | Matching identifier for merge | 🔗 Join Key |
| `Calories` | Actual calories burned (float) | 🎯 **TARGET** |

---

### 🔢 Combined Dataset Stats

```
📦 Merged Dataset — calories_data
├── 🗃️  Total Records          →  15,000 rows
├── 📐  Total Features          →  9 columns (8 input + 1 target)
├── 👩  Female Participants     →  7,553  (50.4%)
├── 👨  Male Participants       →  7,447  (49.6%)
├── 📅  Age Range               →  20 – 79 years
├── 💓  Heart Rate Range        →  67 – 128 bpm
├── 🌡️  Body Temp Range         →  37.1°C – 41.5°C
├── 🔥  Calories Min            →  1 calorie
├── 🔥  Calories Max            →  314 calories
└── 🔥  Calories Mean           →  ~89.5 calories
```

---

## 🔬 End-to-End ML Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  📥  LOAD DATA          →   Read exercise.csv + calories.csv       │
│         ↓                                                          │
│  🔗  MERGE DATASETS     →   pd.concat() on User_ID axis           │
│         ↓                                                          │
│  🔍  EDA                →   Shape, Info, Nulls, Duplicates         │
│         ↓                                                          │
│  📊  UNIVARIATE         →   Histograms, Boxplots, Countplot        │
│         ↓                                                          │
│  📈  BIVARIATE          →   Scatterplots + Correlation Matrix      │
│         ↓                                                          │
│  🛠️  FEATURE ENG.       →   Encode Gender (male→0, female→1)      │
│         ↓                                                          │
│  ✂️  FEATURE SELECTION  →   Drop User_ID, isolate Calories         │
│         ↓                                                          │
│  🔀  TRAIN-TEST SPLIT   →   80% Train | 20% Test                  │
│         ↓                                                          │
│  🚀  XGBRegressor       →   Gradient Boosted Trees                 │
│         ↓                                                          │
│  📉  EVALUATE           →   Mean Absolute Error (MAE)              │
│         ↓                                                          │
│  ✅  RESULT             →   MAE ≈ 1.5 Calories 🔥                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Step-by-Step Notebook Breakdown

<details>
<summary><b>📦 Step 1 — Importing Libraries</b> 🖱️ click to expand</summary>
<br>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```

> ⚡ `xgboost` is imported as a standalone package — it's not bundled with scikit-learn. Install it separately with `pip install xgboost`.

</details>

---

<details>
<summary><b>🔗 Step 2 — Loading & Merging the Two Datasets</b> 🖱️ click to expand</summary>
<br>

```python
# Load both files
calories      = pd.read_csv("calories.csv")        # 15,000 x 2
exercise_data = pd.read_csv("exercise.csv")        # 15,000 x 8

# Merge horizontally on matching row order
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
# Result: 15,000 x 9
```

> 🔗 Both datasets share the same `User_ID` order, so a simple `axis=1` concat perfectly aligns records — no `.merge()` needed!

</details>

---

<details>
<summary><b>🔍 Step 3 — Exploratory Data Analysis (EDA)</b> 🖱️ click to expand</summary>
<br>

| 🔎 Check | 📋 Finding |
|:---|:---|
| Combined Shape | `(15000, 9)` |
| Missing Values | ✅ Zero nulls across all columns |
| Duplicate Rows | ✅ Zero duplicates confirmed |
| Gender Split | 7,553 Female 👩 \| 7,447 Male 👨 |
| Calorie Range | 1 🔥 (min) to 314 🔥 (max) |
| Average Calories | ~89.5 calories per session |

```python
calories_data.info()              # Column types, null counts
calories_data.isnull().sum()      # → All zeros ✅
calories_data.duplicated().sum()  # → Zero ✅
```

</details>

---

<details>
<summary><b>📊 Step 4 — Univariate & Bivariate Analysis</b> 🖱️ click to expand</summary>
<br>

**📌 Univariate Plots:**

| 📊 Plot | 🔍 Feature | 💡 Purpose |
|:---|:---|:---|
| 📶 Histogram | `User_ID` | Distribution of user IDs |
| 📦 Box Plot | `User_ID` | Outlier check |
| 📊 Count Plot | `Gender` | Male vs Female ratio |
| 🥧 Pie Chart | `Gender` | Proportional gender split |

**📌 Bivariate Plots:**

| 📈 Plot | 🔍 Features | 💡 Insight |
|:---|:---|:---|
| 🔵 Scatter Plot | `Age` vs `Height` | Physical relationship check |
| 🔢 Correlation | `Age` & `Height` | Numerical correlation value |

```python
# Gender visual breakdown
sns.countplot(x='Gender', data=calories_data)

# Physiological relationship
sns.scatterplot(x='Age', y='Height', data=calories_data)

# Numerical correlation
calories_data[['Age', 'Height']].corr()
```

</details>

---

<details>
<summary><b>🛠️ Step 5 — Feature Engineering</b> 🖱️ click to expand</summary>
<br>

**🔹 Encoding Categorical Variable (Gender)**

XGBoost requires numerical input — text labels were mapped to binary integers:

```python
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# male   → 0  👨
# female → 1  👩
```

**🔹 Feature & Target Separation**

```python
# Drop non-predictive ID column + target column
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)

# Target: what we want to predict
Y = calories_data['Calories']

# X shape → (15000, 6)  |  Y shape → (15000,)
```

> ✅ **6 clean features** used for prediction: `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`

</details>

---

<details>
<summary><b>🔀 Step 6 — Train-Test Split</b> 🖱️ click to expand</summary>
<br>

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)
```

```
📊 Data Split Breakdown
├── 🏋️  Training Set   →  12,000 samples  (80%)
└── 🧪  Testing Set    →   3,000 samples  (20%)
```

> 🎲 `random_state=2` ensures the same split every run — fully reproducible results!

</details>

---

<details>
<summary><b>🚀 Step 7 — XGBoost Regressor Training</b> 🖱️ click to expand</summary>
<br>

```python
from xgboost import XGBRegressor

# Initialize the model
model = XGBRegressor()

# Train on 12,000 samples
model.fit(X_train, Y_train)
```

> ⚡ XGBoost internally builds an **ensemble of decision trees**, each one correcting the errors of the previous — that's the "boosting" in Gradient Boosting!

| ⚙️ Parameter | 🔧 Default Value | 💡 Meaning |
|:---|:---|:---|
| `n_estimators` | 100 | Number of boosting rounds |
| `learning_rate` | 0.3 | Step size per round |
| `max_depth` | 6 | Max tree depth per round |
| `objective` | `reg:squarederror` | Regression loss function |

</details>

---

<details>
<summary><b>📉 Step 8 — Model Evaluation (MAE)</b> 🖱️ click to expand</summary>
<br>

```python
# Predict on unseen test data
test_data_prediction = model.predict(X_test)

# Calculate Mean Absolute Error
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error =", mae)
```

### 🏆 Results

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   📉  Mean Absolute Error (MAE)  →  ~1.5 kcal  │
│                                                 │
│   💡  Interpretation:                           │
│       On average, the model's calorie           │
│       prediction is off by only 1.5 calories!  │
│       That's less than 2% error on an           │
│       89-calorie mean. Outstanding! 🏆          │
│                                                 │
└─────────────────────────────────────────────────┘
```

> 🎯 An MAE of **~1.5 calories** on a target ranging 1–314 is **exceptional accuracy** for a default out-of-the-box model with zero hyperparameter tuning!

</details>

---

## ⚡ Why XGBoost? The Secret Weapon Explained

```
┌───────────────────────────────────┬──────────────────────────────────────────────┐
│      ✨ XGBoost Superpower         │       🏋️  Why It Dominates Here              │
├───────────────────────────────────┼──────────────────────────────────────────────┤
│  Gradient Boosting ensemble       │  Learns complex nonlinear calorie patterns    │
│  Built-in regularization (L1/L2)  │  Prevents overfitting on 15k records         │
│  Handles mixed data types natively│  Numerical + encoded categorical features     │
│  Extremely fast training          │  15k rows trained in seconds                 │
│  Works great out-of-the-box       │  Default params already give MAE ≈ 1.5       │
│  Scale-invariant (tree-based)     │  No feature scaling needed! 🌳               │
└───────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 🔥 What Actually Burns the Most Calories?

Based on the feature relationships in the dataset:

```
🏆 Top Calorie-Burning Factors (Estimated Importance)
│
├── ⏱️  Duration       →  Longer workout = more calories burned
├── 💓  Heart Rate     →  Higher intensity = more fuel consumed
├── 🌡️  Body Temp      →  Elevated temp = high metabolic activity
├── ⚖️  Weight         →  Heavier body = more energy to move
├── 📅  Age            →  Metabolic rate varies with age
└── ♀️♂️ Gender        →  Physiological differences in metabolism
```

---

## 🆚 Why Not Linear Regression?

| 📊 Metric | 📉 Linear Regression | ⚡ XGBoost Regressor |
|:---|:---:|:---:|
| Handles nonlinearity | ❌ No | ✅ Yes |
| Sensitive to outliers | ⚠️ High | ✅ Low |
| Feature interactions | ❌ Manual only | ✅ Automatic |
| Default accuracy | 🔴 Lower | 🟢 Much Higher |
| Needs feature scaling | ⚠️ Yes | ✅ No |
| Interpretability | 🟢 Simple | ⚠️ Moderate |

> 🏆 XGBoost wins on every accuracy metric while remaining fast and practical for real-world use!

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/XGBRegressor_Calories_Burnt_Prediction.git
cd "XGBRegressor_Calories_Burnt_Prediction/Gradient Boosting Linear Regression Project - Calories Burnt Prediction"
```

### 2️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### 3️⃣ Launch the Notebook
```bash
jupyter notebook "Gradient_Boosting_Linear_Regression_Project_-_Calories_Burnt_Prediction.ipynb"
```

> ✅ Both `exercise.csv` and `calories.csv` are referenced with **relative paths** — keep all three files in the **same folder** and it works instantly, no path changes needed!

---

## 📋 Requirements

```
Python         >= 3.7
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost          ← ⚡ Key dependency — install separately!
jupyter
```

---

## 📂 File Reference

| 📄 File | 📋 What's Inside |
|:---|:---|
| `Gradient_Boosting_Linear_Regression_Project_-_Calories_Burnt_Prediction.ipynb` | Full pipeline: Load → Merge → EDA → Feature Engineering → XGBoost Training → MAE Evaluation (55 cells) |
| `exercise.csv` | 15,000 rows of physiological & exercise data — Gender, Age, Height, Weight, Duration, Heart Rate, Body Temp |
| `calories.csv` | 15,000 rows of actual calories burned — the prediction target |

---

## 🌍 Real-World Applications

<div align="center">

> ⌚ **Fitness Wearables** (Apple Watch, Fitbit) — calorie estimation from heart rate + movement
>
> 🏥 **Clinical Nutrition** — prescribed calorie burn for weight management programs
>
> 🏟️ **Sports Science** — athlete performance & recovery optimization
>
> 📱 **Health Apps** — personalized workout recommendations based on predicted burn
>
> 🤖 **AI Personal Trainers** — dynamically adjust workouts to hit calorie targets

</div>

---

## 📌 Key Takeaways

```
✅  Two CSVs seamlessly merged into one powerful dataset
✅  15,000 records — large enough for robust generalization
✅  Clean data — zero nulls, zero duplicates right out of the box
✅  XGBoost achieves a remarkable MAE of ~1.5 calories with default params
✅  Only 6 features needed — lean, interpretable, and highly efficient
✅  Regression pipeline — predicts continuous values, not binary classes
✅  No feature scaling required — XGBoost is tree-based & scale-invariant 🌳
✅  Fully reproducible with fixed random_state=2
```

---

## 📜 License

Distributed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

### 💬 *"An algorithm that can predict your calorie burn better than your fitness tracker — that's the power of XGBoost."*

<br>

⭐ **Found this useful? Star the repo and spread the knowledge!** ⭐

`🔥 Built with passion for ML + Health & Fitness Analytics`

</div>
