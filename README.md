# Credit Card Fraud Detection

Detect fraudulent credit card transactions using classical machine learning models on the well-known **Kaggle Credit Card Fraud Detection** dataset.

This repository is centered around a single Jupyter notebook:

- `credit_card_fraud_detection.ipynb`

The notebook walks through:
- Loading and exploring the dataset (`creditcard.csv`)
- Handling **extreme class imbalance**
- Preprocessing (scaling `Time` and `Amount`)
- Training multiple models
- Evaluating using **ROC-AUC**, **Precision–Recall**, confusion matrix, and classification report
- Threshold tuning and model saving

---

## Dataset

- **Dataset:** Credit Card Fraud Detection (European cardholders)
- **Source:** Kaggle
- **File expected by notebook:** `creditcard.csv`

The dataset is highly imbalanced (fraud is ~0.17%). Because of this, **accuracy is not a reliable metric**.

### How to get the data

1. Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the same folder as the notebook:

```
Credit_Card_Fraud_Detection/
├─ credit_card_fraud_detection.ipynb
└─ creditcard.csv
```

> Tip: Do not commit `creditcard.csv` to GitHub (it is large). Use `.gitignore` or download instructions instead.

---

## Project Structure

```
Credit_Card_Fraud_Detection/
├─ credit_card_fraud_detection.ipynb
└─ README.md
```

---

## Approach

### Features
- `V1` … `V28`: PCA-transformed features provided by the dataset
- `Time`, `Amount`: scaled using `StandardScaler`
- Target: `Class` (0 = normal, 1 = fraud)

### Class imbalance handling
The notebook demonstrates **Random Undersampling** for faster training:

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
```

It also uses model-based imbalance handling:
- `class_weight='balanced'` (Logistic Regression / Random Forest)
- `scale_pos_weight` (XGBoost)

### Models trained
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

---

## Results (from the notebook output)

Reported ROC-AUC scores on the test set (approx.):

- **Logistic Regression:** ~0.976
- **Random Forest:** ~0.978
- **XGBoost:** ~0.980

> Note: With heavily imbalanced data, a high ROC-AUC does not automatically mean high precision. The notebook also plots **Precision–Recall curves** and demonstrates **threshold tuning**.

---

## How to Run

### 1) Create a virtual environment (recommended)

Windows / PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

### 2) Install dependencies

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib jupyter
```

### 3) Start Jupyter and run the notebook

```powershell
jupyter notebook
```

Open and run:
- `credit_card_fraud_detection.ipynb`

---

## Notes / Best Practices

- **Precision–Recall** metrics are often more informative than ROC-AUC for fraud detection.
- Consider a **time-based split** (chronological) to reduce leakage when working with transaction data.
- Undersampling speeds up training but discards many normal examples; alternatives include:
  - SMOTE / SMOTEENN
  - training on full data with class weights
  - anomaly detection methods

---

## Model Export

The notebook shows how to save the trained model:

```python
import joblib
joblib.dump(xgb, 'best_fraud_model.pkl')
```

---

## License

If you plan to publish this on GitHub, consider adding a license (e.g., MIT).
