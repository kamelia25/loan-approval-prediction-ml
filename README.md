# Loan Approval Prediction — Machine Learning Project

Binary classification model to predict mortgage loan approval 
outcomes using the Florida 2013 dataset.

**Course:** Introduction to Machine Learning | Tel Aviv University  
**Built by:** Kamelia Atwan

---

## What this project does

Given an applicant's personal, financial, and regional data, 
the model predicts whether a mortgage loan will be 
approved or not.

---

## Dataset

- Florida 2013 mortgage dataset
- ~58,000 samples, 32 features
- Target variable: loan approval (binary: approved / not approved)

---

## Approach

**Data Exploration**
- Analyzed distributions, outliers, and correlations
- Key finding: Feature C strongly correlates with 
  population (0.88) and owner-occupied units (0.95)

**Preprocessing**
- Dropped columns with >99% missing values
- IQR capping for outliers
- Log transformation + Min-Max normalization
- One-Hot Encoding for categorical features
- Grouped 67 counties into 3 Florida regions
- Feature count reduced from 81 → selected features via PCA

**Models trained**
| Model | Validation Accuracy |
|---|---|
| Logistic Regression | baseline |
| K-Nearest Neighbors (k=5) | — |
| Random Forest (100 estimators) | ~78.8% |
| MLP Neural Network (100 neurons) | ~81.4% |
| XGBoost | AUC = 0.8749 |

---

## Results

- Random Forest: 78.8% accuracy, AUC = 0.88
- MLP: 81.4% validation accuracy — no overfitting
- XGBoost AUC: Train 0.8817 / Test 0.8749
- Top features: Feature A, loan_amount, applicant_income

---

## Technologies

Python · scikit-learn · XGBoost · pandas · 
NumPy · Matplotlib · Seaborn · Jupyter Notebook

---

## How to run

pip install pandas numpy scikit-learn xgboost 
matplotlib seaborn
jupyter notebook notebook_8.ipynb
