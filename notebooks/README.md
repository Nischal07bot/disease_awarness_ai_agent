# AI Agent for Disease Risk Awareness and Prevention
## CS336 Artificial Intelligence and Machine Learning - Assignment 3

A comprehensive machine learning project implementing an AI agent for disease risk assessment using PIMA Indian Diabetes and UCI Heart Disease datasets.

---

## Project Overview

This project develops an end-to-end AI system that:
1. **Preprocesses** medical datasets with imputation and outlier detection
2. **Engineers** new features through domain knowledge and statistical methods
3. **Trains** multiple supervised models (Logistic Regression, Random Forest, Gradient Boosting, SVM) with hyperparameter optimization
4. **Explains** predictions using SHAP, LIME, and feature importance analysis
5. **Implements** an AI Agent for personalized risk classification and recommendations
6. **Analyzes** ethical considerations including fairness, bias detection, and privacy

---

## Datasets

### PIMA Indian Diabetes
- **Samples:** 768
- **Features:** 8 clinical measurements
- **Target:** Diabetes diagnosis (0/1)
- **Source:** Kaggle / UCI ML Repository

### UCI Heart Disease
- **Samples:** 303
- **Features:** 13 clinical/demographic attributes
- **Target:** Heart disease presence (0/1)
- **Source:** UCI ML Repository

---

## Project Structure

```
AIML_LAB/
├── notebooks/                          # Jupyter notebooks (execution order)
│   ├── 01_data_understanding_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_visualization_explainability.ipynb
│   └── 05_agent_recommendation_ethics.ipynb
│
├── data/                               # Data directory
│   ├── pima_X_train.csv               # Training features
│   ├── pima_X_test.csv                # Test features
│   ├── pima_y_train.csv               # Training labels
│   ├── pima_y_test.csv                # Test labels
│   ├── pima_X_train_engineered.csv    # Engineered features (train)
│   ├── pima_X_test_engineered.csv     # Engineered features (test)
│   ├── pima_mi_scores.csv             # Mutual Information scores
│   ├── pima_model_results.csv         # Model performance metrics
│   │
│   ├── heart_X_train.csv              # Similar structure for Heart Disease
│   ├── heart_X_test.csv
│   ├── heart_y_train.csv
│   ├── heart_y_test.csv
│   ├── heart_X_train_engineered.csv
│   ├── heart_X_test_engineered.csv
│   ├── heart_mi_scores.csv
│   └── heart_model_results.csv
│
├── models/                             # Trained models
│   ├── pima_best_model.pkl            # Best PIMA model
│   ├── heart_best_model.pkl           # Best Heart Disease model
│   ├── scaler_pima.pkl                # StandardScaler (PIMA)
│   └── scaler_heart.pkl               # StandardScaler (Heart)
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Installation & Setup

### 1. Clone/Navigate to Project
```bash
cd /home/abhishek/AIML_LAB
```

### 2. Create and Activate Virtual Environment (RECOMMENDED)
```bash
# Create virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Why:**
- Keeps project dependencies isolated from system Python
- Prevents version conflicts with other projects
- Ensures reproducibility across machines

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

**Critical Step:** Install `ipykernel` in the virtual environment
```bash
pip install ipykernel
```

This is required for Jupyter to recognize and use your virtual environment.

### 4. Register Kernel with Jupyter (One-time Setup)
```bash
# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name aiml_env --display-name "Python (AIML)"
```

This command makes your virtual environment appear as a kernel option when launching Jupyter.

### 5. Launch Jupyter Notebook
```bash
# Ensure venv is activated, then:
jupyter notebook
```

### 6. Select Kernel in Jupyter (IMPORTANT)

**If kernel options don't appear, try these steps:**

1. **Close Jupyter completely:**
   ```bash
   # Press Ctrl+C in terminal where Jupyter is running
   # Then close all browser tabs
   ```

2. **Kill any hanging processes:**
   ```bash
   pkill -f jupyter
   pkill -f ipython
   ```

3. **Restart Jupyter:**
   ```bash
   source venv/bin/activate
   jupyter notebook
   ```

4. **Select kernel from dropdown:**
   - Look for **"Select Kernel"** button in top-right corner of notebook
   - Click it and choose **"Python (AIML)"** from the list
   - If button not visible: **Kernel menu** → **Change Kernel** → **"Python (AIML)"**

### 7. Deactivate Virtual Environment (when done)
```bash
deactivate
```

---

## Execution Guide

### Notebook Sequence (MUST execute in order)

#### **Notebook 1: Data Understanding & Preprocessing** (18 KB)
**What it does:**
- Load PIMA and Heart Disease datasets
- Exploratory Data Analysis (EDA): distributions, correlations, class balance
- Preprocessing pipeline: missing value imputation, outlier removal (IQR method)
- Standardization (StandardScaler) and train-test split (80/20, stratified)

**Outputs:**
- Training/test CSV files (features & labels)
- Scaler PKL files for reproducibility
- Visualization: class distribution charts

**Execution Time:** ~2-3 minutes

---

#### **Notebook 2: Feature Engineering** (20 KB)
**What it does:**
- Correlation analysis and heatmaps
- Mutual Information (MI) scoring for feature relevance
- Variance threshold analysis
- **New features created:**
  - PIMA: Age_Group, Glucose_Insulin_Ratio, Pregnancies_Age_Interaction
  - Heart: age_group, chol_bp_ratio, heart_rate_indicator
- PCA exploratory analysis (95% variance retention)

**Outputs:**
- Engineered feature CSV files
- MI scores for feature ranking
- PCA analysis results

**Execution Time:** ~2-3 minutes

---

#### **Notebook 3: Modelling** (24 KB)
**What it does:**
- Train 4 supervised models with GridSearchCV hyperparameter tuning
- Models: Logistic Regression, Random Forest, Gradient Boosting, SVM
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices and model comparison

**Outputs:**
- Model performance comparison CSV
- Best trained models (PKL files)

**Execution Time:** ~5-10 minutes (hyperparameter tuning)

---

#### **Notebook 4: Visualization & Explainability** (22 KB)
**What it does:**
- Feature importance plots from tree models
- ROC and Precision-Recall curves
- SHAP summary plots (Shapley value explanations)
- LIME local explanations (individual predictions)
- Partial Dependence Plots (how features affect predictions)

**Outputs:**
- Multiple explainability visualizations
- Feature ranking by importance
- ROC-AUC and AP scores

**Execution Time:** ~3-5 minutes

---

#### **Notebook 5: AI Agent, Recommendations & Ethics** (18 KB)
**What it does:**
- Implements `DiseaseRiskAgent` class with three risk levels:
  - Low Risk (probability < 0.3)
  - Medium Risk (0.3-0.7)
  - High Risk (> 0.7)
- Generates personalized recommendations per risk level
- **Fairness Analysis:**
  - Demographic parity (difference in positive prediction rates)
  - Equalized odds (difference in true positive rates)
  - Bias detection across age groups
- Privacy & ethical guidelines (HIPAA, consent, transparency)
- Deployment readiness checklist

**Outputs:**
- Risk classifications and recommendations
- Fairness metrics
- Ethical deployment guidelines

**Execution Time:** ~1-2 minutes

---

## Key Results Summary

### Model Performance (Test Set Accuracy)

| Dataset | Model | Accuracy | F1-Score | ROC-AUC |
|---------|-------|----------|----------|---------|
| PIMA | Random Forest | ~0.78 | 0.72 | 0.85 |
| Heart Disease | Gradient Boosting | ~0.85 | 0.82 | 0.91 |

### Feature Engineering Impact
- **PIMA:** 8 → 11 features (added: Age_Group, Glucose_Insulin_Ratio, interaction term)
- **Heart:** 13 → 16 features (added: age_group, chol_bp_ratio, heart_rate_indicator)
- **PCA Result:** Both datasets compress to 5-6 components for 95% variance retention

### Fairness Findings
- **Demographic Parity Difference (PIMA):** ~0.08 (acceptable, < 0.1)
- **Equalized Odds Difference (PIMA):** ~0.12 (monitor for fairness)
- Models show reasonable fairness across age groups

---

## How to Use the AI Agent

### Example: Predict Risk for a Patient
```python
# Load agent from Notebook 5
agent = DiseaseRiskAgent(pima_model, 'pima')

# Get prediction and recommendations for a patient
patient_data = pima_X_test.iloc[0].values
result = agent.predict_and_recommend(patient_data)

print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']}")
for rec in result['recommendations']:
    print(f"  - {rec}")
```

### Risk Classification Logic
1. Model outputs probability of disease (0-1)
2. Agent maps probability to risk level:
   - **Low Risk** → Maintain current lifestyle
   - **Medium Risk** → Increase health monitoring & lifestyle changes
   - **High Risk** → Urgent consultation with healthcare provider
3. Personalized recommendations generated based on risk level

---

## Ethical Considerations

### Key Principles Implemented

1. **Transparency**: Explainability via SHAP, LIME, feature importance
2. **Fairness**: Demographic parity and equalized odds analysis
3. **Privacy**: User data encryption, HIPAA compliance recommended
4. **Accountability**: AI agent is decision support, not medical diagnosis
5. **User Rights**: Data access, rectification, deletion, explanation

### Limitations Acknowledged
- Models trained on specific populations (may not generalize universally)
- Requires regular retraining as new data becomes available
- Should never replace professional medical consultation
- Models may exhibit performance variations across demographic groups

---

## Deployment Recommendations

✅ **Pre-Deployment Checklist:**
1. Validate models on independent test dataset
2. Set up monitoring for model drift and fairness metrics
3. Implement user consent and privacy mechanisms
4. Generate explainability reports for each prediction
5. Establish clear liability disclaimers
6. Plan quarterly retraining cycle

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- **Data Processing:** pandas, numpy, scipy
- **ML Models:** scikit-learn, xgboost
- **Visualization:** matplotlib, seaborn
- **Explainability:** shap, lime
- **Jupyter:** jupyter

---

## Troubleshooting

### Kernel Issues

**Problem:** Kernel options don't appear in Jupyter

**Solution:**
```bash
# 1. Complete shutdown
pkill -f jupyter
pkill -f ipython

# 2. Restart from scratch
source venv/bin/activate
jupyter notebook
```

Then in notebook, click **Select Kernel** → **"Python (AIML)"**

---

**Problem:** "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Kernel not using venv. Change kernel to **"Python (AIML)"**

---

**Problem:** Jupyter not found

**Solution:** 
```bash
source venv/bin/activate
pip install jupyter
jupyter notebook
```

---

## Author & Course
**Course:** CS336 Artificial Intelligence and Machine Learning  
**Assignment:** 3 - AI Agent for Disease Risk Awareness and Prevention  
**Date:** March 2026

---

## References

- PIMA Indian Diabetes Dataset: [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- UCI Heart Disease Dataset: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- SHAP Documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
- LIME Documentation: [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)

---

**End of README**
