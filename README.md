# Loan_Default_Prediction

A full machine learning pipeline for predicting loan default risk, built on a dataset of ~58,600 loan application records. 

## Project Structure
- EDA.ipynb: Data cleaning, univariate and multivariate analysis, class imbalance assessment
- Modeling.ipynb: feature preprocessing pipelines, model training, hyperparameter tuning, and evaluation
- Loan_Default_Prediction_Report.pdf: full written report covering methodology and results

## Summary
The dataset contains 11 borrower and loan features alongside a binary target (loan_status). After cleaning and EDA, seven models were trained and evaluated with a focus on recall and F1 score, given a 6:1 class imbalance within the target variable.

| Model | Precision | Recall | F1 | AUC |
|:---|---:|---:|---:|---:|
| Logistic Regression | 77.5% | 47.0% | 58.5% | 0.903 |
| Logistic Regression + Class Weights | 43.6% | 83.6% | 57.3% | 0.903 |
| Decision Tree | 69.9% | 70.7% | 70.3% | 0.828 |
| Decision Tree + Class Weights | 73.3% | 68.0% | 70.6% | 0.819 |
| Random Forest (Tuned) | 93.8% | 69.3% | 79.7% | 0.941 |
| XGBoost (Tuned) | 74.5% | 82.6% | 78.3% | 0.959 |
| Neural Network (Tuned) | 90.6% | 69.0% | 78.3% | 0.928 |


XGBoost is the recommended model for lending applications, achieving the highest recall (82.6%) and AUC (0.959), minimizing missed defaults at the cost of a modest false positive rate. 

## Key Findings
- loan_grade is the strongest predictor of default, with default rates ranging from 4.9% (Grade A) to 81.2% (Grade G)
- loan_percent_income, loan_int_rate, and cb_person_default_on_file (prior default history) are also strong predictors
- class imbalance required stratified sampling and imbalance-aware training strategies across all models

## Tools & Libraries
Python, scikit-learn, XGBoost, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn
