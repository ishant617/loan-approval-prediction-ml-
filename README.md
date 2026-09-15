# loan-approval-prediction-ml-
End-to-end ML pipeline for predicting loan approval using Logistic Regression &amp; KNN — includes EDA, feature engineering, encoding, scaling, and model evaluation.


A complete end-to-end Machine Learning project that predicts whether a loan application
will be approved based on applicant financial and demographic data.

## 📌 Project Highlights
- Exploratory Data Analysis (EDA) with visualizations
- Feature Engineering: log transforms, polynomial features, encoding
- Label Encoding & One-Hot Encoding for categorical variables
- Correlation Heatmap to identify key predictors
- Train-Test Split (80/20) with StandardScaler normalization
- Two ML models: Logistic Regression & K-Nearest Neighbors (KNN)
- Full evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

- ## 🛠️ Tech Stack
Python | Pandas | NumPy | Scikit-learn | Seaborn | Matplotlib

## 📂 Dataset
Loan applicant data including: Gender, Marital Status, Education Level,
Employment Status, Applicant Income, Coapplicant Income, Credit Score,
DTI Ratio, Property Area, and Loan Status.

## 🚀 Models Used
| Model                | Key Parameters                        |
|---------------------|---------------------------------------|
| Logistic Regression | class_weight='balanced', max_iter=1000|
| KNN Classifier      | n_neighbors=5, weights='distance'     |


approval
- Class imbalance was handled using balanced class weights

## 📊 Key Findings
- Credit Score is the strongest predictor of loan approval
- DTI Ratio has a significant negative correlation with 
