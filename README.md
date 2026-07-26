# Loan-Approval-Prediction

This project focuses on building a predictive model to determine the likelihood of loan approval based on various applicant attributes. The primary goal is to demonstrate a typical machine learning workflow, from data loading and preprocessing to model training and evaluation.

Dataset
The dataset used for this project is Loan.csv, which contains information about loan applicants, including their demographics, income, loan amount, credit history, and loan status (approved or not approved).

Setup
The following Python libraries are required to run this notebook:

pandas
numpy
matplotlib
seaborn
scikit-learn
Project Steps
1. Data Loading and Inspection
The Loan.csv dataset is loaded into a pandas DataFrame. Initial inspection includes viewing the first few rows (df.head()), checking data types and non-null counts (df.info()), getting descriptive statistics (df.describe()), and listing column names (df.columns).

2. Missing Value Handling
Missing values are handled using imputation strategies:

Categorical Columns (Gender, Married, Dependents, Self_Employed): Missing values are filled with the mode (most frequent value) of each respective column.
Numerical Columns (LoanAmount, Loan_Amount_Term): Missing values are filled with the median of each respective column.
Credit History: Missing values are filled with the mode of the Credit_History column.
3. Feature Engineering
New features are created to enhance the model's predictive power:

total_income: Sum of ApplicantIncome and CoapplicantIncome.
loan_income_ratio: Ratio of LoanAmount to total_income.
Dependents: The '3+' category is replaced with '3', and the column is converted to integer type.
4. Encoding
Categorical variables are transformed into a numerical format suitable for machine learning models:

Target Variable (Loan_Status): Encoded as 1 for 'Y' (Approved) and 0 for 'N' (Not Approved).
Categorical Features (Gender, Married, Education, Self_Employed, Property_Area): One-Hot Encoded using pd.get_dummies(), with drop_first=True to avoid multicollinearity.
5. Exploratory Data Analysis (EDA)
Visualizations are used to understand data distributions and relationships:

Credit History vs Loan Approval: Count plot showing the impact of credit history on loan approval.
Applicant Income Distribution: Histogram with KDE showing income distribution for approved vs. not approved loans.
Loan Amount Distribution: Histogram with KDE showing loan amount distribution for approved vs. not approved loans.
Approval Rate by Education: Count plot comparing loan approval for 'Graduate' vs. 'Not Graduate' individuals.
Property Area vs Loan Status: Count plots for 'Semiurban' and 'Urban' property areas against loan status.
Correlation Heatmap: A heatmap to visualize correlations between all features.
6. Feature and Target Splitting
The dataset is split into features (X) and the target variable (y). The Loan_Status column is dropped from X.

7. Train-Test Split
The data is further split into training (80%) and testing (20%) sets using train_test_split, ensuring stratification on the target variable (Loan_Status) to maintain the class distribution in both sets.

8. Model Training (Logistic Regression)
A Logistic Regression model is initialized with max_iter=1000 and trained on the X_train and y_train datasets.

9. Prediction
The trained model makes predictions on the X_test dataset, generating y_pred.

10. Model Evaluation
The model's performance is evaluated using several metrics:

Accuracy Score: 85.37%
Precision Score: 83.84%
Recall Score: 97.65%
F1 Score: 90.22%
Classification Report: Provides a detailed breakdown of precision, recall, and f1-score for each class.
Confusion Matrix: Visualized to show true positives, true negatives, false positives, and false negatives.
11. Coefficient Analysis
The coefficients of the Logistic Regression model are extracted and displayed to understand the importance and direction of influence of each feature on the loan approval status. Features are sorted by their coefficient values to identify the most impactful predictors.
