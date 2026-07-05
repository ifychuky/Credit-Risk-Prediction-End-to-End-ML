# Credit-Risk-Prediction-End-to-End-ML
End-to-end machine learning pipeline for predicting loan default risk using clustering, classification models, and imbalance handling techniques.

## 🔄 Project Workflow

This project followed a complete end-to-end machine learning lifecycle, transforming raw borrower data into actionable credit risk insights.

### 1. Business Understanding & Problem Definition

Defined the business challenge:

Financial institutions need to identify borrowers with a higher likelihood of loan default in order to improve lending decisions, minimize financial losses, and manage risk effectively.

The project objective was to:
- Discover hidden borrower segments using unsupervised learning
- Build predictive models capable of identifying potential loan defaulters
- Improve minority class detection through imbalance handling techniques


---

### 2. Data Acquisition & Storage

- Retrieved the Loan Default Prediction dataset
- Loaded the dataset into the Python environment
- Stored and queried structured data using SQLite
- Performed SQL-based exploration to identify early business patterns


Workflow:

Raw Dataset
        ↓
SQLite Database
        ↓
SQL Exploration
        ↓
Analytical Insights


---

### 3. Exploratory Data Analysis (EDA)

Performed detailed data exploration to understand:

- Dataset structure and feature types
- Missing values
- Target variable distribution
- Borrower financial characteristics
- Relationships between features and loan default behavior

Key objective:
Understand the data before applying machine learning algorithms.


---

### 4. Data Cleaning & Preprocessing

Prepared the dataset for modeling through:

- Missing value treatment
- Duplicate checking
- Data type correction
- Outlier assessment
- Categorical variable preparation

Each preprocessing decision was guided by preserving meaningful borrower information.


---

### 5. Feature Engineering

Created additional predictive features from existing borrower information to better capture financial behavior.

Feature engineering focused on:

- Borrower repayment capacity
- Financial risk indicators
- Relationships between existing variables

The goal was to provide machine learning models with stronger signals beyond the original raw features.


---

### 6. Encoding & Feature Scaling

Prepared features for machine learning:

- Applied appropriate categorical encoding techniques
- Transformed categorical variables into numerical representations
- Scaled numerical features to improve algorithm performance


---

### 7. Customer Risk Segmentation (Unsupervised Learning)

Applied K-Means clustering to discover natural borrower groups.

Steps:

Feature Data
       ↓
K-Means Clustering
       ↓
Cluster Evaluation
       ↓
Borrower Risk Personas


Techniques used:
- Elbow Method
- Silhouette Score
- PCA Visualization

Outcome:

Generated borrower segments to better understand different financial behavior patterns.


---

### 8. Predictive Modeling (Supervised Learning)

Built classification models to predict loan default risk.

Models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Modeling Pipeline:

Training Data
       ↓
Machine Learning Models
       ↓
Performance Evaluation
       ↓
Best Model Selection


---

### 9. Class Imbalance Optimization

Since loan default cases are typically less frequent, imbalance handling was prioritized.

Techniques explored:

- SMOTE (Synthetic Minority Oversampling Technique)
- Class Weight Adjustment
- Decision Threshold Optimization

Focus:

Improve the model's ability to detect high-risk borrowers rather than only maximizing accuracy.


---

### 10. Business Interpretation & Recommendations

Translated machine learning results into practical insights:

- Identified borrower groups with higher default tendencies
- Compared model trade-offs
- Recommended deployment strategy
- Suggested future improvements


---

## 🚀 Future Improvements

Future versions of this project can include:

- Advanced boosting models (XGBoost, LightGBM)
- Model explainability using SHAP values
- Deployment through Streamlit or FastAPI
- Real-time loan risk scoring system
- Application of similar risk prediction methods to healthcare datasets
