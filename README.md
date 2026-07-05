# Credit Risk Prediction & Customer Segmentation Using Machine Learning (Capstone Project)

## Project Overview

This project develops an end-to-end machine learning solution for predicting loan default risk and understanding borrower behavior.

The objective was not only to maximize prediction accuracy, but to build a practical risk assessment framework that balances model performance with business decision-making.

The workflow combines:
- SQL-based data exploration
- Exploratory data analysis
- Feature engineering
- Unsupervised customer segmentation
- Classification modeling
- Imbalanced learning optimization
- Business interpretation

## 🔄 Project Workflow

This project followed a complete end-to-end machine learning lifecycle, transforming raw borrower data into actionable credit risk insights.

## Tools & Technologies

- Python
- Pandas
- NumPy
- SQLite
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced Learning Techniques
- Jupyter Notebook

## My Role

As team lead, I was responsible for:

- Designing the machine learning workflow
- Coordinating project execution
- Leading data preprocessing and feature engineering decisions
- Building and evaluating classification models
- Implementing imbalance handling strategies
- Translating model results into business recommendations

## Key Result

The optimized Random Forest model increased default detection recall from 4.5% to 63.8%, significantly improving the model's ability to identify high-risk borrowers.

---

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
  
 <img width="876" height="547" alt="image" src="https://github.com/user-attachments/assets/9dd86a85-4e91-4afa-b9cf-f873290847d2" />

The dataset showed imbalance between default and non-default borrowers, requiring techniques beyond accuracy-based evaluation.

- Borrower financial characteristics

-  Correlation Heatmap

 <img width="1002" height="787" alt="image" src="https://github.com/user-attachments/assets/ad7f4e30-b43b-456d-b712-99d4b6271628" />

Visualizing correlations between numerical variables to identify relationships and potential predictive patterns.

- Relationships between features and loan default behavior

<img width="1489" height="1226" alt="image" src="https://github.com/user-attachments/assets/2f4002f2-a146-4361-baa1-6f773324a9f5" />

KDE Plot was used to visualise the distribution of each numerical feature across loan default categories.


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
  
<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/8a8795b6-4c9a-4247-a934-1d01c15bb0d0" />

The elbow method was used to identify an appropriate number of borrower segments.

- Silhouette Score
  Silhouette analysis was performed to evaluate cluster cohesion and separation across different cluster values.
  
- PCA Visualization
  
<img width="844" height="547" alt="image" src="https://github.com/user-attachments/assets/4a179f2a-a85d-4dff-8300-fa2ada79acd2" />

PCA was used to reduce dimensions and visualize borrower risk groups.


Outcome:

Generated borrower segments to better understand different financial behavior patterns.


Cluster Interpretation/Borrower Risks Personas:

#### Cluster 0 — Financially Strained Borrowers
Low-income borrowers with poor credit scores and extremely high repayment pressure.

#### Cluster 1 — Strong Credit High-Debt Borrowers
High-income borrowers with excellent credit scores but high debt burden.

#### Cluster 2 — Financially Stable Low-Pressure Borrowers
High-income borrowers with poor credit scores but relatively low debt burden and lower repayment pressure.

#### Cluster 3 — Credit-Constrained Borrowers
High-income borrowers with very weak credit scores and high debt obligations.

---

### 8. Predictive Modeling (Supervised Learning)

Built classification models to predict loan default risk.

Models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Model Evaluation:  

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/17565f7d-160c-4982-bd78-d16a3d2860d7" />

Model evaluation showed why accuracy alone can be misleading for imbalanced datasets. Although Logistic Regression and Random Forest achieved approximately 89% accuracy, their ability to identify loan defaults was limited.

F1-score was prioritized because detecting potential defaulters requires balancing precision and recall. The Decision Tree model achieved the strongest F1-score (0.211), making it the preferred baseline model for identifying higher-risk applicants.


Modeling Pipeline:

Training Data
       ↓
Machine Learning Models
       ↓
Performance Evaluation
       ↓
Best Model Selection


Confusion Matrix:

<img width="1674" height="495" alt="image" src="https://github.com/user-attachments/assets/da87e354-be5a-4075-af14-ae4033e5c0c6" />

Since missing potential defaulters carries financial risk, recall and false negatives were considered carefully.
Confusion matrices were used to understand each model beyond accuracy as described below:

-True Negatives: correctly predicted non-defaulters
-False Positives: safe customers wrongly flagged
-False Negatives: defaulters your model missed (very important in lending)
-True Positives: correctly detected defaulters


ROC-AUC Curve Analysis

Accuracy alone was not sufficient for evaluating this loan default prediction model because the dataset contained class imbalance. 

To better understand how well each model separated customers likely to default from those unlikely to default, the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) was evaluated as shown below:

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/0780a947-bf9e-4b1d-bea7-87af863be7ed" />

Key Observations from ROC Plot

- Logistic Regression achieved the strongest ROC-AUC score (0.762), showing the best ability to distinguish between default and non-default customers.
- Random Forest performed competitively with an ROC-AUC score of 0.742, demonstrating strong predictive capability.
- Decision Tree had weaker generalization, suggesting possible overfitting.

Although accuracy scores were similar, ROC-AUC provided deeper insight into real-world model performance, especially for financial risk prediction where identifying high-risk customers is more important than overall accuracy alone.


---

### 9. Class Imbalance Optimization

Since loan default cases are typically less frequent, imbalance handling was prioritized.

Techniques explored:

- SMOTE (Synthetic Minority Oversampling Technique)
- Class Weight Adjustment
- Decision Threshold Optimization

Focus:

Improve the model's ability to detect high-risk borrowers rather than only maximizing accuracy.


Class Imbalance Optimization Results:

During model evaluation, the baseline Random Forest achieved high accuracy but struggled to identify loan defaulters due to class imbalance.
Since detecting high-risk borrowers was the primary business objective, additional optimization strategies were applied:

- Class weight adjustment
- Decision threshold tuning

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/d5d650ba-6089-4e4d-b41e-880c13056a68" />

The class-weighted Random Forest improved the F1-score from **0.084 to 0.349** and increased recall from **0.045 to 0.638**, making it significantly better at identifying potential defaulters.

Although overall accuracy decreased, the optimized model provided a better balance between prediction performance and business risk management.
The RF + class_weight model is the best-performing option based on the business goal of identifying loan defaulters.

---
### 10. Customer Segmentation Using K-Means Clustering

Beyond predicting loan default, I applied unsupervised machine learning to identify borrower segments with different risk profiles.

K-Means clustering was used to group customers based on similar characteristics, and each cluster was analyzed by its observed default behavior.

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/7bf02617-4011-45de-a6ea-213f4bf40c1b" />

Key Insights:

- Cluster 0 represented the highest-risk borrower segment with a default rate of **20.48%**.
- Clusters 1, 2, and 3 showed considerably lower default rates between **9–10%**.
- This segmentation approach can support targeted risk management strategies instead of treating all borrowers equally.

Combining classification models with customer segmentation provides both prediction and explainability — helping financial institutions understand not only **who may default**, but also **which borrower groups carry higher risk**.

---

### 11. Business Interpretation & Recommendations

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

---

## Personal Reflection

This project strengthened my understanding that successful machine learning is not only about building accurate models, but about making decisions that solve real-world problems.

My biggest learning was balancing technical performance with practical impact, especially when dealing with imbalanced data where the minority outcome carries significant importance.

As a pharmacist transitioning deeper into data science, this project represents my growing interest in applying machine learning beyond finance.

The same principles used in credit risk prediction — classification, risk stratification, segmentation, and predictive analytics — can support healthcare applications such as patient risk prediction, medication access analysis, and public health decision-making.

This project therefore, represents an important step in my machine learning journey as I continue developing deeper expertise, with a long-term interest in applying predictive analytics and AI to healthcare challenges.
