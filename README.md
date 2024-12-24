# 📊 **Customer Churn Prediction Project**

## 📌 **Project Overview**
In today's competitive business landscape, **customer retention** is crucial for long-term success. This project aims to **predict customer churn** using advanced machine learning algorithms, helping businesses **identify at-risk customers** and implement **targeted retention strategies**. By analyzing historical customer behavior, demographics, and subscription details, this model empowers data-driven decision-making for **improving customer satisfaction** and **reducing churn rates**.

---

## 🎯 **Problem Statement**
Customer churn can result in significant **revenue loss** and **reduced market share**. This project aims to:
- **Predict the likelihood of customer churn** using machine learning models.
- **Identify critical factors** driving customer churn.
- **Enable proactive engagement strategies** to retain high-risk customers.

---

## 📊 **Dataset Description**
The dataset includes **58,592 customer records** with **44 attributes**, providing information about:
- **CustomerID:** Unique identifier for customers.  
- **Age:** Age of the customer.  
- **Gender:** Male or Female.  
- **Location:** City-based data (e.g., Houston, Los Angeles, Miami).  
- **Subscription_Length_Months:** Duration of subscription in months.  
- **Monthly_Bill:** Monthly payment amount.  
- **Total_Usage_GB:** Total data usage in GB.  
- **Churn:** Binary indicator (1 = Churned, 0 = Retained).  

**Source:** Kaggle Dataset

---

## 🛠️ **Tech Stack & Tools**
- **Programming Language:** Python  
- **Data Manipulation:** Pandas, NumPy  
- **Data Visualization:** Matplotlib, Seaborn  
- **Machine Learning Framework:** Scikit-Learn, TensorFlow, Keras  
- **Classification Algorithms:** Logistic Regression, Decision Tree, KNN, SVM, Naive Bayes, Random Forest, Gradient Boosting, XGBoost  
- **Model Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- **Data Preprocessing:** StandardScaler, PCA, SMOTE  
- **Hyperparameter Tuning:** GridSearchCV  
- **Model Validation:** Cross-Validation, Early Stopping  
- **IDE/Workflow:** Jupyter Notebook  

---

## 📈 **Model Workflow**
1. **Data Preprocessing:**  
   - Handle missing values and outliers.  
   - Apply `StandardScaler` for feature scaling.  
   - Address class imbalance with `SMOTE`.  
2. **Exploratory Data Analysis (EDA):**  
   - Visualize trends and distributions.  
   - Analyze feature importance.  
3. **Feature Engineering:**  
   - Apply `Variance Inflation Factor (VIF)` to remove multicollinearity.  
4. **Model Training & Evaluation:**  
   - Train models: Logistic Regression, Random Forest, XGBoost.  
   - Evaluate using metrics like Accuracy, Precision, Recall, ROC-AUC.  
5. **Hyperparameter Tuning:**  
   - Optimize model parameters using `GridSearchCV`.  
6. **Model Deployment:**  
   - Save best-performing models.  

---

## 📊 **Evaluation Metrics**
- **Accuracy:** Overall correctness of predictions.  
- **Precision:** Proportion of positive predictions that were correct.  
- **Recall:** Ability to detect all positive cases.  
- **F1-Score:** Balance between precision and recall.  
- **ROC-AUC:** Model's ability to distinguish between classes.

---

## 🎯 **Key Findings**
- **Random Forest Classifier** achieved the highest accuracy at **69.14%**.  
- Top churn predictors: **Subscription Length**, **Monthly Bill**, and **Total Usage GB**.  
- Visualization dashboards provided insights for targeted retention strategies.  

---

## 📁 **Project Structure**
