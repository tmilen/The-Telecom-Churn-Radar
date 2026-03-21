# The-Telecom-Churn-Radar

A machine learning-powered web application that predicts whether a telecom customer is likely to churn based on behavioral, service, and financial data.

---

## Project Overview

Customer churn is a critical problem in the telecom industry. Identifying customers who are likely to leave allows businesses to take proactive actions to improve retention.

This project builds a complete end-to-end machine learning solution that:
- Cleans and preprocesses customer data
- Explores patterns through EDA
- Trains multiple classification models
- Deploys the best model using a Streamlit web application

---

## Model Development

Three classification models were implemented and compared:

- Logistic Regression ✅ (Selected Model)
- Decision Tree
- Random Forest

### 📊 Model Performance

| Model | Accuracy |
|------|--------|
| Logistic Regression | **93.33%** |
| Random Forest | 90.00% |
| Decision Tree | 86.67% |

Logistic Regression was selected as the final model due to its strong performance and better generalization.

---

## Overfitting Handling

The dataset contains only **150 samples**, which increases the risk of overfitting.

To address this:

- A **train-test split (80/20)** with stratification was applied
- A **pipeline** was used to prevent data leakage
- Model performance was evaluated on unseen test data

### Result:
- Train Accuracy: **98.33%**
- Test Accuracy: **93.33%**

The small gap (~5%) indicates **mild overfitting**, which is acceptable given the dataset size. Overall, the model generalizes well.

---

## Data Preprocessing

The following preprocessing steps were applied:

- Missing values handled based on business meaning:
  - Invoice-related columns → filled with 0
  - Numerical features → filled with median
- Categorical variables → One-Hot Encoding
- Numerical features → StandardScaler
- Date columns → removed (for simplicity)

All preprocessing steps were integrated using a **Pipeline** to ensure consistency and avoid data leakage.

---

## Web Application (Streamlit)

[Visit the live website](https://telecom-churn-radar.streamlit.app/)

A simple and interactive web interface was built using **Streamlit**, allowing users to:

- Input customer details
- Predict churn status
- View churn probability
- Visualize feature importance (target graph)

> Note: In real-world systems, many features are retrieved automatically from backend databases. In this project, all required inputs are collected via the UI to match the trained model.

---

## Target Graph

The application includes a **Feature Importance Chart**, showing which factors most influence churn prediction.

This helps in understanding:
- Customer behavior patterns
- Key drivers of churn

---

## How to Run the Application

### 1. Clone the repository in local

### 2. Set up the virtual environment using the command prompt and activate it

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit app

```bash
streamlit run app.py
```

---

## Conclusion

The final Logistic Regression model achieved strong performance with 93% accuracy, demonstrating its effectiveness in predicting customer churn.

Despite the small dataset size, careful preprocessing and proper evaluation ensured a reliable and generalizable model.

This project highlights how machine learning can support data-driven decision-making in real-world business scenarios.
