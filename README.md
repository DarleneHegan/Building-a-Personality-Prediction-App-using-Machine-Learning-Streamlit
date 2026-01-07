# Building-a-Personality-Prediction-App-using-Machine-Learning-Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://apps-personality-prediction.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 🚀 Live Demo
**Try the application here:** 👉 **[Personality Prediction Web App](https://apps-personality-prediction.streamlit.app/)**

---

## 📌 Project Overview
This project is an implementation of a machine learning classification model designed to predict a person's personality type—**Introvert** or **Extrovert**—based on their daily behavioral patterns.

The primary goal was to analyze how social habits (such as time spent alone, friend circle size, and social media activity) correlate with personality types and to deploy the best-performing model into an interactive web application using **Streamlit**.

## 📊 Dataset
The dataset was sourced from Kaggle, consisting of **2,900 samples**.
- **Features:** 7 behavioral attributes (e.g., social energy, network size, posting frequency).
- **Target:** Personality Type (Binary Class: Introvert / Extrovert).

## 🔍 Exploratory Data Analysis (EDA) Insights
During the analysis phase, distinct behavioral patterns emerged:
* **Introverts:** Tend to spend more time alone, maintain smaller friend circles, post less frequently on social media, and often report feeling drained after socializing.
* **Extroverts:** Are generally more socially active, maintain larger friend circles, post frequently, and rarely feel drained after social interactions.

## ⚙️ Methodology
The project followed a structured End-to-End Machine Learning pipeline:

1.  **Data Preparation:** Data cleaning, label encoding, and feature scaling.
2.  **Modeling:** A comparative analysis was conducted using four algorithms:
    * Logistic Regression
    * Support Vector Machine (SVM)
    * Random Forest
    * XGBoost
3.  **Evaluation:** Models were assessed based on Accuracy, Precision, Recall, and F1-Score.
4.  **Deployment:** The best model was deployed via Streamlit Cloud.

## 🏆 Model Performance
After extensive testing, **Support Vector Machine (SVM)** emerged as the best-performing model.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **93%** |
| **Precision** | **91%** |
| **Recall** | **93%** |
| **F1-Score** | **92%** |

> *These results demonstrate that daily social behaviors serve as strong predictors for distinguishing personality types.*

## 🛠 Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
