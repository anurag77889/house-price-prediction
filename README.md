# 🏡 USA Housing Price Prediction

## 📘 Project Overview
This project aims to predict house prices in the USA based on several features such as average income, house age, number of rooms, and population using a **Linear Regression** model.

It’s part of my **30 Days of Machine Learning Sprint**, where I build one real-world ML project every day to strengthen my data science skills and showcase them publicly.

---

## 🎯 Objectives
- Build a regression model to predict housing prices.  
- Understand relationships between different area-level features.  
- Evaluate model accuracy using statistical metrics.  
- Visualize the performance and residuals.

---

## 🧠 Dataset Information
**Dataset:** [USA Housing Dataset (Kaggle)](https://www.kaggle.com/datasets)  
**Columns:**
- `Avg. Area Income` — Average income of residents in the area  
- `Avg. Area House Age` — Average age of houses  
- `Avg. Area Number of Rooms` — Average number of rooms per house  
- `Avg. Area Number of Bedrooms` — Average number of bedrooms per house  
- `Area Population` — Population of the area  
- `Price` — Target variable (house price)  
- `Address` — Dropped during preprocessing (non-numeric)

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 🧩 Project Workflow
1. **Import Libraries** – Load essential Python packages.  
2. **Load Dataset** – Import the USA Housing CSV file.  
3. **Data Cleaning** – Drop irrelevant columns and check for missing data.  
4. **Exploratory Data Analysis (EDA)** – Visualize data trends and correlations.  
5. **Feature Selection** – Select relevant features for modeling.  
6. **Model Training** – Apply Linear Regression.  
7. **Model Evaluation** – Use MAE, RMSE, and R² metrics.  
8. **Visualization** – Plot actual vs predicted prices and residuals.

---

## 📊 Results
- **Model Used:** Linear Regression  
- **R² Score:** 0.91  
- **MAE:** ~80,000 (approximate)  
- **Key Insights:**
  - *Average Area Income* and *House Age* were the strongest predictors.
  - The model explained 91% of the variance in housing prices.

---

## 📈 Visualizations
- Actual vs Predicted Price Scatter Plot  
- Residual Distribution Plot  

<img width="1366" height="768" alt="Screenshot (184)" src="https://github.com/user-attachments/assets/9f0c7189-e020-4519-9625-14c79c77bc5c" />

---

## 🚀 Future Improvements
- Try advanced models like Random Forest, Gradient Boosting, or XGBoost.  
- Include categorical features like location or city.  
- Deploy using Streamlit or Flask.

---

## 🧾 Author
👤 **Anurag Swarnakar**  
Part of the *30 Days of Machine Learning Sprint* 🚀  
Connect with me on [LinkedIn](https://www.linkedin.com/)  

---

## 🏷️ Tags
`#MachineLearning` `#Regression` `#DataScience` `#Python` `#LinearRegression` `#MLProjects`
