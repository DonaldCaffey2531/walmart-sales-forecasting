📊 Walmart Sales Forecasting with AI
🔹 Project Overview

This project builds an AI-powered sales forecasting dashboard using the Walmart Store Sales dataset (Kaggle).
The goal is to predict future weekly sales across Walmart stores using time-series forecasting models (Prophet, XGBoost) and visualize the results in an interactive Streamlit dashboard.

🔹 Features

Data cleaning & preprocessing (missing values, holiday events, etc.)

Exploratory Data Analysis (sales trends, seasonal patterns)

Forecasting with Prophet + ML models

Interactive dashboard with filters (Store, Date range)

Actual vs Forecast comparison with confidence intervals

Ready for deployment (Streamlit Cloud / Hugging Face Spaces)

🔹 Tech Stack

Python: Pandas, Numpy, Scikit-learn

Forecasting: Prophet, XGBoost

Visualization: Plotly, Seaborn

Dashboard: Streamlit

Version Control: Git & GitHub

🔹 Dataset

Source: Walmart Store Sales Forecasting (Kaggle)

Data includes weekly sales across multiple Walmart stores with features like:

Store, Dept, Date

Weekly_Sales

Holidays, Fuel Price, CPI, Unemployment

🔹 Installation & Setup

Clone the repository:

git clone https://github.com/your-username/walmart-sales-forecasting.git
cd walmart-sales-forecasting


Install dependencies:

pip install -r requirements.txt


Run the dashboard:

streamlit run app/dashboard.py

🔹 Usage

Select a store in the dashboard.

View historical vs predicted sales.

Adjust forecast horizon (weeks ahead).

Use insights to plan inventory, staffing, and promotions.

🔹 Results

Prophet model captures weekly and yearly seasonality.

XGBoost improves short-term prediction accuracy.

Dashboard provides actionable AI insights for business planning.

(Insert example chart here — you can save a Plotly figure as a PNG and upload it.)

🔹 Future Improvements

Add deep learning models (LSTM, Transformers).

Integrate real-time data updates.

Build multi-store forecast comparison dashboards.

🔹 Author

👤 Donald Caffey
📧 crazy2531@outlook.com

🌐 LinkedIn Profile
