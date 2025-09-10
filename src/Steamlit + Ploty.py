import streamlit as st
import plotly.express as px
import pandas as pd

# Load Walmart sales data (example)
df = pd.read_csv("./../datasets/xgboost_forecasts.csv")

st.title("📊 Walmart Sales Forecasting Dashboard")

# Sidebar filters
store = st.selectbox("Select Store:", df["Store"].unique())
dept = st.selectbox("Select Department:", df["Dept"].unique())

# Filter data
filtered = df[(df["Store"] == store) & (df["Dept"] == dept)]

# Plotly line chart
fig = px.line(filtered, x="Date", y="Weekly_Sales", title="Weekly_Sales")
fig.add_scatter(x=filtered["Date"], y=filtered["Forecast_XGBoost"], 
                mode="lines", name="Forecast_XGBoost")


st.plotly_chart(fig)