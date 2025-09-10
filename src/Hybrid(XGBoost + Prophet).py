import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ------------------------------
# 1. Prepare Data
# ------------------------------
# Example: Store 1, Dept 1
store_df = df[(df["Store"]==1) & (df["Dept"]==1)].copy()
store_df = store_df.sort_values("Date")

prophet_df = store_df.rename(columns={"Date":"ds", "Weekly_Sales":"y"})

# Train-test split
train = prophet_df.iloc[:-12]   # all except last 12 weeks
test = prophet_df.iloc[-12:]    # last 12 weeks

# ------------------------------
# 2. Prophet Model (Base Forecast)
# ------------------------------
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
prophet_model.fit(train)

future = prophet_model.make_future_dataframe(periods=len(test), freq="W")
forecast = prophet_model.predict(future)

# Extract Prophet predictions
prophet_pred = forecast[["ds","yhat"]].set_index("ds").iloc[-len(test):]

# ------------------------------
# 3. Calculate Residuals (Error left by Prophet)
# ------------------------------
train_forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=0, freq="W"))
residuals = train["y"].reset_index(drop=True) - train_forecast["yhat"].iloc[:len(train)].reset_index(drop=True)

train["residuals"] = residuals

# ------------------------------
# 4. Train XGBoost on Residuals
# ------------------------------
# Feature Engineering (Lag features, Month, Year)
store_df["Lag1"] = store_df["y"].shift(1)
store_df["Lag4"] = store_df["y"].shift(4)
store_df["Month"] = store_df["ds"].dt.month
store_df["Year"] = store_df["ds"].dt.year

# Align with residuals
train_ml = store_df.iloc[:len(train)].dropna()
train_ml["residuals"] = train["residuals"].iloc[-len(train_ml):].values

X_train = train_ml[["Lag1","Lag4","Month","Year"]]
y_train = train_ml["residuals"]

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)

# ------------------------------
# 5. Predict Residuals on Test Set
# ------------------------------
test_ml = store_df.iloc[-len(test):].dropna()
X_test = test_ml[["Lag1","Lag4","Month","Year"]]

residual_pred = xgb_model.predict(X_test)

# ------------------------------
# 6. Hybrid Forecast = Prophet + Residuals
# ------------------------------
hybrid_pred = prophet_pred["yhat"].values + residual_pred

# Evaluate
rmse = np.sqrt(mean_squared_error(test["y"], hybrid_pred))
print("Hybrid (Prophet + XGBoost) RMSE:", rmse)

# ------------------------------
# 7. Visualization
# ------------------------------
plt.figure(figsize=(12,6))
plt.plot(test["ds"], test["y"], label="Actual")
plt.plot(test["ds"], prophet_pred["yhat"], label="Prophet Only")
plt.plot(test["ds"], hybrid_pred, label="Hybrid Prophet+XGBoost")
plt.legend()
plt.title("Hybrid Forecasting: Prophet + XGBoost")
plt.show()
