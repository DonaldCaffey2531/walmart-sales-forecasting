import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Example: Filter for one store & department
store_df = df[(df["Store"] == 1) & (df["Dept"] == 1)].copy()
store_df = store_df.sort_values("Date")

# Train-test split
train = store_df.iloc[:-12]
test = store_df.iloc[-12:]

# Fit ARIMA model
model = ARIMA(train["Weekly_Sales"], order=(2,1,2))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate
rmse = np.sqrt(mean_squared_error(test["Weekly_Sales"], forecast))
print("ARIMA RMSE:", rmse)
