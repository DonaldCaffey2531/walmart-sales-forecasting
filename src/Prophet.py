from prophet import Prophet

# Prophet needs columns: ds (date), y (target)
prophet_df = store_df.rename(columns={"Date":"ds", "Weekly_Sales":"y"})

# Train-test split
train = prophet_df.iloc[:-12]
test = prophet_df.iloc[-12:]

# Fit model
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_model.fit(train)

# Forecast
future = prophet_model.make_future_dataframe(periods=len(test), freq="W")
forecast = prophet_model.predict(future)

# Evaluate
prophet_pred = forecast.iloc[-len(test):]["yhat"]
prophet_rmse = np.sqrt(mean_squared_error(test["y"], prophet_pred))
print("Prophet RMSE:", prophet_rmse)

# Plot
prophet_model.plot(forecast)
