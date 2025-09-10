import statsmodels.api as sm

# Fit SARIMA model (p,d,q)(P,D,Q,s)
sarima_model = sm.tsa.statespace.SARIMAX(train["Weekly_Sales"],
                                         order=(1,1,1),
                                         seasonal_order=(1,1,1,52), # 52 weeks in year
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)
sarima_fit = sarima_model.fit()

# Forecast
sarima_forecast = sarima_fit.forecast(steps=len(test))

# Evaluate
sarima_rmse = np.sqrt(mean_squared_error(test["Weekly_Sales"], sarima_forecast))
print("SARIMA RMSE:", sarima_rmse)
