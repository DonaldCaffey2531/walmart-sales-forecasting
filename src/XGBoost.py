import xgboost as xgb
from sklearn.model_selection import train_test_split

# Feature engineering
df["Lag1"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(1)
df["Lag4"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(4)
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Drop missing rows (from lag features)
df_ml = df.dropna()

# Example for Store 1, Dept 1
store_ml = df_ml[(df_ml["Store"]==1)&(df_ml["Dept"]==1)]

X = store_ml[["Lag1","Lag4","Month","Year"]]
y = store_ml["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost
model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
model_xgb.fit(X_train, y_train)

# Predict
y_pred = model_xgb.predict(X_test)

# Evaluate
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("XGBoost RMSE:", xgb_rmse)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="XGBoost Forecast")
plt.legend()
plt.show()
