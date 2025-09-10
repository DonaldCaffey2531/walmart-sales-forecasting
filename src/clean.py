import pandas as pd 
import numpy as np

#load datasets
train = pd.read_csv("../datasets/train.csv")
features = pd.read_csv("../datasets/feature.csv")
stores = pd.read_csv("../datasets/stores.csv")

#convert Date to datetime
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

# Merge train + features
df = train.merge(features, on=["Store", "Date"], how="left")

# Merge with stores metadata
df = df.merge(stores, on="Store", how="left")

print("Final dataset shape:", df.shape)
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill markdowns with 0 (no promotion that week)
for col in ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]:
    df[col] = df[col].fillna(0)

# Fill CPI/Unemployment with forward fill (carry last value)
df["CPI"] = df["CPI"].fillna(method="ffill")
df["Unemployment"] = df["Unemployment"].fillna(method="ffill")

# Check missing values
print(df.isnull().sum())

# Fill markdowns with 0 (no promotion that week)
for col in ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]:
    df[col] = df[col].fillna(0)

# Fill CPI/Unemployment with forward fill (carry last value)
df["CPI"] = df["CPI"].fillna(method="ffill")
df["Unemployment"] = df["Unemployment"].fillna(method="ffill")

# Drop rows with missing target
df = df.dropna(subset=["Weekly_Sales"])

# Save clean dataset
df.to_csv("data/clean_walmart.csv", index=False)

print("Preprocessing done! Clean dataset saved.")