import pandas as pd
import os

# Make sure folder exists
os.makedirs("data/version_1", exist_ok=True)

# Load original dataset
df = pd.read_csv("data/version_1/Warehouse_and_Retail_Sales.csv")

print("Original Shape:", df.shape)
print("Original Columns:", df.columns)

# STEP 1: Keep only numeric columns
df = df.select_dtypes(include=["number"])

# STEP 2: Remove missing values
df.dropna(inplace=True)

# STEP 3: Remove duplicate rows
df.drop_duplicates(inplace=True)

# STEP 4: Standardize column names (lowercase)
df.columns = [col.lower() for col in df.columns]

# STEP 5: Save cleaned dataset
df.to_csv("data/version_1/data_v1.csv", index=False)

print("✅ Preprocessing Complete")
print("Final Shape:", df.shape)