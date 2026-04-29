import pandas as pd

# Load data
df = pd.read_csv("data/processed/data_v1.csv")

# 1. Check missing values
print("Missing Values:\n", df.isnull().sum())

# 2. Check duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# 3. Data types
print("\nData Types:\n", df.dtypes)

# 4. Basic statistics
print("\nSummary:\n", df.describe())