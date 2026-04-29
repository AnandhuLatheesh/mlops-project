import pandas as pd
import os

print(os.listdir("data/raw"))
# Create folders
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/versions/v2", exist_ok=True)

# Load raw data
df = pd.read_csv(r"C:\Users\aleen\OneDrive\Desktop\end-to-end-mlops\data\raw\Warehouse_and_Retail_Sales.csv")

print("Original Shape:", df.shape)
print("Original Columns:", df.columns)

# STEP 1: Keep only numeric columns
df = df.select_dtypes(include=["number"])

# STEP 2: Remove missing values
df.dropna(inplace=True)

# STEP 3: Remove duplicates
df.drop_duplicates(inplace=True)

# STEP 4: Fix column names (lowercase + underscore)
df.columns = df.columns.str.lower().str.replace(" ", "_")

# STEP 5: Remove negative values
df = df[(df['retail_sales'] >= 0) &
        (df['retail_transfers'] >= 0) &
        (df['warehouse_sales'] >= 0)]

# STEP 6: Save cleaned data
df.to_csv("data/processed/data_v2.csv", index=False)

# STEP 7: Save version
df.to_csv("data/versions/v2/data_v2.csv", index=False)

print("✅ Preprocessing Complete")
print("Final Shape:", df.shape)


def get_data():
    return df