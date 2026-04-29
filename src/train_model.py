import pandas as pd
from data_preprocessing import get_data
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import joblib

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load data
data = get_data()

print("Data Loaded Successfully!")
print(data.head())

# Correct column names
X = data.drop("warehouse_sales", axis=1)
y = data["warehouse_sales"]

# Encoding
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Model v1 MSE:", mse)

# Tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluate tuned
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
print("Model v2 (Tuned) MSE:", mse_best)

# Save models
joblib.dump(model, "models/model_v1.pkl")
joblib.dump(best_model, "models/model_v2.pkl")

print("Models saved successfully!")

# Save logs
log = pd.DataFrame({
    "Version": ["v1", "v2"],
    "Model": ["RandomForest", "Tuned RandomForest"],
    "MSE": [mse, mse_best]
})

log.to_csv("logs/model_log.csv", index=False)

print("Model log saved!")
