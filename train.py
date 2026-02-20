"""
Medical Cost Personal Dataset
Linear Regression Model to Predict Charges

Expected loss (MSE) <= 19,000,000
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1. Load dataset
# =========================

data_path = "./insurance.csv"
df = pd.read_csv(data_path)

# =========================
# 2. Preprocessing
# =========================

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. Train Model
# =========================

model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 4. Predictions
# =========================

y_pred = model.predict(X_test)

# =========================
# 5. Evaluation
# =========================

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== Model Performance ===")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

if mse <= 19000000:
    print("Requirement met: Loss <= 19,000,000")
else:
    print("Requirement NOT met")

# =========================
# 6. Save report as CSV
# =========================

os.makedirs("../results", exist_ok=True)

report = pd.DataFrame({
    "MSE": [mse],
    "RMSE": [rmse],
    "R2_Score": [r2]
})

report.to_csv("../results/report.csv", index=False)

print("Report saved to results/report.csv")