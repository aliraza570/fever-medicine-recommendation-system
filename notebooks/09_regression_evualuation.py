# ============================================
# Ridge Regressor – COMPLETE EVALUATION
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------------
# Load Feature Selected Data WITH Target
# --------------------------------------------
DATA_PATH = "data/processed/train_selected_features_with_target.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("✅ Data loaded successfully")
print(df.shape)

# --------------------------------------------
# Target & Features
# --------------------------------------------
target_col = "Temperature"

X = df.drop(columns=[target_col])
y = df[target_col]

# --------------------------------------------
# Scaling
# --------------------------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])

# --------------------------------------------
# Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# Ridge Regressor (Best Params)
# --------------------------------------------
model = Ridge(
    alpha=0.6770314141029844,
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------
# Predictions
# --------------------------------------------
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# --------------------------------------------
# Metrics
# --------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n✅ REGRESSION EVALUATION")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# ============================================
# GRAPHS (ALL BLUE)
# ============================================

# 1️⃣ Actual vs Predicted
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="black", linestyle="--")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted")
plt.show()

# 2️⃣ Residuals vs Predicted
plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals, color="blue", alpha=0.6)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

# 3️⃣ Residual Distribution
plt.figure(figsize=(6,5))
sns.histplot(residuals, kde=True, color="blue")
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.show()

# 4️⃣ Prediction Error Line Plot
plt.figure(figsize=(7,5))
plt.plot(y_test.values, color="blue", label="Actual")
plt.plot(y_pred, color="black", linestyle="--", label="Predicted")
plt.title("Actual vs Predicted (Line)")
plt.legend()
plt.show()

# 5️⃣ Feature Coefficients (Importance)
coef_df = pd.DataFrame({
    "Feature": numeric_cols,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient")

plt.figure(figsize=(8,6))
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="blue")
plt.title("Ridge Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.show()

# --------------------------------------------
# Save Evaluation Metrics
# --------------------------------------------
metrics_df = pd.DataFrame({
    "RMSE": [rmse],
    "MAE": [mae],
    "R2_Score": [r2]
})

save_path = "data/processed/ridge_regression_evaluation.csv"
metrics_df.to_csv(save_path, index=False)

print(f"✅ Evaluation metrics saved at: {save_path}")

# -----------------------------
# Save Predictions, Actuals, Residuals
# -----------------------------
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Residual": residuals
})

pred_save_path = "data/processed/ridge_regression_predictions.csv"
results_df.to_csv(pred_save_path, index=False)
print(f"✅ Predictions, Actuals & Residuals saved at: {pred_save_path}")
