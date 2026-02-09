# ===============================
# Model Training: Ridge Regressor on Selected Features
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")
import os

# -----------------------------
# Load Train-Selected Features with Target
# -----------------------------
train_file = "data/processed/train_selected_features_with_target.csv"  # ONLY processed feature CSV

if not os.path.exists(train_file):
    raise FileNotFoundError(f"Train file not found: {train_file}")

df = pd.read_csv(train_file)
print("✅ Train-selected features loaded")
print("Shape:", df.shape)

# -----------------------------
# Define Target Column
# -----------------------------
target_col = "Temperature"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in CSV. Columns available: {df.columns.tolist()}")

# Features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# Scale Numeric Features
# -----------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])
print(f"Numeric features used: {len(numeric_cols)}")

# -----------------------------
# Ridge Regressor with Best Hyperparameters
# -----------------------------
best_params = {
    "alpha": 0.6770314141029844,
    "max_iter": 1000,
    "solver": "auto",
    "random_state": 42
}

regressor = Ridge(**best_params)

# -----------------------------
# 5-Fold Cross Validation (RMSE)
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Use negative MSE in cross_val_score and then convert to RMSE
neg_mse_scores = cross_val_score(regressor, X_scaled, y, cv=cv, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-neg_mse_scores)

print("\n✅ Regression Model Training Completed")
print("Cross-Validated RMSE Scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Best Hyperparameters:", best_params)

# -----------------------------
# Save Training Results to CSV
# -----------------------------
df_results = pd.DataFrame({
    "RMSE_Score": rmse_scores
})

save_path = "data/processed/training_regressor_results.csv"
df_results.to_csv(save_path, index=False)
print(f"✅ Model training results saved successfully at: {save_path}")
