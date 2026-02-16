# ===============================
# Model Training: Ridge Regressor on Selected Features
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# -----------------------------
# Create folders if not exist
# -----------------------------
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Load Train Data
# -----------------------------
train_file = "data/processed/train_selected_features_with_target.csv"

if not os.path.exists(train_file):
    raise FileNotFoundError(f"Train file not found: {train_file}")

df = pd.read_csv(train_file)
print("Train-selected features loaded")
print("Shape:", df.shape)

# -----------------------------
# Target Column
# -----------------------------
target_col = "Temperature"

if target_col not in df.columns:
    raise ValueError("Target column not found")

X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# Scale Features
# -----------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])
print("Numeric features used:", len(numeric_cols))

# -----------------------------
# Ridge Regressor
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

neg_mse_scores = cross_val_score(regressor, X_scaled, y, cv=cv, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-neg_mse_scores)

print("\nRegression Model Training Completed")
print("RMSE Scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())

# -----------------------------
# Save Results CSV
# -----------------------------
df_results = pd.DataFrame({
    "Fold": np.arange(1, len(rmse_scores)+1),
    "RMSE": rmse_scores
})

csv_path = "data/processed/training_regressor_results.csv"
df_results.to_csv(csv_path, index=False)
print("Training results saved:", csv_path)

# ==========================================================
# GRAPH 1: BAR PLOT
# ==========================================================
plt.figure(figsize=(7,5), dpi=300)
plt.bar(df_results["Fold"], df_results["RMSE"])

plt.xlabel("Fold Number")
plt.ylabel("RMSE")
plt.title("Regressor Model Trained - Cross Validation RMSE (Bar View)")

save_path1 = "reports/figures/regressor_cross_validation_bar.png"
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Graph saved:", save_path1)

# ==========================================================
# GRAPH 2: LINE PLOT
# ==========================================================
plt.figure(figsize=(7,5), dpi=300)
plt.plot(df_results["Fold"], df_results["RMSE"], marker='o', label="Fold RMSE")
plt.axhline(df_results["RMSE"].mean(), linestyle="--", label="Mean RMSE")

plt.xlabel("Fold Number")
plt.ylabel("RMSE")
plt.title("Regressor Model Trained - Cross Validation RMSE Trend")
plt.legend()

save_path2 = "reports/figures/regressor_cross_validation_line.png"
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Graph saved:", save_path2)

# ==========================================================
# GRAPH 3: MODERN SWARM PLOT (WITH DETAIL)
# ==========================================================
plt.figure(figsize=(7,5), dpi=300)

sns.swarmplot(y=rmse_scores, size=8)

mean_rmse = rmse_scores.mean()
best_rmse = rmse_scores.min()
worst_rmse = rmse_scores.max()

# Mean line
plt.axhline(mean_rmse, linestyle="--", linewidth=2, label=f"Mean RMSE = {mean_rmse:.3f}")

plt.ylabel("RMSE Value")
plt.title("Regressor Model Trained - RMSE Distribution (Swarm View)")
plt.legend()

# Supporting detail text
detail_text = (
    "Each dot = 1 fold RMSE\n"
    "Lower RMSE = better model\n"
    "Tight cluster = stable model\n"
    f"Best RMSE = {best_rmse:.3f}\n"
    f"Worst RMSE = {worst_rmse:.3f}"
)

plt.text(
    0.02, 0.75,
    detail_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    bbox=dict(facecolor='white', alpha=0.6)
)

save_path3 = "reports/figures/regressor_swarm_rmse.png"
plt.savefig(save_path3, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Swarm plot saved:", save_path3)
