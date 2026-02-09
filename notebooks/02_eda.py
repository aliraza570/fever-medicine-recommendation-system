# ===============================
# Medicine ML Project: Data Cleaning, Normalization, and Correlation Analysis
# ===============================

import pandas as pd
import numpy as np
import os
from scipy.stats import shapiro, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -----------------------------
# Step 0: Load Cleaned Dataset
# -----------------------------
df = pd.read_csv("data/processed/cleaned_medicine_data.csv")
print(f"✅ Loaded dataset with shape: {df.shape}\n")
print(df.head())

# -----------------------------
# Step 1: Select Numeric Columns
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nNumeric Columns:", list(numeric_cols))

# -----------------------------
# Step 2: Shapiro-Wilk Normality Test (Before Normalization)
# -----------------------------
print("\nShapiro–Wilk Normality Test (Before Normalization):")
for col in numeric_cols:
    stat, p_value = shapiro(df[col])
    result = "NORMAL" if p_value > 0.05 else "NOT NORMAL"
    print(f"{col}: p-value = {p_value:.5f} → {result}")

# -----------------------------
# Step 3: Histograms with KDE for Numeric Features
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, color='blue', bins=30)
    plt.title(f"Normality Check: {col}", fontsize=14, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Step 4: Normalization
# -----------------------------
# Option 1: Standardization (mean=0, std=1)
scaler = StandardScaler()
# Option 2: Min-Max Scaling (0-1)
# scaler = MinMaxScaler()

df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# Step 5: Shapiro-Wilk Normality Test (After Normalization)
# -----------------------------
print("\nShapiro–Wilk Normality Test (After Normalization):")
for col in numeric_cols:
    stat, p_value = shapiro(df_scaled[col])
    result = "NORMAL" if p_value > 0.05 else "NOT NORMAL"
    print(f"{col}: p-value = {p_value:.5f} → {result}")

# -----------------------------
# Step 6: Spearman Correlation and P-values
# -----------------------------
corr_matrix = df_scaled[numeric_cols].corr(method='spearman')

# P-value matrix
pval_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
for col1 in numeric_cols:
    for col2 in numeric_cols:
        _, pval = spearmanr(df_scaled[col1], df_scaled[col2])
        pval_matrix.loc[col1, col2] = pval
pval_matrix = pval_matrix.astype(float)

print("\nSpearman Correlation Matrix:\n", corr_matrix.round(3))
print("\nP-value Matrix:\n", pval_matrix.round(5))

# -----------------------------
# Step 7: Heatmap Visualization
# -----------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, 
    annot=True,        # Show correlation coefficients
    fmt=".2f", 
    cmap="Blues",      # Professional blue palette
    square=True, 
    linewidths=0.5, 
    cbar_kws={'shrink': 0.8, 'label': 'Spearman r'}
)
plt.title("Spearman Correlation Heatmap (Non-Parametric)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 8: Save CSV Files (Full Dataset + Correlation Matrix)
# -----------------------------
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

# 1️⃣ Save full normalized dataset (1000 rows × 20 columns)
normalized_file = os.path.join(save_dir, "cleaned_normalized_data.csv")
df_scaled.to_csv(normalized_file, index=False)
print(f"✅ Full normalized dataset saved at: {normalized_file}")
print(f"Shape: {df_scaled.shape}")  # Should show (1000, 20)

# 2️⃣ Save Spearman correlation matrix 
corr_file = os.path.join(save_dir, "spearman_correlation.csv")
corr_matrix.to_csv(corr_file, index=True)
print(f"✅ Spearman correlation matrix saved at: {corr_file}")
print(f"Correlation matrix shape: {corr_matrix.shape}")  
