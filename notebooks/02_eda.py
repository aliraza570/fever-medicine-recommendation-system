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
print("Loaded dataset with shape:", df.shape)
print(df.head())

# -----------------------------
# Graph save path for report
# -----------------------------
report_fig_dir = os.path.join("reports", "figures")
os.makedirs(report_fig_dir, exist_ok=True)

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
    print(f"{col}: p-value = {p_value:.5f} -> {result}")

# ==========================================================
# Step 3: ALL FEATURES IN ONE IMAGE (VIOLIN + BOX) BEFORE NORMALIZATION
# ==========================================================
rows = len(numeric_cols)
cols = 2
plt.figure(figsize=(14, rows*2), dpi=300)

for i, col in enumerate(numeric_cols):
    plt.subplot(rows, cols, i+1)

    sns.violinplot(y=df[col], color="blue", inner=None)
    sns.boxplot(y=df[col], width=0.25, color="blue")

    plt.title(f"{col}", fontsize=10, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("")

plt.suptitle("All Features Distribution Before Normalization", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])

save_path = os.path.join(report_fig_dir, "all_features_distribution_before_norm.png")
plt.savefig(save_path, dpi=300)
plt.show()

# -----------------------------
# Step 4: Normalization
# -----------------------------
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# Step 5: Shapiro-Wilk Normality Test (After Normalization)
# -----------------------------
print("\nShapiro–Wilk Normality Test (After Normalization):")
for col in numeric_cols:
    stat, p_value = shapiro(df_scaled[col])
    result = "NORMAL" if p_value > 0.05 else "NOT NORMAL"
    print(f"{col}: p-value = {p_value:.5f} -> {result}")

# ==========================================================
# Step 6: ALL FEATURES IN ONE IMAGE AFTER NORMALIZATION
# ==========================================================
rows = len(numeric_cols)
cols = 2
plt.figure(figsize=(14, rows*2), dpi=300)

for i, col in enumerate(numeric_cols):
    plt.subplot(rows, cols, i+1)

    sns.violinplot(y=df_scaled[col], color="blue", inner=None)
    sns.boxplot(y=df_scaled[col], width=0.25, color="blue")

    plt.title(f"{col}", fontsize=10, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("")

plt.suptitle("All Features Distribution After Normalization", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])

save_path = os.path.join(report_fig_dir, "all_features_distribution_after_norm.png")
plt.savefig(save_path, dpi=300)
plt.show()

# -----------------------------
# Step 7: Spearman Correlation and P-values
# -----------------------------
corr_matrix = df_scaled[numeric_cols].corr(method='spearman')

pval_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
for col1 in numeric_cols:
    for col2 in numeric_cols:
        _, pval = spearmanr(df_scaled[col1], df_scaled[col2])
        pval_matrix.loc[col1, col2] = pval
pval_matrix = pval_matrix.astype(float)

print("\nSpearman Correlation Matrix:\n", corr_matrix.round(3))
print("\nP-value Matrix:\n", pval_matrix.round(5))

# -----------------------------
# Step 8: Heatmap Visualization
# -----------------------------
plt.figure(figsize=(12, 10), dpi=300)
sns.heatmap(
    corr_matrix, 
    annot=True,
    fmt=".2f", 
    cmap="Blues",
    square=True, 
    linewidths=0.5, 
    cbar_kws={'shrink': 0.8, 'label': 'Spearman r'}
)
plt.title(
    "Spearman Correlation Heatmap\nUsed to detect strong positive & negative relationships",
    fontsize=16,
    fontweight='bold'
)
plt.tight_layout()

heatmap_path = os.path.join(report_fig_dir, "spearman_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.show()

# -----------------------------
# Step 9: Save CSV Files
# -----------------------------
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

normalized_file = os.path.join(save_dir, "cleaned_normalized_data.csv")
df_scaled.to_csv(normalized_file, index=False)
print("Full normalized dataset saved at:", normalized_file)
print("Shape:", df_scaled.shape)

corr_file = os.path.join(save_dir, "spearman_correlation.csv")
corr_matrix.to_csv(corr_file, index=True)
print("Spearman correlation matrix saved at:", corr_file)
print("Correlation matrix shape:", corr_matrix.shape)
