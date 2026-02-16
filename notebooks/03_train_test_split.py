# ===============================
# Train-Test Split, Plots & CSV Save
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.plotting import parallel_coordinates

# -----------------------------
# Step 0: Load normalized dataset
# -----------------------------
df = pd.read_csv("data/processed/cleaned_normalized_data.csv")
print(f"Dataset loaded with shape: {df.shape}")

# -----------------------------
# Create figure save folder for report
# -----------------------------
fig_dir = "reports/figures"
os.makedirs(fig_dir, exist_ok=True)

# -----------------------------
# Step 1: Define Features & Target
# -----------------------------
target_col = "Age"
X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# Step 2: Train-Test Split (80/20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

train_df = X_train.copy()
train_df[target_col] = y_train

test_df = X_test.copy()
test_df[target_col] = y_test

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# -----------------------------
# Step 3: Numeric Columns
# -----------------------------
numeric_cols = train_df.select_dtypes(include=['number']).columns

# -----------------------------
# Step 4: Train Set Spearman Heatmap (BLUE)
# -----------------------------
plt.figure(figsize=(12, 10), dpi=300)
sns.heatmap(
    train_df[numeric_cols].corr(method='spearman'),
    annot=True,
    fmt=".2f",
    cmap="Blues",
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Spearman r'}
)
plt.title("Train Set Spearman Correlation Heatmap", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "train_spearman_heatmap.png"), dpi=300)
plt.show()

# -----------------------------
# Step 5: Test Set Spearman Heatmap (BLUE)
# -----------------------------
plt.figure(figsize=(12, 10), dpi=300)
sns.heatmap(
    test_df[numeric_cols].corr(method='spearman'),
    annot=True,
    fmt=".2f",
    cmap="Blues",
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Spearman r'}
)
plt.title("Test Set Spearman Correlation Heatmap", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "test_spearman_heatmap.png"), dpi=300)
plt.show()

# =========================================================
# Step 6: TRAIN PARALLEL COORDINATES PLOT (BLUE)
# =========================================================
train_plot_df = train_df.copy()

# Age ko category banaya for color grouping
train_plot_df['Age_Group'] = pd.qcut(train_plot_df['Age'], q=4, labels=["Low","Mid-Low","Mid-High","High"])

plt.figure(figsize=(18,8), dpi=300)
parallel_coordinates(
    train_plot_df[numeric_cols.tolist() + ['Age_Group']],
    class_column='Age_Group',
    color=['blue','blue','blue','blue'],
    alpha=0.3
)
plt.title("Train Feature Parallel Coordinates Plot", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "train_parallel_plot.png"), dpi=300)
plt.show()

# =========================================================
# Step 7: TEST PARALLEL COORDINATES PLOT (BLUE)
# =========================================================
test_plot_df = test_df.copy()
test_plot_df['Age_Group'] = pd.qcut(test_plot_df['Age'], q=4, labels=["Low","Mid-Low","Mid-High","High"])

plt.figure(figsize=(18,8), dpi=300)
parallel_coordinates(
    test_plot_df[numeric_cols.tolist() + ['Age_Group']],
    class_column='Age_Group',
    color=['blue','blue','blue','blue'],
    alpha=0.3
)
plt.title("Test Feature Parallel Coordinates Plot", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "test_parallel_plot.png"), dpi=300)
plt.show()

# -----------------------------
# Step 8: Save Train & Test CSVs
# -----------------------------
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

train_file = os.path.join(save_dir, "train_data.csv")
train_df.to_csv(train_file, index=False)
print(f"Train dataset saved at: {train_file}")

test_file = os.path.join(save_dir, "test_data.csv")
test_df.to_csv(test_file, index=False)
print(f"Test dataset saved at: {test_file}")
