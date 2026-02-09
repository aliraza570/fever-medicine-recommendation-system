# ===============================
# Train-Test Split, Plots & CSV Save (Corrected)
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Step 0: Load normalized dataset
# -----------------------------
df = pd.read_csv("data/processed/cleaned_normalized_data.csv")
print(f"✅ Dataset loaded with shape: {df.shape}")

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

# Merge X + y for saving
train_df = X_train.copy()
train_df[target_col] = y_train

test_df = X_test.copy()
test_df[target_col] = y_test

print(f"\n✅ Train shape: {train_df.shape}")
print(f"✅ Test shape:  {test_df.shape}")

# -----------------------------
# Step 3: Numeric Columns
# -----------------------------
numeric_cols = train_df.select_dtypes(include=['number']).columns

# -----------------------------
# Step 4: Train Set Spearman Heatmap
# -----------------------------
plt.figure(figsize=(12, 10))
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
plt.show()

# -----------------------------
# Step 5: Test Set Spearman Heatmap
# -----------------------------
plt.figure(figsize=(12, 10))
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
plt.show()

# -----------------------------
# Step 6: Train Set Countplots
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df[col], bins=30, color='blue', kde=False)
    plt.title(f"Train Set Countplot: {col}", fontsize=14, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Step 7: Test Set Countplots
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(test_df[col], bins=30, color='blue', kde=False)
    plt.title(f"Test Set Countplot: {col}", fontsize=14, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Step 8: Save Train & Test CSVs
# -----------------------------
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

train_file = os.path.join(save_dir, "train_data.csv")
train_df.to_csv(train_file, index=False)
print(f"✅ Train dataset saved at: {train_file}")

test_file = os.path.join(save_dir, "test_data.csv")
test_df.to_csv(test_file, index=False)
print(f"✅ Test dataset saved at: {test_file}")
