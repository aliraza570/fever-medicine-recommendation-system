import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -----------------------------
# Load TRAIN data
# -----------------------------
train_df = pd.read_csv("data/processed/train_feature_engineered.csv")

print("Train data loaded successfully")
print("Shape:", train_df.shape)

# -----------------------------
# Target column
# -----------------------------
target_col = "Temperature"  # Update your target here

# -----------------------------
# Separate X and y (NO DATA LEAKAGE)
# -----------------------------
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# -----------------------------
# Numeric features only
# -----------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
print(f"Numeric features after removing constants: {len(numeric_cols)}")

# -----------------------------
# Spearman correlation (non-parametric)
# -----------------------------
records = []
for col in numeric_cols:
    # Skip constant columns to avoid warnings
    if X[col].nunique() <= 1:
        continue
    corr, p_val = spearmanr(X[col], y)
    records.append([col, corr, p_val])

corr_df = pd.DataFrame(
    records,
    columns=["Feature", "Spearman_Corr", "P_Value"]
)

# -----------------------------
# INDUSTRY THRESHOLD
# |corr| >= 0.20 and p-value < 0.05
# -----------------------------
selected_df = corr_df[
    (corr_df["P_Value"] < 0.05) &
    (corr_df["Spearman_Corr"].abs() >= 0.20)
].sort_values(
    by="Spearman_Corr",
    key=lambda x: x.abs(),
    ascending=False
)

# -----------------------------
# Final selected features
# -----------------------------
selected_features = selected_df["Feature"].tolist()

# ✅ Add the target column to the selected features DataFrame
X_selected = train_df[selected_features + [target_col]]

# -----------------------------
# OUTPUT
# -----------------------------
print("\n✅ STRONGLY RELATED FEATURES WITH TARGET (Temperature)\n")

if selected_features:
    print("Selected Feature Names:")
    for f in selected_features:
        print(f"- {f}")
else:
    print("⚠️ No feature passed the selection criteria")

print("\nSelected Feature Data Shape:")
print(X_selected.shape)

# -----------------------------
# OPTIONAL: Show head
# -----------------------------
print("\nSelected Features Head:")
print(X_selected.head())

# -----------------------------
# Correlation Heatmap (Blue Palette)
# -----------------------------
if selected_features:  # Only if we have features
    plt.figure(figsize=(12, 10))
    corr_matrix = X_selected.corr(method='spearman')
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        square=True, 
        linewidths=0.5, 
        cbar_kws={'shrink': 0.8, 'label': 'Spearman r'}
    )
    plt.title("Spearman Correlation Heatmap (Selected Features + Target)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No features selected → correlation heatmap cannot be plotted")

# -----------------------------
# Folder to save
# -----------------------------
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# File path
# -----------------------------
selected_file = os.path.join(save_dir, "train_selected_features_with_target.csv")

# -----------------------------
# Save to CSV
# -----------------------------
X_selected.to_csv(selected_file, index=False)

print(f"✅ Selected features with target saved at: {selected_file}")
