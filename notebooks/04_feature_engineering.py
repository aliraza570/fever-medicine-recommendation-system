# ===============================
# Feature Engineering (TRAIN DATA ONLY)
# ===============================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------
# FIXED: Absolute report figure folder
# -----------------------------
BASE_DIR = os.getcwd()  # current project root
fig_dir = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(fig_dir, exist_ok=True)

print("Graphs will save in:", fig_dir)

# -----------------------------
# Load TRAIN dataset only
# -----------------------------
train_df = pd.read_csv("data/processed/train_data.csv")
print("Original Train Shape:", train_df.shape)

# -----------------------------
# Separate Target (VERY IMPORTANT)
# -----------------------------
target_col = "Age"
y_train = train_df[target_col]
X_train = train_df.drop(columns=[target_col])

# -----------------------------
# Feature Engineering Starts
# -----------------------------

# Interaction Feature (Health Index)
X_train["Temp_HeartRate_Index"] = X_train["Temperature"] * X_train["Heart_Rate"]

# Environmental Stress Feature
X_train["Env_Stress_Score"] = (
    X_train["Temperature"] +
    X_train["Humidity"] +
    X_train["AQI"]
)

# BMI Category Feature
X_train["BMI_Category"] = np.where(
    X_train["BMI"] < 18.5, 0,
    np.where(X_train["BMI"] < 25, 1,
    np.where(X_train["BMI"] < 30, 2, 3))
)

# Symptom Severity Score
symptom_cols = [
    "Fever_Severity_encoded",
    "Headache_encoded",
    "Body_Ache_encoded",
    "Fatigue_encoded"
]

X_train["Symptom_Severity_Score"] = X_train[symptom_cols].sum(axis=1)

# Lifestyle Risk Score
lifestyle_cols = [
    "Smoking_History_encoded",
    "Alcohol_Consumption_encoded",
    "Physical_Activity_encoded"
]

X_train["Lifestyle_Risk_Score"] = X_train[lifestyle_cols].sum(axis=1)

# -----------------------------
# Combine engineered features with target
# -----------------------------
train_fe_df = X_train.copy()
train_fe_df[target_col] = y_train

print("\nFeature Engineering Completed")
print("New Train Shape:", train_fe_df.shape)

new_features = [
    "Temp_HeartRate_Index",
    "Env_Stress_Score",
    "BMI_Category",
    "Symptom_Severity_Score",
    "Lifestyle_Risk_Score"
]

print("\nNewly Created Features:")
print(new_features)

print("\nSample Rows:")
print(train_fe_df[new_features].head())

# ============================================================
# GRAPH 1: Correlation Heatmap
# ============================================================
plt.figure(figsize=(8,6), dpi=300)
corr = train_fe_df[new_features].corr()
plt.imshow(corr, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(new_features)), new_features, rotation=45)
plt.yticks(range(len(new_features)), new_features)
plt.title("Feature Engineering Correlation Heatmap")

plt.tight_layout()

# SAVE FIRST
save_path1 = os.path.join(fig_dir, "feature_correlation_heatmap.png")
plt.savefig(save_path1, dpi=300)

print("Graph saved at:", save_path1)

# SHOW AFTER SAVE
plt.show()
plt.close()


# ============================================================
# GRAPH 2: Distribution Graph
# ============================================================
plt.figure(figsize=(10,8), dpi=300)
train_fe_df[new_features].hist(figsize=(10,8))
plt.suptitle("Distribution of New Engineered Features")

plt.tight_layout()

# SAVE FIRST
save_path2 = os.path.join(fig_dir, "feature_distribution.png")
plt.savefig(save_path2, dpi=300)

print("Graph saved at:", save_path2)

# SHOW AFTER SAVE
plt.show()
plt.close()


# -----------------------------
# Save file
# -----------------------------
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

train_file = os.path.join(save_dir, "train_feature_engineered.csv")
train_fe_df.to_csv(train_file, index=False)

print("Train feature-engineered data saved at:", train_file)
print("Shape:", train_fe_df.shape)
