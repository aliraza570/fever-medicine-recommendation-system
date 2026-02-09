# ===============================
# Feature Engineering (TRAIN DATA ONLY)
# ===============================

import pandas as pd
import numpy as np
import os


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

# 1️⃣ Interaction Feature (Health Index)
X_train["Temp_HeartRate_Index"] = X_train["Temperature"] * X_train["Heart_Rate"]

# 2️⃣ Environmental Stress Feature
X_train["Env_Stress_Score"] = (
    X_train["Temperature"] +
    X_train["Humidity"] +
    X_train["AQI"]
)

# 3️⃣ BMI Category Feature
X_train["BMI_Category"] = np.where(
    X_train["BMI"] < 18.5, 0,
    np.where(X_train["BMI"] < 25, 1,
    np.where(X_train["BMI"] < 30, 2, 3))
)

# 4️⃣ Symptom Severity Score
symptom_cols = [
    "Fever_Severity_encoded",
    "Headache_encoded",
    "Body_Ache_encoded",
    "Fatigue_encoded"
]

X_train["Symptom_Severity_Score"] = X_train[symptom_cols].sum(axis=1)

# 5️⃣ Lifestyle Risk Score
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

# -----------------------------
# Results
# -----------------------------
print("\n✅ Feature Engineering Completed")
print("New Train Shape:", train_fe_df.shape)

print("\n🆕 Newly Created Features:")
new_features = [
    "Temp_HeartRate_Index",
    "Env_Stress_Score",
    "BMI_Category",
    "Symptom_Severity_Score",
    "Lifestyle_Risk_Score"
]
print(new_features)

print("\nSample Rows:")
print(train_fe_df[new_features].head())

# Folder ensure karo
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

# File path
train_file = os.path.join(save_dir, "train_feature_engineered.csv")

# Save to CSV
train_fe_df.to_csv(train_file, index=False)

print(f"✅ Train feature-engineered data saved at: {train_file}")
print(f"Shape: {train_fe_df.shape}")
