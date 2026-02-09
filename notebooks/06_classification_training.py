# ===============================
# Model Training: XGB Classifier on Selected Features
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings("ignore")

# -----------------------------
# Load Train-Selected Features with Target
# -----------------------------
train_file = "data/processed/train_selected_features_with_target.csv"

if not os.path.exists(train_file):
    raise FileNotFoundError(f"Train file not found: {train_file}")

df = pd.read_csv(train_file)
print("✅ Train-selected features loaded")
print("Shape:", df.shape)

# -----------------------------
# Define Target Column and Convert to Classes
# -----------------------------
target_col = "Temperature"
num_classes = 3  # Convert continuous target into 3 classes

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in DataFrame. Columns available: {df.columns.tolist()}")

# Convert continuous target to categorical for classification
df["Target_Class"] = pd.qcut(df[target_col], q=num_classes, labels=False)

# Features and target
X = df.drop(columns=[target_col, "Target_Class"])
y = df["Target_Class"]

# -----------------------------
# Scale Numeric Features
# -----------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])
print(f"Numeric features used: {len(numeric_cols)}")

# -----------------------------
# XGB Classifier with Recommended Hyperparameters
# -----------------------------
best_params = {
    "n_estimators": 50,
    "learning_rate": 0.11666896408356275,
    "max_depth": 2,
    "use_label_encoder": False,
    "eval_metric": "mlogloss"
}

clf = XGBClassifier(**best_params)

# -----------------------------
# 5-Fold Stratified Cross Validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

print("\n✅ Model Training Completed")
print("Cross-Validated Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Best Hyperparameters:", best_params)

# -----------------------------
# Save Training Results to CSV
# -----------------------------
# Prepare DataFrame
df_results = pd.DataFrame({
    "Fold": np.arange(1, len(scores)+1),
    "Accuracy": scores
})
# Add hyperparameters as columns
for key, value in best_params.items():
    df_results[key] = value

# Save to processed folder
save_path = "data/processed/training_classifier_results.csv"
df_results.to_csv(save_path, index=False)

print(f"✅ Model training results saved successfully at: {save_path}")
