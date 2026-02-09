# ============================================
# XGB CLASSIFIER – FINAL TRAINING + EVALUATION
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# --------------------------------------------
# Paths
# --------------------------------------------
DATA_PATH  = "data/processed/train_selected_features_with_target.csv"
MODEL_PATH = "models/xgb_classifier.pkl"

os.makedirs("models", exist_ok=True)

# --------------------------------------------
# Load Data (FEATURE SELECTED + TARGET)
# --------------------------------------------
df = pd.read_csv(DATA_PATH)
print("✅ Feature-selected data loaded")
print(df.shape)

# --------------------------------------------
# Target Conversion (Temperature → Classes)
# --------------------------------------------
target_col = "Temperature"

df["Target_Class"] = pd.qcut(df[target_col], q=3, labels=False)

X = df.drop(columns=[target_col, "Target_Class"])
y = df["Target_Class"]

# --------------------------------------------
# Scale Numeric Features
# --------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------
# Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------
# XGB Classifier (Best Params)
# --------------------------------------------
model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.11666896408356275,
    max_depth=2,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

# --------------------------------------------
# Train FINAL MODEL
# --------------------------------------------
model.fit(X_train, y_train)
print("✅ Final XGB model trained")

# --------------------------------------------
# Save Model
# --------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved at {MODEL_PATH}")

# --------------------------------------------
# Predictions
# --------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# --------------------------------------------
# Metrics
# --------------------------------------------
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")
f1        = f1_score(y_test, y_pred, average="weighted")

print("\n✅ EVALUATION RESULTS")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# --------------------------------------------
# Confusion Matrix
# --------------------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------------------------
# Feature Importance
# --------------------------------------------
plt.figure(figsize=(8,5))
plt.barh(X.columns, model.feature_importances_, color="blue")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# --------------------------------------------
# ROC Curve (Multi-Class)
# --------------------------------------------
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

plt.figure(figsize=(7,6))
for i in range(y_test_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="blue", label=f"Class {i} AUC={roc_auc:.2f}")

plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --------------------------------------------
# Precision–Recall Curve
# --------------------------------------------
plt.figure(figsize=(7,6))
for i in range(y_test_bin.shape[1]):
    precision_c, recall_c, _ = precision_recall_curve(
        y_test_bin[:, i], y_proba[:, i]
    )
    plt.plot(recall_c, precision_c, color="blue")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()

# --------------------------------------------
# Predicted vs Actual
# --------------------------------------------
plt.figure(figsize=(6,5))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
plt.scatter(range(len(y_pred)), y_pred, color="blue", alpha=0.5, label="Predicted")
plt.title("Predicted vs Actual")
plt.legend()
plt.show()

# --------------------------------------------
# Save Metrics
# --------------------------------------------
metrics_df = pd.DataFrame({
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1]
})

metrics_df.to_csv(
    "data/processed/xgb_classifier_evaluation.csv",
    index=False
)

print("✅ Evaluation metrics saved")

# ============================================
# SAVE COMPLETE TRAINED MODEL RESULTS TO CSV
# ============================================

# Predictions on TEST data
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Create result DataFrame
results_df = pd.DataFrame({
    "Actual_Class": y_test.values,
    "Predicted_Class": y_pred
})

# Add probabilities
for i in range(y_proba.shape[1]):
    results_df[f"Prob_Class_{i}"] = y_proba[:, i]

# Save CSV
save_path = "data/processed/xgb_classifier_complete_training_results.csv"
results_df.to_csv(save_path, index=False)

print(f"✅ Complete model training results saved at: {save_path}")
