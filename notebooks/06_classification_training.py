# ===============================
# Model Training: XGB Classifier on Selected Features
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# -----------------------------
# Create folders if not exist
# -----------------------------
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Load Train-Selected Features
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
num_classes = 3

if target_col not in df.columns:
    raise ValueError("Target column not found")

df["Target_Class"] = pd.qcut(df[target_col], q=num_classes, labels=False)

X = df.drop(columns=[target_col, "Target_Class"])
y = df["Target_Class"]

# -----------------------------
# Scale Features
# -----------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])

# -----------------------------
# XGBoost Model
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
# Cross Validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

print("\nClassifier Model Trained Successfully")
print("Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())

# -----------------------------
# Save Results CSV
# -----------------------------
df_results = pd.DataFrame({
    "Fold": np.arange(1, len(scores)+1),
    "Accuracy": scores
})

for key, value in best_params.items():
    df_results[key] = value

csv_path = "data/processed/training_classifier_results.csv"
df_results.to_csv(csv_path, index=False)
print("Training results saved:", csv_path)

# ==========================================================
# GRAPH 1: BAR PLOT
# ==========================================================
plt.figure(figsize=(7,5), dpi=300)
plt.bar(df_results["Fold"], df_results["Accuracy"])

plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.title("Classifier Model Trained - Cross Validation Accuracy (Bar View)")

save_path1 = "reports/figures/classifier_trained_cv_bar.png"
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Graph saved:", save_path1)

# ==========================================================
# GRAPH 2: LINE PLOT
# ==========================================================
plt.figure(figsize=(7,5), dpi=300)
plt.plot(df_results["Fold"], df_results["Accuracy"], marker='o', label="Fold Accuracy")
plt.axhline(df_results["Accuracy"].mean(), linestyle="--", label="Mean Accuracy")

plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.title("Classifier Model Trained - Cross Validation Accuracy Trend")
plt.legend()

save_path2 = "reports/figures/classifier_trained_cv_line.png"
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Graph saved:", save_path2)

# ==========================================================
# GRAPH 3: ECDF MODERN PLOT WITH DETAIL
# ==========================================================
plt.figure(figsize=(7,5), dpi=300)

# Sort accuracy
sorted_acc = np.sort(scores)
yvals = np.arange(1, len(sorted_acc)+1) / float(len(sorted_acc))

plt.plot(sorted_acc, yvals, marker='o', label="Accuracy ECDF")

# Mean line
mean_acc = scores.mean()
plt.axvline(mean_acc, linestyle="--", linewidth=2, label=f"Mean = {mean_acc:.3f}")

# Best & worst
best_acc = scores.max()
worst_acc = scores.min()

plt.axvline(best_acc, linestyle=":", label=f"Best = {best_acc:.3f}")
plt.axvline(worst_acc, linestyle=":", label=f"Worst = {worst_acc:.3f}")

plt.xlabel("Accuracy Score")
plt.ylabel("Cumulative Probability")
plt.title("Classifier Model Trained - Accuracy ECDF Distribution")
plt.legend()

# Supporting detail text
text_msg = (
    "ECDF shows distribution of accuracy across folds\n"
    "Right shift = better model performance\n"
    "Steep curve = stable model"
)

plt.text(
    0.02, 0.75,
    text_msg,
    transform=plt.gca().transAxes,
    fontsize=9,
    bbox=dict(facecolor='white', alpha=0.6)
)

save_path3 = "reports/figures/classifier_trained_ecdf.png"
plt.savefig(save_path3, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("ECDF graph saved:", save_path3)
