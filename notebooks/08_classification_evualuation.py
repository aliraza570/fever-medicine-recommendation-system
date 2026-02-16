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
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

# create folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

DATA_PATH  = "data/processed/train_selected_features_with_target.csv"
MODEL_PATH = "models/xgb_classifier.pkl"

# load
df = pd.read_csv(DATA_PATH)
print("Data loaded:", df.shape)

# target
target_col = "Temperature"
df["Target_Class"] = pd.qcut(df[target_col], q=3, labels=False)

X = df.drop(columns=[target_col, "Target_Class"])
y = df["Target_Class"]

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# model
model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.11666896408356275,
    max_depth=2,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)
print("Classifier model trained")

# save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# prediction
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")
f1        = f1_score(y_test, y_pred, average="weighted")

print("\nEVALUATION RESULTS")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# =====================================================
# GRAPH 1: CONFUSION MATRIX
# =====================================================
plt.figure(figsize=(6,5), dpi=300)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix\nShows correct vs incorrect predictions of classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("reports/figures/classifier_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# GRAPH 2: FEATURE IMPORTANCE
# =====================================================
plt.figure(figsize=(8,5), dpi=300)
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance\nHigher value = more influence on prediction")
plt.xlabel("Importance Score")
plt.savefig("reports/figures/classifier_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# GRAPH 3: ROC CURVE (WITH DETAIL)
# =====================================================
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

plt.figure(figsize=(7,6), dpi=300)
for i in range(y_test_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")

plt.plot([0,1],[0,1],"--", label="Random Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve\nShows model ability to distinguish between classes\nCloser to top-left = better classifier")
plt.legend()

plt.savefig("reports/figures/classifier_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# GRAPH 4: PRECISION-RECALL (WITH DETAIL)
# =====================================================
plt.figure(figsize=(7,6), dpi=300)
for i in range(y_test_bin.shape[1]):
    precision_c, recall_c, _ = precision_recall_curve(
        y_test_bin[:, i], y_proba[:, i]
    )
    plt.plot(recall_c, precision_c, label=f"Class {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve\nShows balance between precision and recall\nHigher curve = better model performance")
plt.legend()

plt.savefig("reports/figures/classifier_precision_recall.png", dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# GRAPH 5: PREDICTED VS ACTUAL
# =====================================================
plt.figure(figsize=(6,5), dpi=300)
plt.scatter(range(len(y_test)), y_test, label="Actual")
plt.scatter(range(len(y_pred)), y_pred, alpha=0.5, label="Predicted")
plt.legend()
plt.title("Predicted vs Actual\nCloser overlap means better prediction accuracy")
plt.savefig("reports/figures/classifier_predicted_vs_actual.png", dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# GRAPH 6: CLASSIFICATION REPORT HEATMAP
# =====================================================
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).iloc[:-1, :].T

plt.figure(figsize=(8,5), dpi=300)
sns.heatmap(report_df, annot=True, cmap="Blues")
plt.title("Classification Report\nPrecision, Recall and F1 per class")
plt.savefig("reports/figures/classifier_classification_report.png", dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# GRAPH 7: LEARNING CURVE
# =====================================================
train_sizes, train_scores, test_scores = learning_curve(
    model, X_scaled, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1,1.0,5)
)

train_mean = train_scores.mean(axis=1)
test_mean  = test_scores.mean(axis=1)

plt.figure(figsize=(7,5), dpi=300)
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation Score")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve\nShows model learning performance\nIf gap small = good generalization")
plt.legend()
plt.savefig("reports/figures/classifier_learning_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# GRAPH 8: CALIBRATION CURVE (TRENDING INDUSTRY PLOT)
# =====================================================
plt.figure(figsize=(7,6), dpi=300)

for i in range(y_proba.shape[1]):
    prob_true, prob_pred = calibration_curve(
        (y_test == i).astype(int),
        y_proba[:, i],
        n_bins=5
    )
    plt.plot(prob_pred, prob_true, marker='o', label=f"Class {i}")

plt.plot([0,1],[0,1],'--', label="Perfect Calibration")

plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.title(
    "Calibration Curve (Reliability Diagram)\n"
    "Shows how much model probabilities are trustworthy\n"
    "Closer to diagonal = well calibrated model"
)
plt.legend()

plt.savefig("reports/figures/classifier_calibration_curve.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nAll graphs saved inside reports/figures")
