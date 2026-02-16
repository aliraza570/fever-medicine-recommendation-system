# ============================================
# Ridge Regressor – COMPLETE EVALUATION (HD)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("reports/figures", exist_ok=True)

# --------------------------------------------
# Load Data
# --------------------------------------------
DATA_PATH = "data/processed/train_selected_features_with_target.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Data loaded successfully")
print(df.shape)

# --------------------------------------------
# Target & Features
# --------------------------------------------
target_col = "Temperature"

X = df.drop(columns=[target_col])
y = df[target_col]

# --------------------------------------------
# Scaling
# --------------------------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])

# --------------------------------------------
# Train Test
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# Model
# --------------------------------------------
model = Ridge(alpha=0.6770314141029844, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------
# Prediction
# --------------------------------------------
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# --------------------------------------------
# Metrics
# --------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\nREGRESSION EVALUATION")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")

# =========================================================
# 1 ACTUAL VS PREDICTED
# =========================================================
plt.figure(figsize=(6,5), dpi=300)
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="black", linestyle="--")

plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted")

plt.text(0.02,0.95,
         "Shows model prediction accuracy.\nCloser to diagonal = better model fit.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 2 RESIDUALS VS PREDICTED
# =========================================================
plt.figure(figsize=(6,5), dpi=300)
plt.scatter(y_pred, residuals, color="blue", alpha=0.6)
plt.axhline(0, color="black", linestyle="--")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")

plt.text(0.02,0.95,
         "Checks model bias.\nRandom scatter around 0 = good model.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_residuals_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 3 RESIDUAL DISTRIBUTION
# =========================================================
plt.figure(figsize=(6,5), dpi=300)
sns.histplot(residuals, kde=True, color="blue")

plt.title("Residual Distribution")
plt.xlabel("Residual")

plt.text(0.02,0.95,
         "Normal distribution of residuals\nindicates stable regression model.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_residual_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 4 LINE PLOT
# =========================================================
plt.figure(figsize=(7,5), dpi=300)
plt.plot(y_test.values, color="blue", label="Actual")
plt.plot(y_pred, color="black", linestyle="--", label="Predicted")

plt.title("Actual vs Predicted Line")
plt.legend()

plt.text(0.02,0.95,
         "Compares actual vs predicted trend.\nCloser overlap = better performance.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_line_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 5 FEATURE COEFFICIENT
# =========================================================
coef_df = pd.DataFrame({
    "Feature": numeric_cols,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient")

plt.figure(figsize=(8,6), dpi=300)
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="blue")

plt.title("Ridge Feature Coefficients")
plt.xlabel("Coefficient Value")

plt.text(0.02,0.95,
         "Shows feature impact on prediction.\nHigher magnitude = stronger influence.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_feature_coefficients.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 6 REGRESSION FIT
# =========================================================
plt.figure(figsize=(6,5), dpi=300)
sns.regplot(x=y_test, y=y_pred, color="blue")

plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Regression Fit Line")

plt.text(0.02,0.95,
         "Shows regression fitting quality.\nStraight line fit = strong model.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_fit_line.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 7 ERROR BOXPLOT
# =========================================================
plt.figure(figsize=(6,5), dpi=300)
sns.boxplot(x=residuals, color="blue")

plt.title("Error Distribution Boxplot")
plt.xlabel("Prediction Error")

plt.text(0.02,0.95,
         "Shows error spread and outliers.\nSmaller box = accurate model.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_error_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 8 HOMOSCEDASTICITY PLOT (NEW)
# =========================================================
plt.figure(figsize=(6,5), dpi=300)
plt.scatter(y_pred, np.sqrt(np.abs(residuals)), color="blue", alpha=0.6)

plt.xlabel("Predicted Values")
plt.ylabel("Sqrt(|Residuals|)")
plt.title("Homoscedasticity Check")

plt.text(0.02,0.95,
         "Checks constant error variance.\nRandom spread = good regression model.",
         transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top')

plt.savefig("reports/figures/reg_homoscedasticity.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# SAVE METRICS
# =========================================================
metrics_df = pd.DataFrame({
    "RMSE": [rmse],
    "MAE": [mae],
    "R2_Score": [r2]
})

save_path = "data/processed/ridge_regression_evaluation.csv"
metrics_df.to_csv(save_path, index=False)
print(f"Evaluation metrics saved at: {save_path}")

# =========================================================
# SAVE PREDICTIONS
# =========================================================
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Residual": residuals
})

pred_save_path = "data/processed/ridge_regression_predictions.csv"
results_df.to_csv(pred_save_path, index=False)
print(f"Predictions saved at: {pred_save_path}")
