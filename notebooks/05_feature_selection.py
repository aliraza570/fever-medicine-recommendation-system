import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Create report figure folder
# -----------------------------
fig_dir = os.path.join("reports", "figures")
os.makedirs(fig_dir, exist_ok=True)

# -----------------------------
# Load TRAIN data
# -----------------------------
train_df = pd.read_csv("data/processed/train_feature_engineered.csv")

print("Train data loaded successfully")
print("Shape:", train_df.shape)

# -----------------------------
# Target column
# -----------------------------
target_col = "Temperature"

# -----------------------------
# Separate X and y
# -----------------------------
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# -----------------------------
# Numeric features only
# -----------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
print("Numeric features after removing constants:", len(numeric_cols))

# -----------------------------
# Spearman correlation
# -----------------------------
records = []
for col in numeric_cols:
    if X[col].nunique() <= 1:
        continue
    corr, p_val = spearmanr(X[col], y)
    records.append([col, corr, p_val])

corr_df = pd.DataFrame(
    records,
    columns=["Feature", "Spearman_Corr", "P_Value"]
)

# -----------------------------
# Selection threshold
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
X_selected = train_df[selected_features + [target_col]]

# -----------------------------
# OUTPUT
# -----------------------------
print("\nSTRONGLY RELATED FEATURES WITH TARGET (Temperature)\n")

if selected_features:
    print("Selected Feature Names:")
    for f in selected_features:
        print("-", f)
else:
    print("No feature passed the selection criteria")

print("\nSelected Feature Data Shape:")
print(X_selected.shape)

print("\nSelected Features Head:")
print(X_selected.head())

# ==========================================================
# GRAPH 1: CORRELATION HEATMAP
# ==========================================================
if selected_features:
    plt.figure(figsize=(12, 10), dpi=300)
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

    heatmap_path = os.path.join(fig_dir, "selected_features_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    print("Heatmap saved at:", heatmap_path)

    plt.show()
    plt.close()
else:
    print("No features selected → heatmap skipped")

# ==========================================================
# GRAPH 2: BAR GRAPH
# ==========================================================
if selected_features:
    plt.figure(figsize=(8,6), dpi=300)

    sns.barplot(
        data=selected_df,
        x="Spearman_Corr",
        y="Feature",
        color="blue"
    )

    plt.title("Selected Feature Correlation with Temperature")
    plt.xlabel("Spearman Correlation Value")
    plt.ylabel("Features")
    plt.tight_layout()

    bar_path = os.path.join(fig_dir, "selected_features_barplot.png")
    plt.savefig(bar_path, dpi=300)
    print("Bar graph saved at:", bar_path)

    plt.show()
    plt.close()
else:
    print("No features selected → bar graph skipped")

# ==========================================================
# GRAPH 3: RIDGE PLOT (BLUE)
# ==========================================================
if selected_features:

    ridge_df = X_selected.melt(id_vars=target_col, var_name="Feature", value_name="Value")

    g = sns.FacetGrid(
        ridge_df,
        row="Feature",
        hue="Feature",
        aspect=4,
        height=1.2,
        palette=["blue"] * len(selected_features)
    )

    g.map(sns.kdeplot, "Value", fill=True, alpha=0.7)
    g.map(sns.kdeplot, "Value", color="blue", lw=1.5)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.fig.subplots_adjust(hspace=-0.6)

    plt.suptitle("Selected Features Ridge Distribution", fontsize=16, fontweight="bold")

    ridge_path = os.path.join(fig_dir, "selected_features_ridge_plot.png")
    plt.savefig(ridge_path, dpi=300, bbox_inches='tight')
    print("Ridge plot saved at:", ridge_path)

    plt.show()
    plt.close()

else:
    print("No features selected → ridge plot skipped")

# -----------------------------
# Save CSV
# -----------------------------
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

selected_file = os.path.join(save_dir, "train_selected_features_with_target.csv")
X_selected.to_csv(selected_file, index=False)

print("Selected features with target saved at:", selected_file)
