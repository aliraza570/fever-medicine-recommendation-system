import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv("data/raw/enhanced_fever_medicine_recommendation.csv")

print(df.head(10))
print(df.shape)
print(df.info())
print(df.isna().sum())
print(df.duplicated().sum())

# Proper way to fill 
df['Previous_Medication'] = df['Previous_Medication'].fillna(df['Previous_Medication'].mode()[0])
print(df.isnull().sum())

# Numeric columns outliers handle 
numeric_cols = ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Heart_Rate']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("Outliers handled successfully")

# Save before/after copies
df_before = df.copy()

# Handle outliers again (for graph comparison)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)

df_after = df.copy()

# =========================
# Graph save path for report
# =========================
report_fig_dir = os.path.join("reports", "figures")
os.makedirs(report_fig_dir, exist_ok=True)

# =========================
# Professional Bubble Plots (blue only)
# =========================
for col in numeric_cols:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=300)

    # Before Outlier Bubble
    axes[0].scatter(
        df_before[col],
        [1]*len(df_before),
        s=50,
        color='blue',
        alpha=0.5
    )
    axes[0].set_title(f'{col} - Before Outlier Handling (Bubble View)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Density View')
    axes[0].set_yticks([])

    # After Outlier Bubble
    axes[1].scatter(
        df_after[col],
        [1]*len(df_after),
        s=50,
        color='blue',
        alpha=0.5
    )
    axes[1].set_title(f'{col} - After Outlier Handling (Bubble View)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Density View')
    axes[1].set_yticks([])

    plt.tight_layout()

    # Save
    save_path = os.path.join(report_fig_dir, f"{col}_outlier_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

# -----------------------------
# Identify categorical columns
# -----------------------------
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical Columns:", categorical_cols)

# Label Encoding
le = LabelEncoder()

for col in categorical_cols:
    df[col + '_encoded'] = le.fit_transform(df[col])

encoded_cols = [col + '_encoded' for col in categorical_cols]
other_cols = [col for col in df.columns if col not in encoded_cols]
df = df[other_cols + encoded_cols]

print(df.head(10))

# Save path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

output_file = os.path.join(PROCESSED_DIR, "cleaned_medicine_data.csv")
df.to_csv(output_file, index=False)

print("Cleaned data saved at:", output_file)

