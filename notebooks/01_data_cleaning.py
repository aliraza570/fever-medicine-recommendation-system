import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os



df = pd.read_csv("data/raw/enhanced_fever_medicine_recommendation.csv")

print(df.head(10))
# check shape of data 
print(df.shape)
# check info 
print(df.info())
# check null value 
print(df.isna().sum())
#check duplicate value 
print(df.duplicated().sum())

# Proper way to fill 
df['Previous_Medication'] = df['Previous_Medication'].fillna(df['Previous_Medication'].mode()[0])

print(df.isnull().sum())

# Numeric columns  outliers handle 
numeric_cols = ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Heart_Rate']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Lower and Upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Outliers ko cap 
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("Outliers handled successfully")

# Numeric columns jisme outliers handle kiye
numeric_cols = ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Heart_Rate']

# Step 1: Save data before outlier
df_before = df.copy()

# Step 2: Handle outliers (IQR method)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)

# Step 3: Save data after outlier
df_after = df.copy()

# =========================
# Step 4: Professional Graph
# =========================
for col in numeric_cols:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: Before Outlier
    sns.histplot(df_before[col], bins=30, color='blue', kde=True, ax=axes[0])
    axes[0].set_title(f'{col} - Before Outlier Handling', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency')
    
    # Bottom: After Outlier
    sns.histplot(df_after[col], bins=30, color='blue', kde=True, ax=axes[1])
    axes[1].set_title(f'{col} - After Outlier Handling', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# -----------------------------
# Step 1: Identify categorical columns (string type)
# -----------------------------
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical Columns:", categorical_cols)

# -----------------------------
# Step 2: Initialize LabelEncoder
# -----------------------------
le = LabelEncoder()

# -----------------------------
# Step 3: Apply Label Encoding
# -----------------------------
# We'll create new columns with suffix '_encoded' so original is safe
for col in categorical_cols:
    df[col + '_encoded'] = le.fit_transform(df[col])

# -----------------------------
# Step 4: Show encoded columns at the end
# -----------------------------
# Move encoded columns to the end of DataFrame
encoded_cols = [col + '_encoded' for col in categorical_cols]
other_cols = [col for col in df.columns if col not in encoded_cols]
df = df[other_cols + encoded_cols]

# -----------------------------
# Step 5: Display sample (top 10 rows) to verify
# -----------------------------
print(df.head(10))


# Current file ka folder (notebooks)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root (one level up from notebooks)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# data/processed path
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Ensure folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Save CSV
output_file = os.path.join(PROCESSED_DIR, "cleaned_medicine_data.csv")
df.to_csv(output_file, index=False)

print(f"✅ Cleaned data saved at: {output_file}")

