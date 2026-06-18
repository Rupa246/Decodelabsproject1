import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("employee_data.csv")

print("Original Dataset")
print(df)

# =========================
# 1. HANDLE MISSING VALUES
# =========================

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Experience'].fillna(
    df['Experience'].median(),
    inplace=True
)

df['Salary'].fillna(
    df['Salary'].mean(),
    inplace=True
)

print("\nAfter Handling Missing Values")
print(df)

# =========================
# 2. OUTLIER DETECTION USING IQR
# =========================

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['Salary'] = np.where(
    df['Salary'] > upper,
    upper,
    df['Salary']
)

print("\nAfter Outlier Treatment")
print(df)

# =========================
# 3. FEATURE ENGINEERING
# =========================

# Feature 1
df['Salary_per_Year_Experience'] = (
    df['Salary'] / (df['Experience'] + 1)
)

# Feature 2
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[20,30,40,50,70],
    labels=[
        'Young',
        'Adult',
        'Senior',
        'Veteran'
    ]
)

# Feature 3
df['Performance_Category'] = pd.cut(
    df['PerformanceScore'],
    bins=[0,70,85,100],
    labels=[
        'Average',
        'Good',
        'Excellent'
    ]
)

print("\nFeature Engineered Dataset")
print(df)

# =========================
# 4. VISUALIZATION (Combined)
# =========================
# Create 1 window that holds 2 subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Salary (Left side)
sns.boxplot(x=df['Salary'], ax=axes[0])
axes[0].set_title("Salary Distribution")

# Plot 2: Age (Right side)
sns.histplot(df['Age'], kde=True, ax=axes[1])
axes[1].set_title("Age Distribution")

# Show both together
plt.tight_layout()
plt.show()

# 1. Get the exact folder path where this script is saved
current_folder = os.path.dirname(os.path.abspath(__file__))

# 2. Combine that folder path with the new file name
output_path = os.path.join(current_folder, "cleaned_employee_data.csv")

# 3. Save the final dataset directly into that exact folder
df.to_csv(output_path, index=False)

print(f"\n🎉 Success! File forced to save at: {output_path}")