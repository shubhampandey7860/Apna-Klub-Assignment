import pandas as pd

file_path = "Dataset.csv"
df = pd.read_csv(file_path)

df.ffill(inplace=True)  
numeric_columns = df.select_dtypes(include=["object"]).columns
for col in numeric_columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(0, inplace=True)  # Fill remaining missing values with zeros
    except ValueError:
        pass  # Column cannot be converted to numeric

# Step 3: Remove Duplicates
df.drop_duplicates(inplace=True)

# Step 4: Handle Outliers (using IQR method)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
df_cleaned = df[~outlier_mask]

# Step 5: Normalize Data (capitalize categorical columns)
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    df_cleaned[col] = df_cleaned[col].str.capitalize()

# Step 6: Handle Noisy Data (remove special characters and extra spaces)
df_cleaned.replace(regex=r"[^\w\s]", value="", inplace=True)
df_cleaned.replace(regex=r"\s+", value=" ", inplace=True)

# Save the cleaned dataset
df_cleaned.to_csv("Cleaned_Dataset.csv", index=False)

print("Data cleaning completed. Cleaned dataset saved as 'Cleaned_Dataset.csv'.")
