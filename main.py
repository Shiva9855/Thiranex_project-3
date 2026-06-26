import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("house_prices.csv")

print("="*60)
print("HOUSE PRICES DATASET")
print("="*60)

# ==========================
# BASIC INFORMATION
# ==========================

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInformation:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# ==========================
# DATA CLEANING
# ==========================

# Convert Price column into numeric

df["Price (in rupees)"] = (
    df["Price (in rupees)"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Price (in rupees)"] = pd.to_numeric(
    df["Price (in rupees)"],
    errors="coerce"
)

# Fill missing values

for col in df.columns:

    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

    else:
        df[col] = df[col].fillna(df[col].mode()[0])
print("\nData Cleaning Completed!")

# ==========================
# STATISTICAL SUMMARY
# ==========================

print("\nStatistical Summary")
print(df.describe())

# ==========================
# PRICE DISTRIBUTION
# ==========================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Price (in rupees)"],
    bins=30,
    kde=True
)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# ==========================
# TOP LOCATIONS
# ==========================

plt.figure(figsize=(12,6))

df["location"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Locations")
plt.ylabel("Number of Houses")
plt.xticks(rotation=45)

plt.show()

# ==========================
# FURNISHING
# ==========================

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x="Furnishing",
              order=df["Furnishing"].value_counts().index)

plt.xticks(rotation=30)

plt.title("Furnishing Type")

plt.show()

# ==========================
# STATUS
# ==========================

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x="Status",
              order=df["Status"].value_counts().index)

plt.xticks(rotation=30)

plt.title("House Status")

plt.show()

# ==========================
# OWNERSHIP
# ==========================

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x="Ownership",
              order=df["Ownership"].value_counts().index)

plt.xticks(rotation=30)

plt.title("Ownership Distribution")

plt.show()

# ==========================
# BATHROOMS
# ==========================

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x="Bathroom")

plt.title("Bathrooms Distribution")

plt.show()

# ==========================
# BALCONY
# ==========================

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x="Balcony")

plt.title("Balcony Distribution")

plt.show()

# ==========================
# CAR PARKING
# ==========================

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x="Car Parking")

plt.title("Car Parking")

plt.show()

# ==========================
# FLOOR ANALYSIS
# ==========================

plt.figure(figsize=(10,5))

df["Floor"].value_counts().head(10).plot(kind="bar")

plt.title("Top Floor Types")

plt.xticks(rotation=45)

plt.show()

# ==========================
# PRICE BOXPLOT
# ==========================

plt.figure(figsize=(8,5))

sns.boxplot(x=df["Price (in rupees)"])

plt.title("Price Outliers")

plt.show()

# ==========================
# NUMERICAL CORRELATION
# ==========================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ==========================
# MOST EXPENSIVE HOUSES
# ==========================

print("\nTop 10 Expensive Houses")

print(
    df[
        ["Title",
         "location",
         "Price (in rupees)"]
    ]
    .sort_values(
        by="Price (in rupees)",
        ascending=False
    )
    .head(10)
)

# ==========================
# LOCATION VS AVERAGE PRICE
# ==========================

avg_price = (
    df.groupby("location")["Price (in rupees)"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

avg_price.plot(kind="bar")

plt.title("Top 10 Costliest Locations")

plt.ylabel("Average Price")

plt.xticks(rotation=45)

plt.show()

# ==========================
# FINAL INSIGHTS
# ==========================

print("\n" + "="*60)
print("FINAL INSIGHTS")
print("="*60)

print(f"Total Houses : {len(df)}")

print(f"Unique Locations : {df['location'].nunique()}")

print(f"Average Price : {df['Price (in rupees)'].mean():,.2f}")

print(f"Maximum Price : {df['Price (in rupees)'].max():,.2f}")

print(f"Minimum Price : {df['Price (in rupees)'].min():,.2f}")

print("\nTop Location:")

print(df["location"].value_counts().head(5))

print("\nMost Common Furnishing:")

print(df["Furnishing"].value_counts().head())

print("\nMost Common Ownership:")

print(df["Ownership"].value_counts().head())

print("\nEDA COMPLETED SUCCESSFULLY")