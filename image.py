import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("house_prices.csv")

# Check first 5 rows
print(df.head())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Graph 1
sns.histplot(df["Price (in rupees)"], kde=True, ax=axes[0,0])
axes[0,0].set_title("Price Distribution")

# Graph 2
sns.countplot(data=df, x="Furnishing", ax=axes[0,1])
axes[0,1].set_title("Furnishing")

# Graph 3
sns.countplot(data=df, x="Bathroom", ax=axes[0,2])
axes[0,2].set_title("Bathrooms")

# Graph 4
sns.countplot(data=df, x="Balcony", ax=axes[1,0])
axes[1,0].set_title("Balcony")

# Graph 5
sns.boxplot(x=df["Price (in rupees)"], ax=axes[1,1])
axes[1,1].set_title("Price Boxplot")

# Graph 6
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True,
            cmap="coolwarm",
            ax=axes[1,2])
axes[1,2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.show()