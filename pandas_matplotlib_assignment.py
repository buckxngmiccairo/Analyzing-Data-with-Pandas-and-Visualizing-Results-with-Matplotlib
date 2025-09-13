# 📘 Assignment: Analyzing Data with Pandas and Visualizing Results with Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------

try:
    # Load example dataset (Iris)
    df = sns.load_dataset("iris")
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the path.")
except Exception as e:
    print(f"❌ Error: {e}")

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset (drop NAs if any)
df = df.dropna()
print("\n✅ Cleaned dataset shape:", df.shape)

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Grouping by species
grouped = df.groupby("species")["petal_length"].mean()
print("\nAverage Petal Length per Species:")
print(grouped)

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# 1. Line Chart
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal_length"], label="Sepal Length")
plt.plot(df.index, df["petal_length"], label="Petal Length")
plt.title("Line Chart: Sepal vs Petal Length")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart
plt.figure(figsize=(7,5))
grouped.plot(kind="bar", color=["#4c72b0", "#55a868", "#c44e52"])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram
plt.figure(figsize=(7,5))
plt.hist(df["sepal_length"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(7,5))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# -------------------------------
# Findings & Observations
# -------------------------------
print("\n📊 Findings & Observations:")
print("- No missing values found in the Iris dataset.")
print("- Setosa has the smallest petal sizes on average.")
print("- Virginica has the largest petal sizes.")
print("- Sepal length distribution is mostly between 5.0–6.5 cm.")
print("- Scatter plot shows clear clusters of species, indicating strong relationships between sepal and petal length.")
