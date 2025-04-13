# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=[col.strip().replace(" ", "_") for col in iris.feature_names])
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# No missing values in Iris dataset, but here's how you'd handle them
# df = df.dropna()  # or use df.fillna(method='ffill') if needed

# Task 2: Basic Data Analysis
print("\nDescriptive statistics:")
print(df.describe())

# Group by species and calculate mean of features
grouped = df.groupby("species").mean()
print("\nAverage values per species:")
print(grouped)

# Task 3: Data Visualization

# Set style for seaborn
sns.set(style="whitegrid")

# 1. Line Chart (not ideal for static features but we simulate one)
plt.figure(figsize=(8, 5))
df.groupby("species").mean().T.plot(kind='line', marker='o')
plt.title("Mean Feature Values per Species")
plt.ylabel("Mean Value")
plt.xlabel("Feature")
plt.grid(True)
plt.legend(title="Species")
plt.tight_layout()
plt.show()

# 2. Bar Chart - Average Petal Length per Species
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="species", y="petal_length", errorbar=None)
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.show()

# 3. Histogram - Sepal Length distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["sepal_length_(cm)"], bins=20, kde=True)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="sepal_length_(cm)", y="petal_length", hue="species")
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
