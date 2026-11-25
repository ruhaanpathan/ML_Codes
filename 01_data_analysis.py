# Exploratory Data Analysis using Pandas, Matplotlib and Seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
df["species"] = df["target"].map(dict(enumerate(data.target_names)))

print("First five rows:\n", df.head())
print("\nShape:", df.shape)
print("\nColumn information:")
df.info()
print("\nDescriptive statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nSpecies counts:\n", df["species"].value_counts())

sns.histplot(data=df, x="sepal length (cm)", hue="species", kde=True)
plt.title("Sepal Length Distribution")
plt.show()

sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df, hue="species")
plt.show()
