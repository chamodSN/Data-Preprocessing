import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = pd.read_csv('../data/Titanic-Dataset.csv')

print(df.head())  # View first rows
print(df.info())  # Check data types, missing values
print(df.describe())  # Stats for outliers/noise

#Task 1 - Data Quality Check

# Check missing values (incomplete data)
missing = df.isnull().sum()
print("Missing values:\n", missing)

# Check duplicates (inconsistencies)
duplicates = df.duplicated().sum()
print("Duplicates:", duplicates)

# Visualize outliers (e.g., boxplot for 'Fare')
sns.boxplot(x=df['Fare'])
plt.show()

#Task 2: Data Cleaning

# Fill missing numerical with mean
imputer_num = SimpleImputer(strategy='mean')
df['Age'] = imputer_num.fit_transform(df[['Age']])

# Fill missing categorical with mode (most frequent)
imputer_cat = SimpleImputer(strategy='most_frequent')
#df['Embarked'] = imputer_cat.fit_transform(df[['Embarked']])
df['Embarked'] = imputer_cat.fit_transform(df[['Embarked']]).ravel()

# Check missing values (incomplete data) after Fill missing numerical with mean and missing categorical with mode
missing = df.isnull().sum()
print("Missing values:\n", missing)

# Drop column with too many missing (e.g., >50%)
df.drop('Cabin', axis=1, inplace=True)  # Titanic example

# Or ignore rows with missing (not recommended if many)
# df.dropna(inplace=True)

# Detect outliers using IQR (for 'Fare')
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR))]
print("Outliers:\n", outliers)

# Remove outliers
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Smoothing by binning (equal-frequency, PDF Page 4)
df['Fare_binned'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])  # Discretize into 4 bins

#After handling outliers check wether they are still there
sns.boxplot(x=df['Fare'])
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR))]
print("Outliers:\n", outliers)

#Task 3: Data Transformation and Discretization

#3.1 - Normalization
# Min-Max Normalization (to [0,1])
scaler_minmax = MinMaxScaler()
df['Age_normalized'] = scaler_minmax.fit_transform(df[['Age']])

# Z-Score Normalization
scaler_z = StandardScaler()
df['Fare_zscore'] = scaler_z.fit_transform(df[['Fare']])

# Decimal Scaling (manual, divide by 10^j where j = len(max_value))
max_fare = df['Fare'].max()
j = len(str(int(max_fare)))
df['Fare_decimal'] = df['Fare'] / (10 ** j)

print("After Normalization : \n", df.head())

#3.2 - Discretization/Binning
# Equal-width binning
df['Age_equal_width'] = pd.cut(df['Age'], bins=3, labels=['Young', 'Adult', 'Senior'])

# Equal-depth (frequency) binning
df['Fare_equal_depth'] = pd.qcut(df['Fare'], q=3, labels=['Low', 'Medium', 'High'])

print("After binning : \n", df.head())

#3.3 - Concept Hierarchy

bins = [0, 18, 35, 60, np.inf]
labels = ['Youth', 'Young Adult', 'Adult', 'Senior']
df['Age_Hierarchy'] = pd.cut(df['Age'], bins=bins, labels=labels)

print("After Concept Hierarchy : \n", df.head())

#Task 4: Data Reduction

#4.1 - Dimensionality Reduction

# PCA (on numerical columns only)
numerical_cols = df.select_dtypes(include=[np.number]).columns
pca = PCA(n_components=2)  # Reduce to 2 dims
df_pca = pd.DataFrame(pca.fit_transform(df[numerical_cols]), columns=['PC1', 'PC2'])
print("Explained Variance:", pca.explained_variance_ratio_)

# Feature Selection (drop irrelevant, e.g., 'PassengerId' in Titanic)
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)  # Irrelevant for prediction

print("After Dimensionality Reduction : \n", df.head())

#4.2 - Numerosity Reduction
# Sampling (simple random, 50% of data)
sample_df = df.sample(frac=0.5, random_state=42)  # Without replacement

# Histogram (for visualization/reduction)
df['Fare'].hist(bins=10)
plt.show()

# Clustering (KMeans for reduction, e.g., group similar Ages)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, n_init=10)
df['Age_Cluster'] = kmeans.fit_predict(df[['Age']])

