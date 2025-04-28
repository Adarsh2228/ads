# Import necessary libraries
import pandas as pd       # For data loading and processing
import numpy as np        # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns     # For advanced graphs
from scipy import stats   # For statistical functions

# Load the dataset into a pandas DataFrame
df = pd.read_csv('supermarket_sales - Sheet1.csv')

# View first few rows of the dataset
print(df.head())

# Basic statistics summary (mean, std dev, min, max, etc.)
print(df.describe())

# Information about dataset (column names, data types, missing values)
print(df.info())

# Find median for all numeric columns
print(df.median())

# Find mode (most frequent value) for selected categorical columns
print(df['Product line'].mode())
print(df['City'].mode())
print(df['Payment'].mode())
print(df['Customer type'].mode())
print(df['Gender'].mode())

# Scatter plot: Tax 5% vs Unit price
plt.scatter(df['Tax 5%'], df['Unit price'], c='blue')
plt.title('Tax vs Unit Price')
plt.xlabel('Tax 5%')
plt.ylabel('Unit Price')
plt.show()

# Scatter plot: Gross income vs Unit price
plt.scatter(df['gross income'], df['Unit price'], c='blue')
plt.title('Gross Income vs Unit Price')
plt.xlabel('Gross Income')
plt.ylabel('Unit Price')
plt.show()

# Scatter plot: Quantity vs Total
plt.scatter(df['Quantity'], df['Total'], c='blue')
plt.title('Quantity vs Total')
plt.xlabel('Quantity')
plt.ylabel('Total')
plt.show()

# Box plot for Tax 5%, Gross Income, and Rating
df[['Tax 5%', 'gross income', 'Rating']].plot(kind='box', title='Boxplot')
plt.show()

# Another box plot specifically for 'Total'
plt.boxplot(df['Total'])
plt.title('Boxplot of Total')
plt.show()

# Calculate 10% Trimmed Mean of 'Total' (removes top/bottom 10% values)
trimmed_mean = stats.trim_mean(df['Total'], 0.1)
print('Trimmed Mean of Total:', trimmed_mean)

# Sum of 'Total' column
print('Sum of Total:', df['Total'].sum())

# Frequency count of each Product Line
print('Frequency of Product Line:\n', df['Product line'].value_counts())

# Variance of all numerical columns
print('Variance:\n', df.var())

# Correlation matrix between numeric columns
print('Correlation Matrix:\n', df.corr())

# Standard Error of Mean for numeric columns
print('Standard Error of Mean:\n', df.sem())

# Manual calculation: Sum of squares for 'Total'
sos = sum(val*val for val in df['Total'])
print('Sum of Squares of Total:', sos)

# Skewness (asymmetry) of numeric columns
print('Skewness:\n', df.skew())

# Kurtosis (peakness) of the 'Total' column
print('Kurtosis of Total:', df['Total'].kurtosis())

# Distribution plot (Density curve) for 'Total'
sns.histplot(df['Total'], kde=True)
plt.title('Distribution of Total')
plt.show()
