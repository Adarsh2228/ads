# Import necessary libraries
import pandas as pd   # for handling datasets
import numpy as np    # for numerical operations
from sklearn.preprocessing import OrdinalEncoder  # to encode categorical variables
from sklearn.linear_model import LinearRegression  # to build regression model

# Load the dataset
df = pd.read_csv("loan_data_set.csv")  # reading the CSV file into a DataFrame

# Find columns which have missing (null) values
na_variables = [var for var in df.columns if df[var].isnull().mean() > 0]
print("Columns with missing values:", na_variables)

# ----------------- Mean Imputation -----------------
# Fill missing values in 'LoanAmount' with the mean value
df_mean = df.copy()  # create a copy of original data
df_mean['LoanAmount'].fillna(df_mean['LoanAmount'].mean(), inplace=True)
print("After Mean Imputation:\n", df_mean.head())

# ----------------- Median Imputation -----------------
# Fill missing values in 'LoanAmount' with the median value
df_median = df.copy()
df_median['LoanAmount'].fillna(df_median['LoanAmount'].median(), inplace=True)
print("After Median Imputation:\n", df_median.head())

# ----------------- Mode Imputation -----------------
# Fill missing values in 'LoanAmount' with the mode value
df_mode = df.copy()
df_mode['LoanAmount'].fillna(df_mode['LoanAmount'].mode()[0], inplace=True)
print("After Mode Imputation:\n", df_mode.head())

# ----------------- Categorical to Numerical -----------------
# Convert all categorical columns to numeric codes using Ordinal Encoding
df_encoded = df.copy()
oe = OrdinalEncoder()
df_encoded = oe.fit_transform(df_encoded)
print("After Encoding:\n", df_encoded)

# ----------------- Random Sample Imputation -----------------
# Fill missing 'LoanAmount' values with random values from existing 'LoanAmount' data
df_random = df.copy()
# Randomly pick non-null LoanAmount samples equal to the number of missing values
random_sample = df_random['LoanAmount'].dropna().sample(df_random['LoanAmount'].isnull().sum(), random_state=0)
# Replace missing values with random samples
random_sample.index = df_random[df_random['LoanAmount'].isnull()].index
df_random.loc[df_random['LoanAmount'].isnull(), 'LoanAmount'] = random_sample
print("After Random Sample Imputation:\n", df_random.head())

# ----------------- Frequent Category Imputation -----------------
# Fill missing 'Gender' values with the most frequent (mode) value
df_freq = df.copy()
most_freq = df_freq['Gender'].mode()[0]  # find most frequent Gender
df_freq['Gender'].fillna(most_freq, inplace=True)
print("Unique values after Frequent Category Imputation for Gender:", df_freq['Gender'].unique())

# ----------------- Regression Imputation -----------------
# Predict missing LoanAmount values using CoapplicantIncome

# Select only 'CoapplicantIncome' and 'LoanAmount' columns
df_regression = df[['CoapplicantIncome', 'LoanAmount']]

# Create training data (where LoanAmount is NOT null)
train_data = df_regression[df_regression['LoanAmount'].notnull()]

# Create test data (where LoanAmount IS null)
test_data = df_regression[df_regression['LoanAmount'].isnull()]

# Define model
lr = LinearRegression()

# Train model to predict LoanAmount using CoapplicantIncome
lr.fit(train_data[['CoapplicantIncome']], train_data['LoanAmount'])

# Predict missing LoanAmount values
predicted_values = lr.predict(test_data[['CoapplicantIncome']])

# Fill missing LoanAmount values with predicted values
df_regression.loc[df_regression['LoanAmount'].isnull(), 'LoanAmount'] = predicted_values

print("After Regression Imputation:\n", df_regression.head())
