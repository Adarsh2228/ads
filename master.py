# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
# Change the path and file name as needed
df = pd.read_csv('your_dataset.csv')  # <-- CHANGE your dataset file name here

# Display first few rows
print(df.head())

# Basic Information about the dataset
print(df.info())

# Basic Statistical Description
print(df.describe())

# Checking Missing Values
print(df.isnull().sum())

# Filling missing values (if needed)
# Example: Fill missing numerical columns with mean
# df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# Example: Fill missing categorical columns with mode
# df['column_name'] = df['column_name'].fillna(df['column_name'].mode()[0])



# Scatter Plot
# Used to visualize relationship between two continuous variables
sns.scatterplot(x='column_x', y='column_y', data=df)  # <-- CHANGE column_x and column_y
plt.title('Scatter Plot')
plt.show()

# Box Plot
# Used to detect outliers and visualize distribution
sns.boxplot(x='column_name', data=df)  # <-- CHANGE column_name
plt.title('Box Plot')
plt.show()

# Histogram
# Used to see the distribution of a single feature
df['column_name'].hist(bins=30)  # <-- CHANGE column_name
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Pair Plot
# Used to visualize relationships between multiple variables
sns.pairplot(df)
plt.title('Pair Plot')
plt.show()

# Heatmap
# Used to visualize correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap - Correlation Matrix')
plt.show()



# Z-Test
# Used to compare sample mean to population mean when population std deviation is known
from statsmodels.stats.weightstats import ztest

# Example: Testing if mean of a column is equal to some value
z_stat, p_val = ztest(df['column_name'], value=expected_mean)  # <-- CHANGE column_name and expected_mean
print(f"Z-Test: Z-statistic = {z_stat}, P-value = {p_val}")

# T-Test (One Sample)
# Used when population standard deviation is NOT known
t_stat, p_val = stats.ttest_1samp(df['column_name'], popmean=expected_mean)  # <-- CHANGE column_name and expected_mean
print(f"T-Test: T-statistic = {t_stat}, P-value = {p_val}")

# T-Test (Two Sample)
# Compare means of two groups
group1 = df[df['group_column'] == 'Group1']['target_column']  # <-- CHANGE group_column and target_column
group2 = df[df['group_column'] == 'Group2']['target_column']  # <-- CHANGE group_column and target_column

t_stat, p_val = stats.ttest_ind(group1, group2)
print(f"Two Sample T-Test: T-statistic = {t_stat}, P-value = {p_val}")

# Chi-Square Test
# Used for categorical data to test independence
from scipy.stats import chi2_contingency

# Example: Chi-square between two categorical columns
contingency_table = pd.crosstab(df['categorical_column1'], df['categorical_column2'])  # <-- CHANGE
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test: Chi2 = {chi2}, P-value = {p}")

# ANOVA Test (Analysis of Variance)
# Used to compare means across more than two groups
from scipy.stats import f_oneway

group1 = df[df['group_column'] == 'Group1']['target_column']  # <-- CHANGE
group2 = df[df['group_column'] == 'Group2']['target_column']  # <-- CHANGE
group3 = df[df['group_column'] == 'Group3']['target_column']  # <-- CHANGE (if third group)

f_stat, p_val = f_oneway(group1, group2, group3)
print(f"ANOVA Test: F-statistic = {f_stat}, P-value = {p_val}")



# Handling Missing Values
# Fill missing numerical values with mean
# df['numerical_column'] = df['numerical_column'].fillna(df['numerical_column'].mean())  # <-- CHANGE

# Fill missing categorical values with mode
# df['categorical_column'] = df['categorical_column'].fillna(df['categorical_column'].mode()[0])  # <-- CHANGE

# Label Encoding
# Converting categorical variables into numbers
le = LabelEncoder()
df['categorical_column'] = le.fit_transform(df['categorical_column'])  # <-- CHANGE

# Train-Test Split
# Splitting dataset into training and testing sets
X = df.drop('target_column', axis=1)  # <-- CHANGE target_column
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE (Synthetic Minority Over-sampling Technique)
# Used when dataset is imbalanced (more samples in one class than others)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts()}")
print(f"After SMOTE: {y_train_smote.value_counts()}")



# Logistic Regression
# Used for binary classification problems
log_reg = LogisticRegression()
log_reg.fit(X_train_smote, y_train_smote)

y_pred_log = log_reg.predict(X_test)

print("Logistic Regression Evaluation:")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")

# Decision Tree Classifier
# Used for both classification and regression; easy to interpret
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_smote, y_train_smote)

y_pred_tree = dtree.predict(X_test)

print("Decision Tree Evaluation:")
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")

# Random Forest Classifier
# Ensemble method - builds multiple decision trees and merges them together
rforest = RandomForestClassifier(random_state=42)
rforest.fit(X_train_smote, y_train_smote)

y_pred_rf = rforest.predict(X_test)

print("Random Forest Evaluation:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
