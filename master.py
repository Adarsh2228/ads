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







# Import necessary libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from scipy.stats import pearsonr
import statistics

# -------------------- Classification --------------------
# Load your dataset
# Example: dataset = pd.read_csv('your_dataset.csv')

# X = features dataframe, y = target column
# Example:
# X = dataset[['feature1', 'feature2', 'feature3']]
# y = dataset['target']

# For exam purpose you can just set correct columns here
# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize Decision Tree Classifier
clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)

# Predict the results
y_pred = clf_tree.predict(X_test)

# Print predictions
print("Predictions on Test Data:\n", y_pred)

# Calculate Evaluation Metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy}")

# Error Rate
error_rate = (fp + fn) / (tp + tn + fp + fn)
print(f"Error Rate: {error_rate}")

# Precision
precision = tp / (tp + fp)
print(f"Precision: {precision}")

# Sensitivity (Recall)
sensitivity = tp / (tp + fn)
print(f"Sensitivity (Recall): {sensitivity}")

# Specificity
specificity = tn / (tn + fp)
print(f"Specificity: {specificity}")

# ROC
roc = math.sqrt((sensitivity**2 + specificity**2)) / math.sqrt(2)
print(f"ROC: {roc}")

# Geometric Mean
geometric_mean = math.sqrt(sensitivity * specificity)
print(f"Geometric Mean: {geometric_mean}")

# F1 Score
f1_score = (2 * sensitivity * precision) / (precision + sensitivity)
print(f"F1 Score: {f1_score}")

# False Positive Rate
fpr = 1 - specificity
print(f"False Positive Rate: {fpr}")

# False Negative Rate
fnr = 1 - sensitivity
print(f"False Negative Rate: {fnr}")

# Power (1 - FNR)
power = 1 - fnr
print(f"Power: {power}")

# Plot ROC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, label='Decision Tree')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# ROC AUC Score
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))

# -------------------- Regression --------------------

# Load your dataset for Regression
# Example: df = pd.read_csv('regression_data.csv')

# Choose your columns
# Example:
# df2 = df[['Price', 'Demand']]
# Rename 'Price' and 'Demand' below as needed

df2['naturalLogPrice'] = np.log(df2['Price'])
df2['naturalLogDemand'] = np.log(df2['Demand'])

# Plot regression line
sns.regplot(x="naturalLogPrice", y="naturalLogDemand", data=df2, fit_reg=True)
plt.show()

# Splitting into independent and dependent variables
X = df2[['naturalLogPrice']]
y = df2['naturalLogDemand']

# Fit Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)
print("Predicted Values:\n", y_pred)

# Pearson Correlation
corr, _ = pearsonr(df2['naturalLogPrice'], df2['naturalLogDemand'])
print('Pearson Correlation:', corr)

# Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)
print('Mean Squared Error:', mse)

# Root Mean Squared Error (RMSE)
rmse = math.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Coefficient of Determination (R^2 Score)
y_mean = np.mean(y)
total_variance = np.sum((y - y_mean) ** 2)
residual_variance = np.sum((y - y_pred) ** 2)
r2_score = 1 - (residual_variance / total_variance)
print('Coefficient of Determination (R^2):', r2_score)

# Root Mean Squared Relative Error (RMSRE)
rmsre = math.sqrt(np.mean(((y - y_pred) / y) ** 2))
print('Root Mean Squared Relative Error:', rmsre)

# Mean Absolute Error (MAE)
mae = np.mean(abs(y - y_pred))
print('Mean Absolute Error:', mae)

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(abs((y - y_pred) / y)) * 100
print('Mean Absolute Percentage Error:', mape)

# -------------------- Hypothesis Testing --------------------

# Load dataset for Hypothesis Testing
# Example: df = pd.read_csv('hypothesis_data.csv')

# Taking two random samples
df1 = df.sample(n=28)
df2 = df.sample(n=100)

# Sample Size
nt = len(df1)
nz = len(df2)

# Sample Mean
x_bar_t = df1['Total'].mean()
x_bar_z = df2['Total'].mean()

# Population Mean
meu = df['Total'].mean()

# Standard Deviation
s_t = df1['Total'].std()
s_z = df2['Total'].std()
sigma = df['Total'].std()

# Z-test
print("Z-Test Result")
Z_SCORE = (x_bar_z - meu) / (sigma / math.sqrt(nz))
print("Z-Score:", Z_SCORE)
critical_val_z = 1.65  # for alpha = 0.05

if Z_SCORE > critical_val_z:
    print("Reject Null Hypothesis")
else:
    print("Do NOT Reject Null Hypothesis")

# T-test
print("\nT-Test Result")
T_SCORE = (x_bar_t - meu) / (s_t / math.sqrt(nt))
print("T-Score:", T_SCORE)
critical_val_t = 1.703  # for alpha = 0.05

if T_SCORE > critical_val_t:
    print("Reject Null Hypothesis")
else:
    print("Do NOT Reject Null Hypothesis")

# Two Sample Independent T-Test
men_arr = []
women_arr = []

# Collecting male and female samples
for i in range(29):  # You can change the sample size
    if df['Gender'].iloc[i] == "Female":
        women_arr.append(df['Quantity'].iloc[i])
    else:
        men_arr.append(df['Quantity'].iloc[i])

men_mean = statistics.fmean(men_arr)
women_mean = statistics.fmean(women_arr)
men_std = statistics.stdev(men_arr)
women_std = statistics.stdev(women_arr)
men_len = len(men_arr)
women_len = len(women_arr)

# Calculate T-Statistic
pooled_std = math.sqrt((((men_len - 1) * men_std ** 2) + ((women_len - 1) * women_std ** 2)) / (men_len + women_len - 2))
t_statistic = abs(men_mean - women_mean) / (pooled_std * math.sqrt((1 / men_len) + (1 / women_len)))

print("\nTwo Sample T-Test Result")
print("T-Statistic:", t_statistic)
critical_val_two_sample = 1.703  # for df = 27

if t_statistic > critical_val_two_sample:
    print("Reject Null Hypothesis")
else:
    print("Do NOT Reject Null Hypothesis")

# Two Sample Independent Z-Test
men_arr = []
women_arr = []

for i in range(100):  # Bigger sample
    if df['Gender'].iloc[i] == "Female":
        women_arr.append(df['Quantity'].iloc[i])
    else:
        men_arr.append(df['Quantity'].iloc[i])

men_mean = statistics.fmean(men_arr)
women_mean = statistics.fmean(women_arr)
men_std = statistics.stdev(men_arr)
women_std = statistics.stdev(women_arr)
men_len = len(men_arr)
women_len = len(women_arr)

# Calculate Z-Statistic
z_statistic = abs(men_mean - women_mean) / math.sqrt((men_std ** 2 / men_len) + (women_std ** 2 / women_len))

print("\nTwo Sample Z-Test Result")
print("Z-Statistic:", z_statistic)
z_critical = 1.645

if z_statistic > z_critical:
    print("Reject Null Hypothesis")
else:
    print("Do NOT Reject Null Hypothesis")
  
