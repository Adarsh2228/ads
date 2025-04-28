# --- IMPORTS ---
import pandas as pd
import numpy as np
import math
import statistics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- 1. HYPOTHESIS TESTING ON SUPERMARKET SALES ---

# Load supermarket sales dataset
df = pd.read_csv('/content/drive/My Drive/ADS/supermarket_sales - Sheet1.csv')

# --- Z-test and T-test for Total Amount Hypothesis ---
df_small = df.sample(n=28)  # Sample for T-test
df_large = df.sample(n=100)  # Sample for Z-test

# Calculate mean, std deviation
pop_mean = df['Total'].mean()
pop_std = df['Total'].std()

# Z-test
z_sample_mean = df_large['Total'].mean()
z_score = (z_sample_mean - pop_mean) / (pop_std / math.sqrt(100))
print("\nZ-Test Score:", z_score)
if z_score > 1.65:
    print("Z-Test: Reject Null Hypothesis")
else:
    print("Z-Test: Do NOT Reject Null Hypothesis")

# T-test
t_sample_mean = df_small['Total'].mean()
t_sample_std = df_small['Total'].std()
t_score = (t_sample_mean - pop_mean) / (t_sample_std / math.sqrt(28))
print("\nT-Test Score:", t_score)
if t_score > 1.703:
    print("T-Test: Reject Null Hypothesis")
else:
    print("T-Test: Do NOT Reject Null Hypothesis")

# --- Two-sample Independent T-Test (Gender vs Quantity) ---

# Prepare male and female Quantity samples (first 29 entries)
men = df[df['Gender'] == 'Male']['Quantity'].head(29)
women = df[df['Gender'] == 'Female']['Quantity'].head(29)

men_mean, women_mean = men.mean(), women.mean()
men_std, women_std = men.std(), women.std()

pooled_std = math.sqrt(((len(men)-1)*men_std**2 + (len(women)-1)*women_std**2) / (len(men) + len(women) - 2))
t_score_gender = abs(men_mean - women_mean) / (pooled_std * math.sqrt(1/len(men) + 1/len(women)))

print("\nTwo Sample T-Test Score:", t_score_gender)
if t_score_gender > 1.703:
    print("Two Sample T-Test: Reject Null Hypothesis")
else:
    print("Two Sample T-Test: Do NOT Reject Null Hypothesis")

# --- Two-sample Independent Z-Test (Gender vs Quantity) (First 100 rows) ---

men = df[df['Gender'] == 'Male']['Quantity'].head(100)
women = df[df['Gender'] == 'Female']['Quantity'].head(100)

men_mean, women_mean = men.mean(), women.mean()
men_std, women_std = men.std(), women.std()

z_score_gender = abs(men_mean - women_mean) / math.sqrt(men_std**2/len(men) + women_std**2/len(women))

print("\nTwo Sample Z-Test Score:", z_score_gender)
if z_score_gender > 1.645:
    print("Two Sample Z-Test: Reject Null Hypothesis")
else:
    print("Two Sample Z-Test: Do NOT Reject Null Hypothesis")

# --- 2. CLASSIFICATION TASK: BREAST CANCER DATASET ---

# Load breast cancer dataset
dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation Metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = (tn + tp) / (tn + tp + fn + fp)
error_rate = (fn + fp) / (tn + tp + fn + fp)
precision = tp / (tp + fp)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
roc_val = math.sqrt((sensitivity**2 + specificity**2) / 2)
gm = math.sqrt(sensitivity * specificity)
f1_score = (2 * sensitivity * precision) / (sensitivity + precision)
fpr = 1 - specificity
fnr = 1 - sensitivity
power = 1 - fnr

print("\nClassification Metrics:")
print(f"Accuracy: {accuracy:.2f}, Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}")
print(f"ROC: {roc_val:.2f}, Geometric Mean: {gm:.2f}, F1 Score: {f1_score:.2f}")
print(f"False Positive Rate: {fpr:.2f}, False Negative Rate: {fnr:.2f}, Power: {power:.2f}")

# Plot ROC Curve
fpr_vals, tpr_vals, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8,8))
plt.plot(fpr_vals, tpr_vals, label='Decision Tree')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# --- 3. REGRESSION TASK: DEMAND PREDICTION ---

# Upload and load excel data
import io
from google.colab import files
uploaded = files.upload()
df_reg = pd.read_excel(io.BytesIO(uploaded['regdata.xlsx']))

# Prepare data
df_reg['naturalLogPrice'] = np.log(df_reg['Price'])
df_reg['naturalLogDemand'] = np.log(df_reg['Dem'])

X = df_reg[['naturalLogPrice']]
y = df_reg['naturalLogDemand']

# Train Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot regression
sns.regplot(x='naturalLogPrice', y='naturalLogDemand', data=df_reg, fit_reg=True)
plt.show()

# Evaluation Metrics
corr, _ = pearsonr(df_reg['naturalLogPrice'], df_reg['naturalLogDemand'])
print("\nRegression Metrics:")
print('Pearson Correlation:', round(corr, 3))

mse = np.mean((y - y_pred) ** 2)
rmse = math.sqrt(mse)
R2 = model.score(X, y)
mae = np.mean(abs(y - y_pred))
mape = 100 * np.mean(abs((y - y_pred) / y))

print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")
print(f"R2 (Coefficient of Determination): {R2:.2f}")
print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")
