# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Step 2: Load the supermarket dataset
df = pd.read_csv('supermarket_sales - Sheet1.csv')

# Step 3: Hypothesis Testing (Z-Test and T-Test on 'Total')
sample_t = df.sample(n=28)
sample_z = df.sample(n=100)

# Calculate means and std dev
mean_pop = df['Total'].mean()
std_pop = df['Total'].std()
mean_t = sample_t['Total'].mean()
mean_z = sample_z['Total'].mean()
std_t = sample_t['Total'].std()

# Z-Test
z_score = (mean_z - mean_pop) / (std_pop / math.sqrt(100))
print("Z-Score:", z_score)
if z_score > 1.65:
    print("Reject Null Hypothesis (Z-Test)")
else:
    print("Do Not Reject Null Hypothesis (Z-Test)")

# T-Test
t_score = (mean_t - mean_pop) / (std_t / math.sqrt(28))
print("T-Score:", t_score)
if t_score > 1.703:
    print("Reject Null Hypothesis (T-Test)")
else:
    print("Do Not Reject Null Hypothesis (T-Test)")

# Step 4: Two-Sample Tests (Male vs Female Quantity)
men = df[df['Gender'] == 'Male']['Quantity'].sample(29)
women = df[df['Gender'] == 'Female']['Quantity'].sample(29)

mean_men, mean_women = men.mean(), women.mean()
std_men, std_women = men.std(), women.std()
n_men, n_women = len(men), len(women)

# Independent T-Test
pooled_std = math.sqrt(((n_men-1)*std_men**2 + (n_women-1)*std_women**2) / (n_men+n_women-2))
t_score2 = abs(mean_men - mean_women) / (pooled_std * math.sqrt(1/n_men + 1/n_women))
print("T-Score (2 sample):", t_score2)
if t_score2 > 1.703:
    print("Reject Null Hypothesis (2 Sample T-Test)")
else:
    print("Do Not Reject Null Hypothesis (2 Sample T-Test)")

# Independent Z-Test
pooled_std_z = math.sqrt((std_men**2 / n_men) + (std_women**2 / n_women))
z_score2 = abs(mean_men - mean_women) / pooled_std_z
print("Z-Score (2 sample):", z_score2)
if z_score2 > 1.645:
    print("Reject Null Hypothesis (2 Sample Z-Test)")
else:
    print("Do Not Reject Null Hypothesis (2 Sample Z-Test)")

# Step 5: Decision Tree Classifier
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nDecision Tree Results:")
print("Accuracy:", (tp+tn)/(tp+tn+fp+fn))
print("Error Rate:", (fp+fn)/(tp+tn+fp+fn))
print("Precision:", tp/(tp+fp))
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# ROC and Geometric Mean
roc = math.sqrt(sensitivity**2 + specificity**2) / math.sqrt(2)
gm = math.sqrt(sensitivity * specificity)
print("ROC:", roc)
print("Geometric Mean:", gm)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.show()

# Step 6: Simple Linear Regression (on uploaded file)
# Uncomment below lines if using outside Colab
# uploaded = files.upload()
# df_reg = pd.read_excel(io.BytesIO(uploaded['regdata.xlsx']))

# Here, assuming the file is already loaded
# df_reg = pd.read_excel('regdata.xlsx')

# For now skipping file upload. You can add it here easily.

# Example regression (dummy data for now)
X = np.random.rand(50, 1) * 10
y = 2.5 * X.flatten() + np.random.randn(50) * 2

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# Metrics
mse = np.mean((y - y_pred) ** 2)
rmse = math.sqrt(mse)
mae = np.mean(abs(y - y_pred))
r2 = model.score(X, y)

print("\nLinear Regression Results:")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

