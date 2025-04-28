# Import necessary libraries
import pandas as pd  # For data handling
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.preprocessing import LabelEncoder  # For encoding categorical features
from imblearn.over_sampling import SMOTE  # For balancing the dataset
from sklearn.tree import DecisionTreeClassifier  # For classification
from sklearn.metrics import accuracy_score  # For evaluating model performance

# Load the dataset
data = pd.read_csv('/mnt/data/Churn_Modelling.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Drop columns that are not useful for prediction (RowNumber, CustomerId, Surname)
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables ('Geography' and 'Gender') into numeric values
le = LabelEncoder()
data['Geography'] = le.fit_transform(data['Geography'])  # Convert country names to numbers
data['Gender'] = le.fit_transform(data['Gender'])  # Convert 'Male'/'Female' to 1/0

# Separate features (X) and target (y)
X = data.drop('Exited', axis=1)  # Features: all columns except 'Exited'
y = data['Exited']  # Target: 'Exited' column

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model on the resampled training data
model.fit(X_train_resampled, y_train_resampled)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after applying SMOTE: {accuracy:.2f}")
