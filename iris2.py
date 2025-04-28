# Import necessary libraries
import pandas as pd  # For handling data in table form (DataFrame)
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.tree import DecisionTreeClassifier  # Machine Learning model
from sklearn.metrics import accuracy_score  # To evaluate model performance

# Load the dataset into a pandas DataFrame
data = pd.read_csv('/mnt/data/Iris.csv')

# Display the first few rows of the dataset to understand the structure
print(data.head())

# Drop the 'Id' column because it is just an identifier, not useful for prediction
data = data.drop('Id', axis=1)

# Separate the features (input variables) and target (output variable)
X = data.drop('Species', axis=1)  # X = all columns except 'Species'
y = data['Species']  # y = only the 'Species' column

# Split the data into training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the labels (species) for the test data
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
