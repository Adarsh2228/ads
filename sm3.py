S# Import necessary libraries
import pandas as pd  # for data handling
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for advanced plotting

# Load the dataset into a pandas DataFrame
data = pd.read_csv('/mnt/data/supermarket_sales - Sheet1.csv')

# Display the first few rows of the data to understand its structure
print(data.head())

# Check the basic information of the dataset (data types, non-null counts)
print(data.info())

# Get a statistical summary of numeric columns
print(data.describe())

# Check if there are any missing values
print(data.isnull().sum())

# Group data by 'Branch' and calculate total 'Total' sales per branch
branch_sales = data.groupby('Branch')['Total'].sum()

# Display total sales per branch
print(branch_sales)

# Plot the total sales per branch as a bar plot
branch_sales.plot(kind='bar', title='Total Sales by Branch', color='skyblue')
plt.xlabel('Branch')
plt.ylabel('Total Sales')
plt.show()

# Count number of customers by 'Payment' method
payment_counts = data['Payment'].value_counts()

# Display payment method counts
print(payment_counts)

# Plot a pie chart for payment method distribution
payment_counts.plot(kind='pie', autopct='%1.1f%%', title='Payment Methods', figsize=(6,6))
plt.ylabel('')  # Hide y-axis label
plt.show()

# Analyze average rating per product line
product_rating = data.groupby('Product line')['Rating'].mean()

# Display average ratings
print(product_rating)

# Plot average rating by product line
product_rating.plot(kind='barh', title='Average Rating by Product Line', color='green')
plt.xlabel('Average Rating')
plt.show()
