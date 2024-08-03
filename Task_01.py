# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Dataset.csv")

# Preprocess the data
data = data.dropna()
label_encoder = LabelEncoder()
data['Cuisines_encoded'] = label_encoder.fit_transform(data['Cuisines'])

# Define features and target variable
X = data[['Average Cost for two', 'Price range', 'Votes', 'Cuisines_encoded']]
y = data['Aggregate rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a regression algorithm and train the model
regressor = LinearRegression()  # You can choose a different regressor
regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Analyze the most influential features affecting restaurant ratings
feature_importance = regressor.coef_
feature_names = X.columns

# Plot feature importance
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Most Influential Features affecting Restaurant Ratings')
plt.show()








