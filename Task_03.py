# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("Dataset.csv")

# Preprocess the data
# Handling missing values
data = data.dropna()  # Assuming dropping missing values for simplicity
# Encoding categorical variables
label_encoder = LabelEncoder()
data['Cuisines_encoded'] = label_encoder.fit_transform(data['Cuisines'])

# Define features and target variable
X = data[['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes', 'Cuisines_encoded']]
y = data['Cuisines']  # Replace 'Cuisine_Category' with the actual column name for your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a classification algorithm and train the model
classifier = RandomForestClassifier()  # You can choose a different classifier
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Analyze the model's performance across different cuisines
# You can further analyze the results, check confusion matrix, etc.
# Identify challenges or biases, if any



# Analyze the model's performance across different cuisines
unique_cuisines = data['Cuisine_Category'].unique()

for cuisine in unique_cuisines:
    subset_data = data[data['Cuisine_Category'] == cuisine]
    X_subset = subset_data[['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes', 'Cuisines_encoded']]
    y_subset = subset_data['Cuisine_Category']

    y_pred_subset = classifier.predict(X_subset)

    accuracy_subset = accuracy_score(y_subset, y_pred_subset)
    classification_rep_subset = classification_report(y_subset, y_pred_subset)

    print(f"\nPerformance for Cuisine: {cuisine}")
    print(f"Subset Size: {len(subset_data)}")
    print(f"Accuracy: {accuracy_subset}")
    print("Classification Report:")
    print(classification_rep_subset)
    print("--------------------------------------------------")
