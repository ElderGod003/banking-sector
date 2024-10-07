import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your data (make sure you adjust the path)
data = pd.read_csv('data/cleaned_data.csv')  # Corrected file path

# Check the column names
print(data.columns)

# Assuming 'y' is the column you want to predict
X = data.drop('y', axis=1)  # Update this based on your actual target column
y = data['y']  # Update this if your target column name is different

# One-Hot Encoding for categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
