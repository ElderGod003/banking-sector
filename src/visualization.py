import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load your cleaned data
data = pd.read_csv(r'C:\Users\devan\Desktop\niyatiprojects24\banking-sector\data\cleaned_data.csv')  # Adjust the path if necessary

# Check the column names
print("Columns in the dataset:", data.columns)

# Assuming 'y' is the column you want to predict
X = data.drop('y', axis=1)  # Change 'target' to 'y' if that's the name in your CSV
y = data['y']  # Change 'target' to 'y' if that's the name in your CSV

# One-Hot Encoding for categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')

# Ensure the visualization directory exists
if not os.path.exists('visualization'):
    os.makedirs('visualization')

# Save the confusion matrix plot
plt.savefig('visualization/confusion_matrix.png')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred)
print(report)

# Save the classification report to a text file
with open('visualization/classification_report.txt', 'w') as f:
    f.write(report)
