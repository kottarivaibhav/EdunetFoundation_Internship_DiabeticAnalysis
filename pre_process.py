import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('health care diabetes.csv')

# Fill missing values with mean
data.fillna(data.mean(), inplace=True)

# Split the data into features and target
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']               # Target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'model/diabetes_model.pkl')
