# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier  # This is the decision tree tool
import pandas as pd  # Helps in handling data

# Create a simple dataset
data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Overcast'],
    'Temperature': ['Hot', 'Cold', 'Mild', 'Mild', 'Cold', 'Hot'],
    'PlayOutside': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

# Convert words into numbers (Computers understand numbers, not words)
mapping = {'Sunny': 0, 'Rainy': 1, 'Overcast': 2, 'Hot': 0, 'Cold': 1, 'Mild': 2, 'Yes': 1, 'No': 0}

# Replace words with numbers
df = pd.DataFrame(data)
df.replace(mapping, inplace=True)

# Separate features (Weather, Temperature) and target (PlayOutside)
X = df[['Weather', 'Temperature']]  # Features: What affects the decision
y = df['PlayOutside']  # Target: The decision

# Create the decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)  # Train the model

# Make a prediction
prediction = model.predict([[0, 0]])  # Predict if we should play when it's Sunny and Hot

# Show results
print("Should we play outside? (1 = Yes, 0 = No):", prediction[0])
