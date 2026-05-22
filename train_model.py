import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the split data
data = joblib.load('train_test_data.pkl')
X_train, X_test, y_train, y_test = data

# Convert to numpy arrays if they aren't already
X_train = np.array(X_train)
y_train = np.array(y_train).ravel()  # Ensure y_train is 1D

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")