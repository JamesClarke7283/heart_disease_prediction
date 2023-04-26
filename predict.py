import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pickle

# Load the saved model
model = load_model('heart_disease_model_50epochs.h5')

# Load and preprocess new data (this should be a DataFrame containing the same features as the training data)
new_data = pd.DataFrame({'sbp': [140], 'tobacco': [0.5], 'ldl': [3.5], 'adiposity': [25], 'famhist': [1], 'type': [60], 'obesity': [28], 'alcohol': [10], 'age': [50]})

# Load the saved scaler from the file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Normalize the new data using the loaded scaler
new_data_normalized = scaler.transform(new_data)

# Make predictions using the loaded model
predictions = model.predict(new_data_normalized)

# Convert predictions to binary class labels (0 or 1)
predicted_labels = (predictions > 0.5).astype(int)

# Map the predicted labels to "Absent" or "Present" strings
predicted_strings = ['Present' if label == 1 else 'Absent' for label in predicted_labels.flatten()]

# Print the predicted strings
print("Predicted strings:", predicted_strings)
