import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import csv

# Function to create the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

base_filename = 'heartdisease_model_dropout'
model_filename = base_filename + '_best_model.h5'
scaler_filename = base_filename + '_scaler.pkl'

# 3. Convert ARFF data to a DataFrame
df = pd.read_csv("dataset.csv")

# 6. Create training and test datasets
X = df.drop('chd', axis=1)
y = df['chd']

# 7. Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the fitted scaler to a file
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)

# Prepare the KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=500, batch_size=len(X), verbose=0)

# Implement cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_val_acc = 0.0

for train_index, val_index in kfold.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Create the model
    model.model = create_model()

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(model_filename, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=len(X_train),
              callbacks=[checkpoint], verbose=0)

    # Evaluate the model on the validation set
    val_acc = model.score(X_val, y_val)

    # Update the best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc

# Output the best validation accuracy
print("Best Validation Accuracy: {:.4f}".format(best_val_acc))
