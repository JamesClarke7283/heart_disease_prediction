import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import pickle
import csv

base_filename = 'heartdisease_model_dropout'
model_filename = base_filename + '.h5'
scaler_filename = base_filename + '_scaler.pkl'

# 1. Download the data
url = "https://www.openml.org/data/download/1592290/phpgNaXZe"
response = requests.get(url)
content = response.content.decode('utf-8')

# 2. Load the data
arff_data, arff_meta = loadarff(StringIO(content))

# 3. Convert ARFF data to a DataFrame
df = pd.read_csv("dataset.csv")

# 6. Create training and test datasets
X = df.drop('chd', axis=1)
y = df['chd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 7. Normalize the data
scaler = StandardScaler()

# Save the fitted scaler to a file
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Build and train the neural network
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Fit the model and get the training history
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.2, callbacks=[early_stop], verbose=1)

# Get the training and validation metrics for each epoch
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Accuracy
print("Training Accuracy: {:.4f}".format(train_acc[-1]))

# Write the metrics to a CSV file
with open('metrics.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])
    for i in range(len(train_acc)):
        writer.writerow([i+1, train_acc[i], val_acc[i], train_loss[i], val_loss[i]])

# Save the trained model
model.save(model_filename)
# Visualize the results

plt.figure(figsize=(12, 4))
epochs = range(1, len(train_acc) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()