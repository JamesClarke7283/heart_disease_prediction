import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

# 1. Download the data
url = "https://www.openml.org/data/download/1592290/phpgNaXZe"
response = requests.get(url)
content = response.content.decode('utf-8')

# 2. Load the data
arff_data, arff_meta = loadarff(StringIO(content))

# 3. Convert ARFF data to a DataFrame
df = pd.DataFrame(arff_data)

# 4. Update the attributes (columns) with significant names
column_names = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'type', 'obesity', 'alcohol', 'age', 'chd']
df.columns = column_names

print(df.head(50))

# 5. Make the necessary corrections on the data
df['famhist'] = df['famhist'].map({b'1': 1, b'2': 0})
df['chd'] = df['chd'].map({b'1': 0, b'2': 1}).astype(np.int64)

print(df.head(50))

# 6. Create training and test datasets
X = df.drop('chd', axis=1)
y = df['chd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
# 7. Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the fitted scaler to a file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 8. Build and train the neural network
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

# Use Batch Gradient Descent by setting batch_size equal to the number of training samples
batch_size = len(X_train)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_split=0.2, verbose=1, callbacks=[early_stop])

# 9. Calculate the accuracy, precision, recall, and F1 score of the neural network
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print all metrics
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Save the trained model
model.save('heart_disease_model_50epochs.h5')

# Visualise the results
plt.figure(figsize=(12, 4))
epochs = range(1, len(history.history['accuracy']) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Training accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Training loss')
plt.plot(epochs, history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

