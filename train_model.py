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
from tensorflow.keras.layers import Dense

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

# 5. Make the necessary corrections on the data
df['famhist'] = df['famhist'].map({b'Present': 1, b'Absent': 0})
df['chd'] = df['chd'].astype(np.int64)

# 6. Create training and test datasets
X = df.drop('chd', axis=1)
y = df['chd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Build and train the neural network
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, verbose=1)

# 9. Calculate the accuracy and precision of the neural network
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
