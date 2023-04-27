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


def sa_heart():
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
    df['famhist'] = df['famhist'].map({b'1': 1, b'2': 0})
    df['chd'] = df['chd'].map({b'1': 0, b'2': 1}).astype(np.int64)
    return df


def main():
    df = sa_heart()
    print(df.head(50))
    df.to_csv("dataset.csv", index=False)


if __name__ == "__main__":
    main()