import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import random

# Variables
# data: Training data numpy array i x m where the i dimension denotes the parameter and m denotes the training example
#
# y: Training data labels. y is a 1 x m vector.
#
def main():
    x = np.load("X.npy")
    y = np.load("Y.npy")

    dataset = np.vstack((x, y))

    temp_dataset = np.transpose(dataset)
    np.random.shuffle(temp_dataset)
    dataset = np.transpose(temp_dataset)

    x_dev = dataset[:-1, :1004]
    y_dev = dataset[-1, :1004]

    x_test = dataset[:-1, 1004:]
    y_test = dataset[-1, 1004:]

    reg = LinearRegression().fit(np.transpose(x_dev), np.transpose(y_dev))
    print(reg.score(np.transpose(x_dev), np.transpose(y_dev)))

if __name__ == '__main__':
    main()
