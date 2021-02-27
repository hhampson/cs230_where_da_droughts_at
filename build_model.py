import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.parameters import *

# Import dataset from saved np arrays
# y.shape == (1, m)
# x.shape == (NUM_VARS, m)
dataset_X = np.load('X.npy')
dataset_Y = np.load('Y.npy')
print(dataset_X.shape)
print(dataset_Y.shape)
print(dataset_Y[0][55:65])

# Splitting train and test data
x_train, x_test, y_train, y_test = train_test_split(dataset_X.T,dataset_Y.T) # random_state controls shuffling

# check output shape
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

n_samples = m
