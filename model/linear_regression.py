## import packages
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

## load dataset
x = np.load('X_v2.npy')
x = x.reshape(x.shape[0],-1)
print(x.shape)
y = np.load('Y_v2.npy')

## split dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=True, shuffle=True)

# check output shape
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# create linear regression object
regr = linear_model.LinearRegression()

# train the model using the training sets
regr.fit(x_train, y_train)

# make predictions using the testing set
y_pred = regr.predict(x_test)

# print coefficients
print('Coefficients: \n', regr.coef_)
# print mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# print coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# plot outputs
var = 0
plt.scatter(x_test[:,var], y_test,  color='black')
plt.scatter(x_test[:,var], y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
