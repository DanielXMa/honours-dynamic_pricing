####### Linear Regression with the p_total dataset

import os
import glob
import pandas as pd
import numpy as np
# from lstmnn import SequenceDataset, LSTMForecaster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

path = os.getcwd()

#read csv file
p_total = pd.read_csv(path + "/p_total.csv", names = ["p_total"])
# new_p_total = p_total[0].values.astype('float64')
y = p_total.to_numpy(dtype='float32')

l = len(p_total)
x = np.linspace(0,l-1,l).reshape(l,1)


# x_train, x_test, y_train, y_test = train_test_split(x, p_total, test_size=0.2, random_state=None,shuffle=False)
# fig = plt.figure()
# plt.plot(x_train, y_train,'blue', x_test, y_test, 'orange')
# plt.show()

model = LinearRegression()

dataset = np.concatenate((x,y), axis = 1)
train, test = train_test_split(dataset, test_size=0.1, random_state=1) # Attempt with random selection of points
test_size = len(test)
test_set_sorted = sorted(test, key = lambda x: x[0])
X_test = np.array([x[0] for x in test_set_sorted]).reshape(test_size, 1)
y_test = np.array([x[1] for x in test_set_sorted]).reshape(test_size, 1)
sum_squares = sum(map(lambda i: i * i, y_test))[0]
len_y_test = len(y_test)

M = 20
ratio = np.array([0.7, 0.8, 0.9])
for j in ratio:
    print("Training Set Size = ", j)
    rmse = np.zeros(M)
    rel_rmse = np.zeros(M)
    for i in range(M):
        print("M = ",i+1)
        train_set, test_set = train_test_split(train, test_size=1-j, random_state=i+1) # Attempt with random selection of points
        train_size = len(train_set)
    # Order the training and test set.
        train_set_sorted = sorted(train_set, key = lambda x: x[0])
        X_train = np.array([x[0] for x in train_set_sorted]).reshape(train_size, 1)
        y_train = np.array([x[1] for x in train_set_sorted]).reshape(train_size, 1)

        # Train the linear regression model
        model.fit(X_train, y_train)
        # Predict and calculate errors
        y_pred = model.predict(X_test)
        # RMSE
        rmse[i] = math.sqrt(mean_squared_error(y_test, y_pred))
        # Relative RMSE
        rel_rmse[i] = math.sqrt(len_y_test*mean_squared_error(y_test, y_pred)/sum_squares)

    # Print RMSE for each training set ratio.    
    print(rmse)
    print("Mean = ",np.mean(rmse))
    print("Min = ",np.amin(rmse))
    print("Max = ",np.amax(rmse))
    print(rel_rmse)
    print("Mean = ",np.mean(rel_rmse))
    print("Min = ",np.amin(rel_rmse))
    print("Max = ",np.amax(rel_rmse))    
    
X_train_plot = np.ravel(X_train)
y_train_plot = np.ravel(y_train)
X_test_plot = np.ravel(X_test)
y_test_plot = np.ravel(y_test)
y_pred_plot = np.ravel(y_pred)


fig = plt.figure()
plt.scatter(X_train_plot, y_train_plot, label = 'Training Dataset')
plt.scatter(X_test_plot, y_test_plot, c = 'orange', label = 'Test Dataset')
plt.scatter(X_test_plot, y_pred_plot, c = 'r', label = 'Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
plt.legend()
plt.show()
