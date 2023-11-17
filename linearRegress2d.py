####### Linear Regression with the D_t dataset

import os
import pandas as pd
import numpy as np
# from lstmnn import SequenceDataset, LSTMForecaster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt


path = os.getcwd()

#read csv file
plane_sd = pd.read_csv(path + "/plane_sd.csv", header=None)
y = plane_sd.to_numpy(dtype='float64')
s, d = plane_sd.shape
# print(s,d) ### s=101, d=151
ls = np.linspace(0,s-1,s)
ld = np.linspace(0,d-1,d)
tuples = []
for i in ls:
    for j in ld:
        tuples.append((i,j))

X = np.array(tuples)
y = np.ravel(y)
y = y.reshape(len(y),1)

model = LinearRegression()

dataset = np.concatenate((X,y), axis = 1)
train, test = train_test_split(dataset, test_size=0.1, random_state=1) # Attempt with random selection of points
X_test = np.array([x[:2] for x in test])
y_test = np.array([x[2] for x in test])
sum_squares = sum(map(lambda i: i * i, y_test))
len_y_test = len(y_test)

M = 5 # Use same values that we had for RBF (M = 5 and 10)
ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
for j in ratio:
    print("Training Set Size = ", j)
    rmse = np.zeros(M)
    rel_rmse = np.zeros(M)
    for i in range(M):
        print("M = ",i+1)
        train_set, test_set = train_test_split(train, test_size=1-j, random_state=i+1) # Attempt with random selection of points

        # Order the training set in terms of the index.
        train_set = sorted(train_set, key = lambda x: x[0])

        # Separate tuples for training into RBF
        X_train = np.array([x[:2] for x in train_set])
        y_train = np.array([x[2] for x in train_set])

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