## RBFNet 2D Network Simulation Model, we obtain results from this code.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import time
from rbfnet_funcs import *

start = time.time()

#### Use the p_total dataset first to test
path = os.getcwd()
plane_sd = pd.read_csv(path + "/plane_sd.csv", header=None)
y = plane_sd.to_numpy(dtype='float64')
s, d = plane_sd.shape
factor = [20, 20] 
print("h =", factor[0])
clusters = round((s-1)/factor[0])*round((d-1)/factor[1])
print("clusters =", clusters)
# w = np.random.randn(clusters)

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

dataset = np.concatenate((X,y), axis = 1)
M = 10 # even at M = 10, it takes a lot of time, and quite computationally expensive.
        # I want to figure out a way to optimise the code if possible.
ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
train, test = train_test_split(dataset, test_size=0.1, random_state=1) # Attempt with random selection of points
X_test = np.array([x[:2] for x in test])
y_test = np.array([x[2] for x in test])
for j in ratio:
    print("Training Set Size = ", j)
    rmse = np.zeros(M)
    rel_rmse = np.zeros(M)
    for i in range(M):
        print("M = ", i+1)
        train_set, test_set = train_test_split(train, test_size=1-j, random_state=i+1) # Attempt with random selection of points

        # Order the training set in terms of the index.
        train_set = sorted(train_set, key = lambda x: x[0])

        # Separate tuples for training into RBF
        X_train = np.array([x[:2] for x in train_set])
        y_train = np.array([x[2] for x in train_set])

        ##### Setting up the model #####
        rbfnet = RBFNet2DUG(lr=1e-2, inferStds=True) # k adjusts the number of clusters in the training model.

        # rbfnet.fit(X_train, y_train)
        rbfnet.fit_fixed_clusters(X_train, y_train, s, d, factor) # provide explanation on factor variable
        y_pred = rbfnet.predict(X_test)

        # RMSE
        rmse[i] = math.sqrt(mean_squared_error(y_test, y_pred))
        # Relative RMSE
        rel_rmse[i] = math.sqrt(len(y_test)*mean_squared_error(y_test, y_pred)/sum(i**2 for i in y_test))

    # Print RMSE for each training set ratio.    
    print(rmse)
    print("Mean = ",np.mean(rmse))
    print("Min = ",np.amin(rmse))
    print("Max = ",np.amax(rmse))
    print(rel_rmse)
    print("Mean = ",np.mean(rel_rmse))
    print("Min = ",np.amin(rel_rmse))
    print("Max = ",np.amax(rel_rmse))


end = time.time()
print(end - start)
