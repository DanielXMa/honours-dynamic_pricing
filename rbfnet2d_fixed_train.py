# This code compares the two clustering methods used for RBF Network 2D.

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import time
from rbfnet_funcs import *

#### 2-Dimensional Case
#### x is 2d, c is also now 2d

# start = time.time()

#### Use the p_total dataset first to test
path = os.getcwd()
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

dataset = np.concatenate((X,y), axis = 1)

train, test = train_test_split(dataset, test_size=0.1, random_state=1) # Attempt with random selection of points
X_test = np.array([x[:2] for x in test])
y_test = np.array([x[2] for x in test])

M = 5 # even at M = 10, it takes a lot of time, and quite computationally expensive.
        # I want to figure out a way to optimise the code if possible.

# Initial values for model
factor = [10, 15]
clusters = round((s-1)/factor[0])*round((d-1)/factor[1])
train_size = 0.8
w = np.random.randn(clusters)

print("k = ", clusters)
print('h = ', factor) # In this case, h_i = h_j
print("training ratio =", train_size)

rmse_kmeans = np.zeros(M)
rel_rmse_kmeans = np.zeros(M)
rmse_fixed_clusters = np.zeros(M)
rel_rmse_fixed_clusters = np.zeros(M)

for i in range(M):
    print("M = ", i+1)
    train_set, test_set = train_test_split(train, test_size=1-train_size, random_state=i+1) # Attempt with random selection of points (fixed 80% training ratio)

    # Order the training set in terms of the index.
    train_set = sorted(train_set, key = lambda x: x[0])

    # Separate tuples for training into RBF
    X_train = np.array([x[:2] for x in train_set])
    y_train = np.array([x[2] for x in train_set])

    ##### Setting up both models #####
    # K-Means
    rbfnet1 = RBFNet2DKM(lr=1e-2, k=clusters, inferStds=True, w = w) # k adjusts the number of clusters in the training model.
    rbfnet1.fit(X_train, y_train)
    y_pred_kmeans = rbfnet1.predict(X_test)

    # Manually chosen clusters
    rbfnet2 = RBFNet2DUG(lr=1e-2, inferStds=True, w = w)
    rbfnet2.fit_fixed_clusters(X_train, y_train, s, d, factor) # provide explanation on factor variable
    y_pred_fixed_clusters = rbfnet2.predict(X_test)

    # RMSE
    rmse_kmeans[i] = math.sqrt(mean_squared_error(y_test, y_pred_kmeans))
    rmse_fixed_clusters[i] = math.sqrt(mean_squared_error(y_test, y_pred_fixed_clusters))
    # Relative RMSE
    rel_rmse_kmeans[i] = math.sqrt(len(y_test)*mean_squared_error(y_test, y_pred_kmeans)/sum(i**2 for i in y_test))
    rel_rmse_fixed_clusters[i] = math.sqrt(len(y_test)*mean_squared_error(y_test, y_pred_fixed_clusters)/sum(i**2 for i in y_test))

# Print RMSE for both methods
print("K-Means clustering, k =", clusters)    
print(rmse_kmeans)
print("Mean = ",np.mean(rmse_kmeans))
print("Min = ",np.amin(rmse_kmeans))
print("Max = ",np.amax(rmse_kmeans))
print(rel_rmse_kmeans)
print("Mean = ",np.mean(rel_rmse_kmeans))
print("Min = ",np.amin(rel_rmse_kmeans))
print("Max = ",np.amax(rel_rmse_kmeans))

print("Factor =",factor)    
print(rmse_fixed_clusters)
print("Mean = ",np.mean(rmse_fixed_clusters))
print("Min = ",np.amin(rmse_fixed_clusters))
print("Max = ",np.amax(rmse_fixed_clusters))
print(rel_rmse_fixed_clusters)
print("Mean = ",np.mean(rel_rmse_fixed_clusters))
print("Min = ",np.amin(rel_rmse_fixed_clusters))
print("Max = ",np.amax(rel_rmse_fixed_clusters))

# end = time.time()
# print(end - start) # Want to see how long the code will take to run.