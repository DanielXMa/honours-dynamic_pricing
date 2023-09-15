import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

#### 2-Dimensional Case

#### x is 2d, c is also now 2d

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

def rbf2d(x, c, s):
    return np.exp(-1 / (2 * s**2) * np.linalg.norm(x-c)**2)

def kmeans2(X,k):
    k_means = KMeans(n_clusters = k, random_state=0, n_init="auto").fit(X)
    clusters = k_means.cluster_centers_ ## equivalent to clusters for 1D case
    stds = np.zeros(k) # sets up standard deviation array.
    labels = k_means.labels_ # tells us which cluster each value belongs to.
                             # This is equivalent to ClosestCluster for the 1D version.

    #### Compute the stds given the clusters
    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[labels == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[labels == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[labels == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))
    # print(stds)
    return clusters, stds

class RBFNet(object):
    """Implementation of a Radial Basis Function Network
    Now we attempt for 2D"""

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf2d, inferStds=True):
        self.k = k # number of neurons
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans2(X, self.k)
        else:
            # use a fixed std 
            self.centers, _ = kmeans2(X, self.k)
            dMax = max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training      #Gradient Descent.
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                
                loss = (y[i] - F).flatten() ** 2

                # backward pass
                error = -(y[i] - F).flatten()
                # print(error.shape) # issue is the shape of the error

                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
                # if epoch%10 == 0:
                #     print("Epoch %d:" % (epoch))
                #     print('Loss: {0:.2f}'.format(loss[0]))


    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)

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
clusters = 100
print("k = ", clusters)
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
        rbfnet = RBFNet(lr=1e-2, k=clusters, inferStds=True) # k adjusts the number of clusters in the training model.

        rbfnet.fit(X_train, y_train)
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
