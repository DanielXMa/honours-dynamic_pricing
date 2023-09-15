import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

def kmeans(X, k):
    """Performs k-means clustering for 1D input
    
    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters
    
    Returns:
        ndarray -- A kx1 array of final cluster centers
    """

    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False

    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)

        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=10, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
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
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std 
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                # print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()
                
                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

                # if epoch % 100 != 0:
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
p_total = pd.read_csv(path + "/p_total.csv", names = ["p_total"])
# new_p_total = p_total[0].values.astype('float64')
y = p_total.to_numpy(dtype='float32')
l = len(p_total)
X = np.linspace(0,l-1,l).reshape(l,1)

# Combine indices from X_train with y_train and then sort based on X_train index
# Note: Random selelection reduces any bias in the dataset
# test_size can vary between 0.1, 0.2 and 0.3.
dataset = np.concatenate((X,y), axis = 1)

### Because both test set and training set is chosen randomly, rmse seems to increase

# M = 20
# ratio = np.array([0.7, 0.8, 0.9])
# for j in ratio:
#     print("Training Set Size = ", j)
#     rmse = np.zeros(M)
#     for i in range(M):
#         print("M = ", i+1)
#         train_set, test_set = train_test_split(dataset, test_size=1-j, random_state=i+1) # Attempt with random selection of points

#         # Order the training and test set.
#         train_set = sorted(train_set, key = lambda x: x[0])
#         test_set = sorted(test_set, key = lambda x: x[0])

#         # Separate tuples for training into RBF
#         X_train = np.array([x[0] for x in train_set]).ravel()
#         y_train = np.array([x[1] for x in train_set]).ravel()
#         X_test = np.array([x[0] for x in test_set]).ravel()
#         y_test = np.array([x[1] for x in test_set]).ravel()

#         # Train Model and Predict
#         rbfnet = RBFNet(lr=1e-2, k=20, inferStds=True)
#         rbfnet.fit(X_train, y_train)
#         y_pred = rbfnet.predict(X_test)

#         # RMSE
#         rmse[i] = math.sqrt(mean_squared_error(y_test, y_pred))
#     # Print RMSE for each training set ratio.    
#     print(rmse)
#     print("Mean = ",np.mean(rmse))
#     print("Min = ",np.amin(rmse))
#     print("Max = ",np.amax(rmse))

## Try for Test set will be fixed , i.e. chosen randomly (i.e. K values)
### randomly select 10% to be our fixed test set for all tests
### Then we choose 70% , 80%, 90% of the rest for training, randomly selected each time
### Then we predict on our fixed test set and then determine RMSE.


M = 50 # number of repetitions/iterations
ratio = np.array([0.7, 0.8, 0.9])
train, test = train_test_split(dataset, test_size=0.1, random_state=1) # Attempt with random selection of points
X_test = np.array([x[0] for x in test]).ravel()
y_test = np.array([x[1] for x in test]).ravel()
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
        X_train = np.array([x[0] for x in train_set]).ravel()
        y_train = np.array([x[1] for x in train_set]).ravel()

        # Train Model and Predict
        rbfnet = RBFNet(lr=1e-2, k=50, inferStds=True) # We can change the number of clusters (k) to 10, 20 or 50

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

# Output of results may vary when M is the same value, but the trend is the same
# Consider also changing the number of clusters


## relative error = 


# # Plot
# plt.figure(figsize=(20,6))
# plt.plot(X, y, '-o', label='true')
# plt.plot(X_test, y_pred, '-o', label='Test Set') #Error
# plt.legend()

# plt.tight_layout()
# plt.show()

