# Define functions and classes for RBF functions, both 1D and 2D.

import numpy as np
from sklearn.cluster import KMeans

# 1-Dimensional
# Code sourced from https://gamedevacademy.org/using-neural-networks-for-regression-radial-basis-function-networks/

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
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
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

#### 2-Dimensional Case
#### x is 2d, c is also now 2d
#### Everything below has been adjusted for 2D and hence original work

def rbf2d(x, c, s): #Original work
    return np.exp(-1 / (2 * s**2) * np.linalg.norm(x-c)**2)

def kmeans2(X,k): #Original work
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

# Choose own cluster centers, and stds
def fixed_cluster(s, d, factor): #Original work
    ls = np.linspace(0,s-1,s) #in this case, s can range from 0 to 100 (s_max)
    ld = np.linspace(0,d-1,d) #in this case, s can range from 0 to 150 (d_max)
    tuples = []
    if len(factor) == 1:
        h = factor[0]
        c_centre = int(h/2)
        # transformation
        for i in ls: #double check to modulus sign, make all numbers integers
            for j in ld:
                if int(i%h == c_centre) and int(j%h == c_centre):
                    tuples.append([i,j])
        tuples = np.array(tuples)
    elif len(factor) == 2:
        h_i = factor[0]
        h_j = factor[1]
        i_centre = int(h_i/2)
        j_centre = int(h_j/2)
        for i in ls: #double check to modulus sign, make all numbers integers
            for j in ld:
                if int(i%h_i == i_centre) and int(j%h_j == j_centre):
                    tuples.append([i,j])
        tuples = np.array(tuples)
    return tuples

class RBFNet2DKM(object): # Original work
    """Implementation of a Radial Basis Function Network
    Now we attempt for 2D"""

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf2d, inferStds=True, w = None):
        self.k = k # number of neurons
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        # These values are now predetermined outside of the model for 2D
        self.w = w
        self.b = np.random.randn(1)
        if self.w is None:
            self.w = np.random.randn(k)
            

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans2(X, self.k)
        else:
            # use a fixed std 
            self.centers, _ = kmeans2(X, self.k)
            dMax = max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training - Gradient Descent.
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
    
class RBFNet2DUG(object): # Original work
    """Implementation of a Radial Basis Function Network
    Now we attempt for 2D where we have a uniform grid cluster centres"""

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf2d, inferStds=True, w = None):
        self.k = k # number of neurons
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        # These values are now predetermined outside of the model for 2D, if not, then we use original values
        self.w = w
        self.b = np.random.randn(1)

    def fit_fixed_clusters(self, X, y, s, d, factor):
        # use a fixed std 
        self.centers = fixed_cluster(s,d,factor) #double check what underscore represents
        print(len(self.centers)) # as a check to see number of clusters
        if self.k != len(self.centers):
            self.k = len(self.centers)
        self.w = np.random.randn(len(self.centers))
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