import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
                if epoch%10 == 0:
                    print("Epoch %d:" % (epoch))
                    print('Loss: {0:.2f}'.format(loss[0]))


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

# randomised train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

##### Setting up the model & prediction #####
rbfnet = RBFNet(lr=1e-2, k=20, inferStds=True)
rbfnet.fit(X_train, y_train)
y_pred = rbfnet.predict(X_test)

# Calcualte RMSE before plotting
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE = ", RMSE)

# # Not sure if plotting is good for a random set.

# # Creating output for graphs #####
# y_train = y_train.reshape(len(y_train),1)
# y_test = y_test.reshape(plane_sd.shape[0] - train_index, plane_sd.shape[1])
# new_y_pred = np.concatenate((y_train, y_pred)).reshape(101,151)
# y_pred = y_pred.reshape(plane_sd.shape[0] - train_index, plane_sd.shape[1]) # More general way of reshaping the prediction

# np.savetxt("rbfnet2d_pred.csv", new_y_pred, delimiter = ',') # To save the dataset if necessary




# ## plot plane_sd
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# XX, YY = np.meshgrid(ld, ls)
# surf1 = ax.plot_surface(XX, YY, plane_sd, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.axes.set_zlim3d(bottom=np.amin(plane_sd) - 1, top=np.amax(plane_sd)) 
# ax.view_init(elev = 30, azim = 120)
# # Add a color bar which maps values to colors.
# fig.colorbar(surf1, shrink=0.5, aspect=5)
# plt.figure(figsize=(18,10))
# ax.set_title("RBF - Fixed Price, 80/20 Train/Test", fontsize = 15)
# plt.show()


# # plot y_pred
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf2 = ax.plot_surface(XX, YY, new_y_pred, cmap=cm.PRGn,
#                        linewidth=0, antialiased=False)
# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.axes.set_zlim3d(bottom=np.amin(new_y_pred) - 1, top=np.amax(new_y_pred)) 
# ax.view_init(elev = 30, azim = 120)
# # Add a color bar which maps values to colors.
# fig.colorbar(surf2, shrink=0.5, aspect=5)
# plt.figure(figsize=(18,10))
# ax.set_title("RBF - Prediction, 80/20 Train/Test", fontsize = 15)
# plt.show()

# ## attempt to plot both - demand and price
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# XX, YY = np.meshgrid(ld, ls[:20])
# surf1 = ax.plot_surface(XX, YY, y_test, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# surf2 = ax.plot_surface(XX, YY, y_pred, cmap=cm.PRGn,
#                        linewidth=0, antialiased=False)

# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.axes.set_zlim3d(bottom=np.amin(y_pred) - 0.5, top=np.amax(y_pred)+0.5) 
# ax.view_init(elev = 0, azim = 90)
# # Add a color bar which maps values to colors.
# # fig.colorbar(surf2, shrink=0.5, aspect=5)
# plt.figure(figsize=(18,10))
# ax.set_title("RBF - Original vs Prediction (Demand), 80/20 Test Set", fontsize = 15)
# plt.show()

# ## attempt to plot both - supply and price
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf1 = ax.plot_surface(XX, YY, y_test, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# surf2 = ax.plot_surface(XX, YY, y_pred, cmap=cm.PRGn,
#                        linewidth=0, antialiased=False)

# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.axes.set_zlim3d(bottom=np.amin(y_pred) - 0.5, top=np.amax(y_pred)+0.5) 
# ax.view_init(elev = 0, azim = 180)
# # Add a color bar which maps values to colors.
# # fig.colorbar(surf2, shrink=0.5, aspect=5)
# plt.figure(figsize=(18,10))
# ax.set_title("RBF - Original vs Prediction (Supply), 80/20 Test Set", fontsize = 15)
# plt.show()