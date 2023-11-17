# This is the code which might not be necessary

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from rbfnet_funcs import *

#### 2-Dimensional Case

#### x is 2d, c is also now 2d

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
train_set, test_set = train_test_split(X, test_size=0.2, random_state=None)
y = np.ravel(y)

train_index = int(0.8*plane_sd.shape[0])+1 # train_index based on supply in this case
X_train = X[0:train_index*d]
X_test = X[train_index*d:]
y_train = y[:train_index*d]
y_test = y[train_index*d:]

clusters = 100
print("k = ", clusters)
rbfnet = RBFNet2DKM(lr=1e-2, k=clusters, inferStds=True)
rbfnet.fit(X_train, y_train)

y_pred = rbfnet.predict(X)

new_y_pred = y_pred.reshape(101,151)
# np.savetxt("rbfnet2d_pred.csv", new_y_pred, delimiter = ',') # To save the dataset if necessary

## plot plane_sd
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
XX, YY = np.meshgrid(ld, ls)
surf1 = ax.plot_surface(XX, YY, plane_sd, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=np.amin(plane_sd) - 1, top=np.amax(plane_sd)) 
ax.view_init(elev = 30, azim = 120)
# Add a color bar which maps values to colors.
fig.colorbar(surf1, shrink=0.5, aspect=5)
plt.figure(figsize=(18,10))
ax.set_title("RBF - Fixed Price", fontsize = 15)
plt.show()


# plot y_pred
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax.plot_surface(XX, YY, new_y_pred, cmap=cm.PRGn,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=np.amin(new_y_pred) - 1, top=np.amax(new_y_pred)) 
ax.view_init(elev = 30, azim = 120)
# Add a color bar which maps values to colors.
fig.colorbar(surf2, shrink=0.5, aspect=5)
plt.figure(figsize=(18,10))
ax.set_title("RBF - Prediction", fontsize = 15)
plt.show()

## attempt to plot both - demand and price
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot_surface(XX, YY, plane_sd, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(XX, YY, new_y_pred, cmap=cm.PRGn,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=np.amin(new_y_pred) - 1, top=np.amax(new_y_pred)) 
ax.view_init(elev = 0, azim = 90)
# Add a color bar which maps values to colors.
# fig.colorbar(surf2, shrink=0.5, aspect=5)
plt.figure(figsize=(18,10))
ax.set_title("RBF - Original vs Prediction (Demand)", fontsize = 15)
plt.show()

## attempt to plot both - supply and price
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot_surface(XX, YY, plane_sd, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(XX, YY, new_y_pred, cmap=cm.PRGn,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=np.amin(new_y_pred) - 1, top=np.amax(new_y_pred)) 
ax.view_init(elev = 0, azim = 180)
# Add a color bar which maps values to colors.
# fig.colorbar(surf2, shrink=0.5, aspect=5)
plt.figure(figsize=(18,10))
ax.set_title("RBF - Original vs Prediction (Supply)", fontsize = 15)
plt.show()

