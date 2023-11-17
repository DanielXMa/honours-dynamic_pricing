## RBFNet 2D Network Uniform Grid Clustering, Plots.

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
# factor = [20]
factor = [20,15]
if len(factor) == 1:
    print("h =", factor[0])
    clusters = round((s-1)/factor[0])*round((d-1)/factor[0])
else:
    print("h_i =", factor[0])
    print("h_j =", factor[1])
    clusters = round((s-1)/factor[0])*round((d-1)/factor[1])
print("clusters =", clusters)
w = np.random.randn(clusters)
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

##### Data Processing #####
dataset = np.concatenate((X,y), axis = 1)
train, test = train_test_split(dataset, test_size=0.1, random_state=1)
X_test = np.array([x[:2] for x in test])
y_test = np.array([x[2] for x in test])
train_set, test_set = train_test_split(train, test_size=0.8, random_state=1)
X_train = np.array([x[:2] for x in train_set])
y_train = np.array([x[2] for x in train_set])

##### Setting up the model #####
rbfnet = RBFNet2DUG(lr=1e-2, inferStds=True, w = w) # k adjusts the number of clusters in the training model.

# rbfnet.fit(X_train, y_train)
rbfnet.fit_fixed_clusters(X_train, y_train, s, d, factor) # provide explanation on factor variable
y_pred = rbfnet.predict(X)

# Creating output for graphs #####
new_y_pred = y_pred.reshape(101,151)
error_plot = abs(plane_sd - new_y_pred)



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

# plot error between original and prediction
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax.plot_surface(XX, YY, error_plot, cmap=cm.BrBG,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=np.amin(error_plot) - 0.2, top=np.amax(error_plot) + 0.2) 
ax.view_init(elev = 30, azim = 120)
# Add a color bar which maps values to colors.
fig.colorbar(surf2, shrink=0.5, aspect=5)
plt.figure(figsize=(18,10))
ax.set_title("RBF - Absolute Error", fontsize = 15)
plt.show()


# Calculate RMSE before plotting
RMSE = math.sqrt(mean_squared_error(y, y_pred))
print("RMSE = ", RMSE)

# ## attempt to plot both - demand and price
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# XX, YY = np.meshgrid(ld, ls[:20])
# surf1 = ax.plot_surface(XX, YY, y_test, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# surf2 = ax.plot_surface(XX, YY, y_pred, cmap=cm.PRGn,
#                        linewidth=0, antialiased=False)

# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.axes.set_zlim3d(bottom=np.amin(new_y_pred) - 0.5, top=np.amax(y_pred)+0.5) 
# ax.view_init(elev = 0, azim = 90)
# # Add a color bar which maps values to colors.
# # fig.colorbar(surf2, shrink=0.5, aspect=5)
# plt.figure(figsize=(18,10))
# ax.set_title("RBF - Original vs Prediction (Demand)", fontsize = 15)
# plt.show()

# ## attempt to plot both - supply and price
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf1 = ax.plot_surface(XX, YY, y_test, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# surf2 = ax.plot_surface(XX, YY, y_pred, cmap=cm.PRGn,
#                        linewidth=0, antialiased=False)

# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.axes.set_zlim3d(bottom=np.amin(new_y_pred) - 0.5, top=np.amax(y_pred)+0.5) 
# ax.view_init(elev = 0, azim = 180)
# # Add a color bar which maps values to colors.
# # fig.colorbar(surf2, shrink=0.5, aspect=5)
# plt.figure(figsize=(18,10))
# ax.set_title("RBF - Original vs Prediction (Supply)", fontsize = 15)
# plt.show()

end = time.time()
print(end - start) # 59.3206250667572 seconds for factor 10
