####### Linear Regression with the D_t dataset to plot the surface

import os
import pandas as pd
import numpy as np
# from lstmnn import SequenceDataset, LSTMForecaster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from matplotlib import cm


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


train_set, test_set = train_test_split(train, test_size=0.2, random_state=1) # Attempt with random selection of points
# Separate tuples for training
X_train = np.array([x[:2] for x in train_set])
y_train = np.array([x[2] for x in train_set])

# Train the linear regression model
model.fit(X_train, y_train)
# Predict and calculate errors
y_pred = model.predict(X)

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
ax.set_title("2D Plane - Fixed Price", fontsize = 15)
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
ax.set_title("Linear Regression - Prediction", fontsize = 15)
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
ax.set_title("Linear Regression - Absolute Error", fontsize = 15)
plt.show()

# # RMSE
# rmse = math.sqrt(mean_squared_error(y_test, y_pred))
# # Relative RMSE
# rel_rmse = math.sqrt(len_y_test*mean_squared_error(y_test, y_pred)/sum_squares)

# # Print RMSE for each training set ratio.    
# print(rmse)

# print(rel_rmse)
