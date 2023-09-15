####### Linear Regression with the p_total dataset

import os
import glob
import pandas as pd
import numpy as np
# from lstmnn import SequenceDataset, LSTMForecaster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

path = os.getcwd()

#read csv file
p_total = pd.read_csv(path + "/p_total.csv", names = ["p_total"])
# new_p_total = p_total[0].values.astype('float64')
p_total = p_total.to_numpy(dtype='float32')

l = len(p_total)
x = np.linspace(0,l-1,l).reshape(l,1)


x_train, x_test, y_train, y_test = train_test_split(x, p_total, test_size=0.2, random_state=None,shuffle=False)
fig = plt.figure()
plt.plot(x_train, y_train,'blue', x_test, y_test, 'orange')
plt.show()


model = LinearRegression()

### Linear regression - model_1 - prediction and results

model.fit(np.array(x_train).reshape((-1, 1)), y_train)
y_test_pred = model.predict(x_test.reshape((-1, 1)))
mse = mean_squared_error(y_test_pred, y_test)
print(mse)
rmse = math.sqrt(mse)
print(rmse)
fig = plt.figure()
plt.plot(x_train, y_train,'blue', x_test, y_test, 'orange')
plt.plot(x_test, y_test_pred, 'red')
plt.show()
