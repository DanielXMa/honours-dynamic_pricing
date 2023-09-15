import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

# print(os.getcwd())
path = os.getcwd() + "/archive"
filenames = glob.glob(path + "/*.csv")

# print(filenames)
# print(os.scandir(path))
# print(os.listdir(path))

# Setting up the data

file = filenames[0]
df_1 = pd.read_csv(file, header=0)
# print(df_1.shape)
date = df_1['Date'] 
# date = pd.to_datetime(date , format="%Y-%m-%d")
date = pd.to_datetime(date).dt.strftime("%Y%m%d")
# date = date.dt.strftime("%Y%m%d"))
df_1['Date'] = date.to_numpy(dtype=int)
# print(df_1['Date'])
x = df_1['Date']
y = df_1['Open'].to_numpy(dtype=float)
# print(type(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None,shuffle=False)
x_test = x_test.to_numpy(dtype=int)
# y_test = y_test.to_numpy(dtype=float)
# print(type(y_test))

model_1 = LinearRegression()

### Linear regression - model_1 - prediction and results

model_1.fit(np.array(x_train).reshape((-1, 1)), y_train)
y_test_pred = model_1.predict(x_test.reshape((-1, 1)))
mse = mean_squared_error(y_test_pred, y_test)
print(mse)