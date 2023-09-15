### Attempt at LSTM

import os
import glob
import pandas as pd
import numpy as np
# from lstmnn import SequenceDataset, LSTMForecaster
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

path = os.getcwd()


### Methodology ###
# start with the p_total file for LSTM

# after, consider the surface combinations 

# potentially for the tensor of same route

################################################################
# Also keep in mind of changin the ratio of trainig and test set.
# structure of 


# Another direction, can consider using a another route


## Also consider using other methods, regression


### p_total #####

#read csv file
p_total = pd.read_csv(path + "/p_total.csv", names = ["p_total"])
# new_p_total = p_total[0].values.astype('float64')
y = p_total.to_numpy(dtype='float32')

########################################
# Splitting data into train and test sets just for visualisation purposes to begin. (for p_total)
l = len(p_total)
x = np.linspace(0,l-1,l).reshape(l,1)


## Plots used to visualise how the data is split.
x_train, x_test, y_train, y_test = train_test_split(x, p_total, test_size=0.2, random_state=None,shuffle=False)

# fig = plt.figure()
# plt.plot(x_train, y_train,'blue', x_test, y_test, 'orange')
# plt.show()

# x_train, x_test, y_train, y_test = train_test_split(x, p_total, test_size=0.2, random_state=1,shuffle=True)
# fig = plt.figure()
# plot_train = plt.scatter(x_train, y_train)
# plot_test = plt.scatter(x_test, y_test)
# plt.legend((plot_train, plot_test), ('Training', 'Test'),loc = 'upper right')
# plt.title('Training and Test Sets')
# plt.show()

# Should the training and test sets be randomly chosen? Yes to reduce bias
train_size = len(x_train)

#### Straight 80/20 split

# train,test = train_test_split(y, test_size=0.2, random_state=None,shuffle=False)

#### Doing a straight 80/20 split would run into risk of bias.



##### Doing a random split for LSTM makes it a bit difficult
### Potentially reorder in increasing order
### Or reorder based on time on new time 

dataset = np.concatenate((x,y), axis = 1)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=1) # Attempt with random selection of points

# Order the training and test set.
train_set = sorted(train_set, key = lambda x: x[0])
test_set = sorted(test_set, key = lambda x: x[0])

# Separate tuples for training for LSTM
train = np.array([x[1] for x in train_set]).reshape(train_size, 1)
train_index = np.array([x[0] for x in train_set]).reshape(train_size, 1)
test = np.array([x[1] for x in test_set]).reshape(len(y) - train_size, 1)
test_index = np.array([x[0] for x in test_set]).reshape(len(y) - train_size, 1)

train = np.float32(train)
test = np.float32(test)



# ### Only way at the moment that outputs when having a random seed.

# train, test = train_test_split(y, test_size=0.2, random_state=1) # Attempt with random selection of points

# train = sorted(train, key = lambda x: x[0])
# test = sorted(test, key = lambda x: x[0])


################################################################

################################################################
# Build LSTM Model

# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)
 
lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
 
model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
 
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(p_total) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(p_total) * np.nan
    test_plot[train_size+lookback:len(p_total)] = model(X_test)[:, -1, :]





# # Plot after modelling
# # plt.plot(p_total, c='b')
# test_plot = test_plot[np.logical_not(np.isnan(test_plot))]
# plt.plot(test, c = 'b', label = "Test Set")
# plt.plot(test_plot, c='g', label = "Prediction")
# plt.legend()
# plt.title('Test Set vs. Prediction')
# # plt.plot(p_total)
# # plt.plot(train_plot)
# # plt.plot(test_plot)

# plt.show()