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

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
 



### p_total #####

#read csv file
p_total = pd.read_csv(path + "/p_total.csv", names = ["p_total"])
# new_p_total = p_total[0].values.astype('float64')
y = p_total.to_numpy(dtype='float32')

########################################
# Splitting data into train and test sets just for visualisation purposes to begin. (for p_total)
l = len(p_total)
x = np.linspace(0,l-1,l).reshape(l,1)

# Should the training and test sets be randomly chosen? Yes to reduce bias
#### Straight 80/20 split
#### Doing a straight 80/20 split would run into risk of bias.

##### Doing a random split for LSTM makes it a bit difficult
### Potentially reorder in increasing order
### Or reorder based on time on new time 

dataset = np.concatenate((x,y), axis = 1)
lookback = 20 # this value is changeable

# Set up the test set
train, test = train_test_split(dataset, test_size=0.1, random_state=1) # Attempt with random selection of points
test_size = len(test)
test_set = sorted(test, key = lambda x: x[0])
test = np.array([x[1] for x in test_set]).reshape(test_size, 1)
test = np.float32(test)
X_test, y_test = create_dataset(test, lookback=lookback)

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

M = 50 # we fix the number of repetitions in this case
ratio = 0.9 # as it seems to show that the higher the ratio the better the error.
batch_size = 10 # 
print("lookback = ", lookback)
print("batch_size = ", batch_size)
for j in ratio:
    print("Training Set Size = ", j)
    rmse = np.zeros(M)
    rel_rmse = np.zeros(M)
    for i in range(M):
        print("M = ",i+1)
        train_set, test_set = train_test_split(train, test_size=1-j, random_state=i+1) # Attempt with random selection of points
        train_size = len(train_set)
    # Order the training and test set.
        train_set = sorted(train_set, key = lambda x: x[0])
    # Separate tuples for training for LSTM
        new_train = np.array([x[1] for x in train_set]).reshape(train_size, 1)
        new_train = np.float32(new_train)
        X_train, y_train = create_dataset(new_train, lookback=lookback)
        
        loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size) # batch size is changeable
        
        n_epochs = 100
        for epoch in range(n_epochs):
            model.train()
            for X_batch, y_batch in loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # # Validation
            # if epoch % 100 != 0:
            #     continue
            model.eval()
            with torch.no_grad():
                # y_pred = model(X_train)
                # train_rmse = np.sqrt(loss_fn(y_pred, y_train))
                y_pred = model(X_test)
                test_rmse = np.sqrt(loss_fn(y_pred, y_test))
                
        
        with torch.no_grad():
            y_pred = model(X_test)
            rmse[i] = np.sqrt(loss_fn(y_pred, y_test))
            rel_rmse[i] = np.sqrt(len(y_test)*loss_fn(y_pred, y_test)/sum((i[-1])**2 for i in y_test)) 

    # rel_rmse[i] = len(y_pred)*test_rmse**2/sum(i**2 for i in y_pred)

    # rmse[i] = np.sqrt(loss_fn(y_pred, y_test))
    # rel_rmse[i] = np.sqrt(len(y_test)*loss_fn(y_pred, y_test)/sum(i**2 for i in y_test))
    print(rmse)
    print("Mean = ",np.mean(rmse))
    print("Min = ",np.amin(rmse))
    print("Max = ",np.amax(rmse))
    print(rel_rmse)
    print("Mean = ",np.mean(rel_rmse))
    print("Min = ",np.amin(rel_rmse))
    print("Max = ",np.amax(rel_rmse))