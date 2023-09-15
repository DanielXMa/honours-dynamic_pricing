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
print(df_1.shape)
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


### LSTM

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train_scaled = scaler.transform(x_train)
# scaler.fit(y_train)
# y_train_scaled = scaler.transform(y_train)


x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train)

new_x_train = torch.stack([x_train_tensor])
new_y_train = torch.stack([y_train_tensor])

# print(x_train_tensor)
# print(y_train_tensor)
print(new_x_train)
print(new_y_train)


################################################################

# Linear Regression Model



################################################################

# Creating LSTM Model

class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(1,1,self.hidden_size),
                           torch.zeros(1,1,self.hidden_size))
        else:
            self.hidden = hidden
            
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
                                          self.hidden)
        
        predictions = self.linear(lstm_out.view(len(x), -1))
        
        return predictions[-1], self.hidden



model_2 = LSTM(input_size=1, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_2.parameters(), lr=0.001)

# ################################################################

# # LSTM - Training

epochs = 600
model_2.train()
for epoch in range(epochs+1):
    for x,y in zip(new_x_train, new_y_train):
        y_hat, _ = model_2(x, None)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        
    if epoch%100==0:
        print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')


# model_2.eval()
# with torch.no_grad():
#     predictions, _ = model_2(x_train, None)
# #-- Apply inverse transform to undo scaling
# # predictions = scaler.inverse_transform(np.array(predictions.reshape(-1,1)))
# print(predictions)

#######################################################################################################################



########################################################################################################################

# LASSO Regression
# Consider different model / create a newer model


########################################################################################################################
# df_1['Date'] = int(date.strftime("%Y%m%d"))
# print(df_1['Date'])

# li = []
# for filename in filenames:
#     df = pd.read_csv(filename, header=0)
#     # stock = str(filename).split('/')[-1]
#     # df['Stock Name'] = stock.partition('.')[0]
#     # print(df['Date'].dtypes)
    
#     df = df.drop(columns=['Dividends', 'Stock Splits', 'Volume'])
#     # # print(df.dtypes)
#     # df['Date'] = df['Date'].astype('datetime64[ns]')
#     df['Date'] = (df['Date']).to_numpy()
    # df['Date'] = np.array(df['Date'].values)
    # print(df['Date'])
    # print(type(df['Date']))
    # print(df['Date'])

    # x,y = df.iloc[:, 0], df.iloc[:, 1]
    # # print(x)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None,shuffle=False)
    # # print(y_train, y_test)
    # model_1.fit(np.array(x_train).reshape((-1, 1)), y_train)
    # y_test_pred = model_1.predict(x_test)
    # mse = mean_squared_error(y_test_pred, y_test)
    # li.append(mse)
    # print(li)
    # # for i in range(len(y_test_pred)):

# print(df['Date'])
    

    
    

# print(li[1])
# df = pd.concat(li, axis=0, ignore_index=True)
# print(df)