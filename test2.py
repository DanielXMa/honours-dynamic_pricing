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

x = np.linspace(0,10,101).reshape((-1, 1))

y = np.random.rand(101,1)
y2 = x*x
# print(x.shape)
# print(y.shape)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None,shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.2, random_state=None,shuffle=False)

linreg = LinearRegression()

### Linear regression - model_1 - prediction and results

linreg.fit(np.array(x_train).reshape((-1, 1)), y_train)
y_test_pred = linreg.predict(x_test.reshape((-1, 1)))
mse = mean_squared_error(y_test_pred, y_test)


print(mse)
# print(new_y.shape)
# plt.figure()
plt.plot(x_test,y_test, 'o', label = 'Initial')
plt.plot(x_test,y_test_pred, label = 'Linear Regression')
plt.legend(loc = 'lower left')
plt.show()


###### LSTM
x_tensor = torch.FloatTensor(x)
x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train)

new_x_tensor = torch.stack([x_tensor]) 
new_x_train = torch.stack([x_train_tensor])
new_y_train = torch.stack([y_train_tensor])

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



lstm = LSTM(input_size=1, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)


y_pred_lstm = []
epochs = 200
lstm.train()
for epoch in range(epochs+1):
    for x,y in zip(new_x_train, new_y_train):
        y_hat, _ = lstm(x, None)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        y_pred_lstm.append(y_hat)
        
    if epoch%20==0:
        print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')

print(y_pred_lstm)