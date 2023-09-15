import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
from sklearn.preprocessing import StandardScaler

# print(os.getcwd())
path = os.getcwd() + "/archive"
filenames = glob.glob(path + "/*.csv")

# print(filenames)
# print(os.scandir(path))
# print(os.listdir(path))

# Setting up the data

file = filenames[0]
df_1 = pd.read_csv(file, header=0)
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

########################################################################

# Model Preparation

########################################################################

# Fit scalers
scalers = {}
for x in df_1.columns:
  scalers[x] = StandardScaler().fit(df_1[x].values.reshape(-1, 1))

# Transform data via scalers
norm_df = df_1.copy()
for i, key in enumerate(scalers.keys()):
  norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
  norm_df.iloc[:, i] = norm

#  Defining a function that creates sequences and targets as shown above
def generate_sequences(df: df_1, tw: 5, pw: 5, target_columns, drop_targets=False):
  '''
  df: Pandas DataFrame of the univariate time-series
  tw: Training Window - Integer defining how many steps to look back
  pw: Prediction Window - Integer defining how many steps forward to predict

  returns: dictionary of sequences and targets for all sequences
  '''
  data = dict() # Store results into a dictionary
  L = len(df)
  for i in range(L-tw):
    # Option to drop target from dataframe
    if drop_targets:
      df.drop(target_columns, axis=1, inplace=True)

    # Get current sequence  
    sequence = df[i:i+tw].values
    # Get values right after the current sequence
    target = df[i+tw:i+tw+pw][target_columns].values
    data[i] = {'sequence': sequence, 'target': target}
  return data

class SequenceDataset(Dataset):

  def __init__(self, df):
    self.data = df

  def __getitem__(self, idx):
    sample = self.data[idx]
    return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])
  
  def __len__(self):
    return len(self.data)

# Here we are defining properties for our model

BATCH_SIZE = 16 # Training batch size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None,shuffle=False)
x_test = x_test.to_numpy(dtype=int)

sequences = generate_sequences(norm_df.dcoilwtico.to_frame(), sequence_len, nout, 'dcoilwtico')
dataset = SequenceDataset(sequences)

# Split the data according to our split ratio and load each subset into a
# separate DataLoader object
train_len = int(len(dataset)*split)
lens = [train_len, len(dataset)-train_len]
train_ds, test_ds = random_split(dataset, lens)
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

#Model Architecture
class LSTMForecaster(nn.Module):


  def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False, dropout=0.2):
    '''
    n_features: number of input features (1 for univariate forecasting)
    n_hidden: number of neurons in each hidden layer
    n_outputs: number of outputs to predict for each training example
    n_deep_layers: number of hidden dense layers after the lstm layer
    sequence_len: number of steps to look back at for prediction
    dropout: float (0 < dropout < 1) dropout ratio between dense layers
    '''
    super().__init__()

    self.n_lstm_layers = n_lstm_layers
    self.nhid = n_hidden
    self.use_cuda = use_cuda # set option for device selection

    # LSTM Layer
    self.lstm = nn.LSTM(n_features,
                        n_hidden,
                        num_layers=n_lstm_layers,
                        batch_first=True) # As we have transformed our data in this way
    
    # first dense after lstm
    self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden) 
    # Dropout layer 
    self.dropout = nn.Dropout(p=dropout)

    # Create fully connected layers (n_hidden x n_deep_layers)
    dnn_layers = []
    for i in range(n_deep_layers):
      # Last layer (n_hidden x n_outputs)
      if i == n_deep_layers - 1:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(nhid, n_outputs))
      # All other layers (n_hidden x n_hidden) with dropout option
      else:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(nhid, nhid))
        if dropout:
          dnn_layers.append(nn.Dropout(p=dropout))
    # compile DNN layers
    self.dnn = nn.Sequential(*dnn_layers)

  def forward(self, x):

    # Initialize hidden state
    hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
    cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)

    # move hidden state to device
    if self.use_cuda:
      hidden_state = hidden_state.to(device)
      cell_state = cell_state.to(device)
        
    self.hidden = (hidden_state, cell_state)

    # Forward Pass
    x, h = self.lstm(x, self.hidden) # LSTM
    x = self.dropout(x.contiguous().view(x.shape[0], -1)) # Flatten lstm out 
    x = self.fc1(x) # First Dense
    return self.dnn(x) # Pass forward through fully connected DNN.


nhid = 50 # Number of nodes in the hidden layer
n_dnn_layers = 5 # Number of hidden fully connected layers
nout = 1 # Prediction Window
sequence_len = 180 # Training Window

# Number of features (since this is a univariate timeseries we'll set
# this to 1 -- multivariate analysis is coming in the future)
ninp = 1

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

# Initialize the model
model = LSTMForecaster(ninp, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)


#### Model Training

# Set learning rate and number of epochs to train over
lr = 4e-4
n_epochs = 20

# Initialize the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Lists to store training and validation losses
t_losses, v_losses = [], []
# Loop over epochs
for epoch in range(n_epochs):
  train_loss, valid_loss = 0.0, 0.0

  # train step
  model.train()
  # Loop over train dataset
  for x, y in trainloader:
    optimizer.zero_grad()
    # move inputs to device
    x = x.to(device)
    y  = y.squeeze().to(device)
    # Forward Pass
    preds = model(x).squeeze()
    loss = criterion(preds, y) # compute batch loss
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
  epoch_loss = train_loss / len(trainloader)
  t_losses.append(epoch_loss)
  
  # validation step
  model.eval()
  # Loop over validation dataset
  for x, y in testloader:
    with torch.no_grad():
      x, y = x.to(device), y.squeeze().to(device)
      preds = model(x).squeeze()
      error = criterion(preds, y)
    valid_loss += error.item()
  valid_loss = valid_loss / len(testloader)
  v_losses.append(valid_loss)
      
  print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
plot_losses(t_losses, v_losses)

## Inference

def make_predictions_from_dataloader(model, unshuffled_dataloader):
  model.eval()
  predictions, actuals = [], []
  for x, y in unshuffled_dataloader:
    with torch.no_grad():
      p = model(x)
      predictions.append(p)
      actuals.append(y.squeeze())
  predictions = torch.cat(predictions).numpy()
  actuals = torch.cat(actuals).numpy()
  return predictions.squeeze(), actuals

### Forecasting

# def one_step_forecast(model, history):
#       '''
#       model: PyTorch model object
#       history: a sequence of values representing the latest values of the time 
#       series, requirement -> len(history.shape) == 2
    
#       outputs a single value which is the prediction of the next value in the
#       sequence.
#       '''
#       model.cpu()
#       model.eval()
#       with torch.no_grad():
#         pre = torch.Tensor(history).unsqueeze(0)
#         pred = self.model(pre)
#       return pred.detach().numpy().reshape(-1)

# def n_step_forecast(data: pd.DataFrame, target: str, tw: int, n: int, forecast_from: int=None, plot=False):
#       '''
#       n: integer defining how many steps to forecast
#       forecast_from: integer defining which index to forecast from. None if
#       you want to forecast from the end.
#       plot: True if you want to output a plot of the forecast, False if not.
#       '''
#     history = data[target].copy().to_frame()
      
#       # Create initial sequence input based on where in the series to forecast 
#       # from.
#     if forecast_from:
#         pre = list(history[forecast_from - tw : forecast_from][target].values)
#     else:
#         pre = list(history[self.target])[-tw:]

#       # Call one_step_forecast n times and append prediction to history
#     for i, step in enumerate(range(n)):
#         pre_ = np.array(pre[-tw:]).reshape(-1, 1)
#         forecast = self.one_step_forecast(pre_).squeeze()
#         pre.append(forecast)
      
#       # The rest of this is just to add the forecast to the correct time of 
#       # the history series
#     res = history.copy()
#     ls = [np.nan for i in range(len(history))]

#       # Note: I have not handled the edge case where the start index + n is 
#       # before the end of the dataset and crosses past it.
#     if forecast_from:
#         ls[forecast_from : forecast_from + n] = list(np.array(pre[-n:]))
#         res['forecast'] = ls
#         res.columns = ['actual', 'forecast']
#     else:
#         fc = ls + list(np.array(pre[-n:]))
#         ls = ls + [np.nan for i in range(len(pre[-n:]))]
#         ls[:len(history)] = history[self.target].values
#         res = pd.DataFrame([ls, fc], index=['actual', 'forecast']).T
#     return res
