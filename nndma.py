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

#### Attempt at making my own neural network.