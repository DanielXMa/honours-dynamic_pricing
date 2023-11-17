# Visualisation of 1D RBF model.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rbfnet_funcs import *

# # sample inputs and add noise
# NUM_SAMPLES = 100
# X = np.random.uniform(0., 1., NUM_SAMPLES)
# X = np.sort(X, axis=0)
# noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
# y = np.sin(2 * np.pi * X)  + noise
# print(X.shape)
# print(y.shape)

#### Use the p_total dataset first to test
path = os.getcwd()
p_total = pd.read_csv(path + "/p_total.csv", names = ["p_total"])
# new_p_total = p_total[0].values.astype('float64')
y = p_total.to_numpy(dtype='float32')
l = len(p_total)
X = np.linspace(0,l-1,l).reshape(l,1)
y = y.ravel()
X = X.ravel()

rbfnet = RBFNet(lr=1e-2, k=20, inferStds=True)
# rbfnet = RBFNet(lr=1e-2, k=2, inferStds=True, epochs = l-1)
rbfnet.fit(X, y)

y_pred = rbfnet.predict(X)

plt.figure(figsize=(20,6))
plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBF-Net')
plt.legend()

plt.tight_layout()
plt.show()
