#RBFNet - 1D
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from rbfnet_funcs import *

#### Use the p_total dataset first to test
path = os.getcwd()
p_total = pd.read_csv(path + "/p_total.csv", names = ["p_total"])
# new_p_total = p_total[0].values.astype('float64')
y = p_total.to_numpy(dtype='float32')
l = len(p_total)
X = np.linspace(0,l-1,l).reshape(l,1)

# Combine indices from X_train with y_train and then sort based on X_train index
# Note: Random selelection reduces any bias in the dataset
# test_size can vary between 0.1, 0.2 and 0.3.
dataset = np.concatenate((X,y), axis = 1)

### Because both test set and training set is chosen randomly, rmse seems to increase

# M = 20
# ratio = np.array([0.7, 0.8, 0.9])
# for j in ratio:
#     print("Training Set Size = ", j)
#     rmse = np.zeros(M)
#     for i in range(M):
#         print("M = ", i+1)
#         train_set, test_set = train_test_split(dataset, test_size=1-j, random_state=i+1) # Attempt with random selection of points

#         # Order the training and test set.
#         train_set = sorted(train_set, key = lambda x: x[0])
#         test_set = sorted(test_set, key = lambda x: x[0])

#         # Separate tuples for training into RBF
#         X_train = np.array([x[0] for x in train_set]).ravel()
#         y_train = np.array([x[1] for x in train_set]).ravel()
#         X_test = np.array([x[0] for x in test_set]).ravel()
#         y_test = np.array([x[1] for x in test_set]).ravel()

#         # Train Model and Predict
#         rbfnet = RBFNet(lr=1e-2, k=20, inferStds=True)
#         rbfnet.fit(X_train, y_train)
#         y_pred = rbfnet.predict(X_test)

#         # RMSE
#         rmse[i] = math.sqrt(mean_squared_error(y_test, y_pred))
#     # Print RMSE for each training set ratio.    
#     print(rmse)
#     print("Mean = ",np.mean(rmse))
#     print("Min = ",np.amin(rmse))
#     print("Max = ",np.amax(rmse))

## Try for Test set will be fixed , i.e. chosen randomly (i.e. K values)
### randomly select 10% to be our fixed test set for all tests
### Then we choose 70% , 80%, 90% of the rest for training, randomly selected each time
### Then we predict on our fixed test set and then determine RMSE.


M = 50 # number of repetitions/iterations
ratio = np.array([0.7, 0.8, 0.9])
epochs = np.array([200,500,1000,2000])
train, test = train_test_split(dataset, test_size=0.1, random_state=1) # Attempt with random selection of points
X_test = np.array([x[0] for x in test]).ravel()
y_test = np.array([x[1] for x in test]).ravel()
for k in epochs:
    print("epochs = ", k)
    for j in ratio:
        print("Training Set Size = ", j)
        rmse = np.zeros(M)
        rel_rmse = np.zeros(M)
        for i in range(M):
            print("M = ", i+1)
            train_set, test_set = train_test_split(train, test_size=1-j, random_state=i+1) # Attempt with random selection of points

            # Order the training set in terms of the index.
            train_set = sorted(train_set, key = lambda x: x[0])

            # Separate tuples for training into RBF
            X_train = np.array([x[0] for x in train_set]).ravel()
            y_train = np.array([x[1] for x in train_set]).ravel()

            # Train Model and Predict
            rbfnet = RBFNet(lr=1e-2, k=20, inferStds=True) # We can change the number of clusters (k) to 10, 20 or 50

            rbfnet.fit(X_train, y_train)
            y_pred = rbfnet.predict(X_test)

            # RMSE
            rmse[i] = math.sqrt(mean_squared_error(y_test, y_pred))
            # Relative RMSE
            rel_rmse[i] = math.sqrt(len(y_test)*mean_squared_error(y_test, y_pred)/sum(i**2 for i in y_test))

        # Print RMSE for each training set ratio.    
        print(rmse)
        print("Mean = ",np.mean(rmse))
        print("Min = ",np.amin(rmse))
        print("Max = ",np.amax(rmse))
        print(rel_rmse)
        print("Mean = ",np.mean(rel_rmse))
        print("Min = ",np.amin(rel_rmse))
        print("Max = ",np.amax(rel_rmse))

# Output of results may vary when M is the same value, but the trend is the same
# Consider also changing the number of clusters


## relative error = 


# # Plot
# plt.figure(figsize=(20,6))
# plt.plot(X, y, '-o', label='true')
# plt.plot(X_test, y_pred, '-o', label='Test Set') #Error
# plt.legend()

# plt.tight_layout()
# plt.show()