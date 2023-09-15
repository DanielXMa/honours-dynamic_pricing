### Attempt at dynamic pricing

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# print(os.getcwd())
path = os.getcwd()
# filenames = glob.glob(path + "/*.csv")

uber = pd.read_csv(path + "/uber_dataset.csv")
# Note: combined the first two rows of the dataset to get the header row in Excel.

route = uber.iloc[0]
# print(route)

sigma_base = route["Timescale Analysis sigma_base"]
sigma_surge = route["Timescale Analysis sigma_surge"]
weight_base = route["Timescale Analysis weight_base"]
weight_surge = route["Timescale Analysis weight_surge"]
weight_zero = route["Timescale Analysis weight_zero"]

# print(np.random.randint(low=10, high=20, size=(20,1)))

num_samples = 20
# pbase = np.zeros(num_samples)

# psurge = np.zeros(num_samples)
# psurge = np.zeros(1)
pbase = []
psurge = []
ptotal = []
# total = np.zeros(num_samples)
minutes = np.random.randint(low=10,high=20, size=(num_samples,1))

# initialise random seed



for i in range(len(minutes)):
    base = np.random.randint(low=10, high=20, size=(int(minutes[i]+1),1))
    total = np.zeros(int(minutes[i]+1))
    # delta = np.random.normal(low=0, high=1, size=(int(minutes[i]),1))
    delta = np.random.standard_normal() #mean and standard deviation according to paper
    total[0] = np.random.randint(low=15, high=30)
    for j in range(len(delta)):
        total[j+1] = (total[j]/base[j] + delta[j])*base[j+1]
    surge = total[minutes[i]] - base[minutes[i]]
    pbase.append(float(base[minutes[i]]))
    ptotal.append(float(total[minutes[i]]))
    psurge.append(float(surge))

print(pbase)
print(psurge)
print(ptotal)

plt.plot(pbase)
plt.axis([0, 20, 10, 20])
plt.show()

plt.plot(ptotal, label = "total cost")
plt.plot(psurge, label = "surge cost")
# plt.axis([0, 20])
plt.show()

# p_base = 10*np.ones(num_samples)
# p_surge = 10*np.ones(num_samples)


# will need to think about price as function of three variables


# total[0] = 10
# pbase[0] = 5



######################### DIFFERENT SECTION #######################################

#Define the Gaussian function
def gauss(x, x0, w, sigma):
    return w*(1/np.sqrt(2*np.pi*sigma ** 2)) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# randnums= np.random.randint(8,13,200)
# randnums_2 = np.random.uniform(8,13, size = (1,200))

# randbase = np.random.normal(0,sigma_base,num_samples)
# randsurge = np.random.normal(0,sigma_surge,num_samples)


# dp1 = weight_base*randbase + weight_surge*randsurge 
# dp = weight_base*randbase + weight_surge*randsurge + (dp1**2 < 10**(-7))*weight_zero

dp = np.random.rand(num_samples,1)
# print(dp)
# print(dp.shape)


# new_p_base = p_base + dp
# print(new_p_base)
# plt.plot(new_p_base)
# plt.axis([0, 200, 9.9, 10.9])
# plt.show()

# for num in randnums_2:
#     num = round(num,2)

# print(randnums_2)

# total_price = weight_zero + weight_base*





