### This file generates the datasets for the simulations.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import skewnorm
from price_funcs import price_fixed_time, price_fixed_supply, price_fixed_demand

path = os.getcwd()

uber = pd.read_csv(path + "/uber_dataset.csv")
# Note: combined the first two rows of the dataset to get the header row in Excel.

route = uber.iloc[0]
# print(route)

sigma_base = route["Timescale Analysis sigma_base"]
sigma_surge = route["Timescale Analysis sigma_surge"]
weight_base = route["Timescale Analysis weight_base"]
weight_surge = route["Timescale Analysis weight_surge"]
weight_zero = route["Timescale Analysis weight_zero"]

print(sigma_base, weight_base, sigma_surge, weight_surge, weight_zero)


##### If we wanted to consider another route
# route1 = uber.iloc[2]
# sigma_base1 = route1["Timescale Analysis sigma_base"]
# sigma_surge1 = route1["Timescale Analysis sigma_surge"]
# weight_base1 = route1["Timescale Analysis weight_base"]
# weight_surge1 = route1["Timescale Analysis weight_surge"]
# weight_zero1 = route1["Timescale Analysis weight_zero"]


##### Generate p_base from the formula in the thesis
# p_base = p_0 + p_t delta_t + p_l delta_l
# p_0 can be constant, p_l and delta_l are assumed to be constant
# p_t might be constant and delta_t can vary (between 0.6 and 2) (not yet to be confirmed)

num_samples = 1080
sample = np.ones(num_samples+1)

x = np.linspace(-2,18,num_samples+1)
y = np.linspace(-11,9,num_samples+1)
minutes = ((skewnorm.pdf(x,1) + skewnorm.pdf(y,1))*5)+10

# 
np.random.seed(3)
p_0     = sample*4 + np.random.normal(size = num_samples+1)/2
p_l     = sample*10
delta_l = sample*1
p_t     = sample*2
delta_t = minutes

p_base = p_0 + p_l*delta_l + p_t*delta_t
x1 = np.linspace(0,num_samples, num_samples+1)
print(x1)
plt.figure(figsize=(20,6))
plt.scatter(x1,p_base)
plt.title('p_base', fontsize = 20)
plt.show()



##### generate delta_p using the formula and the weights
np.random.seed(1)
randbase = np.random.normal(0,scale = sigma_base, size = num_samples+1)
plt.figure(figsize=(20,6))
plt.scatter(x1, randbase)
plt.title("randbase", fontsize = 20)
plt.show()

np.random.seed(2)
randsurge = np.random.normal(0,scale = sigma_surge, size = num_samples+1)
plt.figure(figsize=(20,6))
plt.scatter(x1, randsurge)
plt.title("randsurge", fontsize = 20)
plt.show()
# print(randsurge)

print(sigma_surge, sigma_base, weight_surge, weight_base, weight_zero)
s_base  = sample*sigma_base
s_surge = sample*sigma_surge
w_base  = sample*weight_base
w_surge = sample*weight_surge
# w_zero  = sample*weight_zero

delta_p = w_base*randbase + w_surge*randsurge #w_base should be larger than w_surge   

plt.figure(figsize=(20,6))
plt.plot(x1,delta_p)
plt.title("delta_p", fontsize = 20)
plt.show()

for i in range(len(delta_p)):
    if (delta_p[i])**2 < 10**(-14): # change of value of upper bound
        delta_p[i] = delta_p[i] + weight_zero

plt.figure(figsize=(20,6))
plt.plot(x1,delta_p)
plt.title("delta_p + w_zero", fontsize = 20)
plt.show()


##### generate p_surge ############


p_surge = np.zeros(num_samples+1)

p_surge[0] = 1 #for simplicity
for i in range(0,num_samples):
    p_surge[i+1] = ((p_surge[i]/p_base[i]) + delta_p[i])*p_base[i+1]

print(p_surge)
#p_surge[p_surge < 0] = 0 # Potentially to change any negative values to zero, as price should never be negative
print(np.max(p_surge))
########################################################################

plt.figure(figsize=(20,6))
plt.plot(p_surge, label = "surge cost")
plt.title("p_surge", fontsize = 20)
plt.show()

p_total = p_base + p_surge 
plt.figure(figsize=(20,6))
plt.plot(p_total, label = "total cost")
plt.xlabel("Time (mins)", fontsize = 20)
plt.title("p_total", fontsize = 20)
plt.show()


# This is to save the dataset, to be used for simulations.
# np.savez(p_total, x = x1)
# np.savetxt("p_total.csv", p_total, delimiter = ',')

##### create sequence for supply (S) and demand (D)

max_s = 100
max_d = 150

supply = np.linspace(0,max_s,max_s+1) # second variable needs to be changed
demand = np.linspace(0,max_d,max_d+1) # second variable needs to be changed

 ### Potential 0.1 G of space

# 2D Arrays
################################
# Fix time - pick one element from p_base and p_surge
# create grid of supply and demand elements

# Then we get a plane of 100 x 100 elements
pbase = p_base[0]
psurge = p_surge[0]

plane_sd = np.zeros((max_s + 1, max_d + 1), dtype=float)

for i in range(max_s+1):
   fix_supply = supply[i]*np.ones(max_d+1)
   plane_sd[i] = price_fixed_time(pbase, psurge, fix_supply, demand)

# np.savetxt("plane_sd.csv", plane_sd, delimiter = ',')

## plot plane_sd
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(demand, supply)
surf = ax.plot_surface(X, Y, plane_sd, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=np.amin(plane_sd) - 1, top=np.amax(plane_sd)) 
ax.view_init(elev = 30, azim = 120)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5,location='right', aspect=5)
plt.figure(figsize=(20,10))

ax.set_title("Fixed Time", fontsize = 15)
plt.show()
################################

fix_s = supply[20]
plane_pd = np.zeros((max_d + 1, num_samples + 1), dtype=float)
for i in range(max_d + 1):
    fix_demand = demand[i]*np.ones(num_samples+1)
    plane_pd[i] = price_fixed_supply(p_base, p_surge, fix_s, fix_demand)

# np.savetxt("plane_pd.csv", plane_pd, delimiter = ',')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x1, demand)
surf = ax.plot_surface(X, Y, plane_pd, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=30, top=42) 
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5,location='right', aspect=5)
plt.figure(figsize=(20,10))
ax.set_title("Fixed Supply", fontsize = 15)
plt.show()


# Fix supply as a constant
# fix_supply = supply[i], 0 <= i <= max_s + 1

# Another surface 1080 x 100

################################

# Fix demand as a constant
fix_d = demand[80]

plane_ps = np.zeros((max_s + 1, num_samples + 1), dtype=float)
for i in range(max_s + 1):
    fix_supply = supply[i]*np.ones(num_samples+1)
    plane_ps[i] = price_fixed_demand(p_base, p_surge, fix_supply, fix_d)

# np.savetxt("plane_ps.csv", plane_ps, delimiter = ',')

# Surface 1080 x 100

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x1, supply)
surf = ax.plot_surface(X, Y, plane_ps, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.set_zlim3d(bottom=30, top=40) 
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, location='right', aspect=5)
plt.figure(figsize=(20,10))
ax.set_title("Fixed Demand", fontsize = 15)
plt.show()

################################

# Full dataset 3D array aka a tensor
# try to generate the full dataset.
# My attempt at creating a 3D dataset was 62 MB. A bit too big. But here is the code for it commented out.


# 1080 x 100 x 100 elements

# tensor_psd = torch.zeros(max_s + 1, max_d + 1, num_samples + 1)
# # print(list(tensor_psd.size()))
# for i in range(max_s + 1):
#     fixed_s = supply[i]
#     tensor_plane_pd = np.zeros((max_d + 1, num_samples + 1), dtype=float)
#     for j in range(max_d + 1):
#         fixed_d = demand[j]*np.ones(num_samples+1)
#         tensor_plane_pd[j] = price_fixed_supply(p_base, p_surge, fixed_s, fixed_d)
#         # convert array to tensor
#     tensor_psd[i] = torch.from_numpy(tensor_plane_pd)

# torch.save(tensor_psd, 'tensor_psd.pt')


################################################################





