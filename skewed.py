import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import random
# fig, ax = plt.subplots(1, 1)
# a = 4
# mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
# x = np.linspace(skewnorm.ppf(0.01, a),
#                 skewnorm.ppf(0.99, a), 20)
# ax.plot(x, skewnorm.pdf(x, a),
#        'r-', lw=5, alpha=0.6, label='skewnorm pdf')

# rv = skewnorm(a)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# np.random.seed(seed = 2)
# r = skewnorm.rvs(3, size=20)
# print(r)

# np.random.seed(seed = 0)
# # minutes = skewnorm.rvs(3, loc = 10, size = 200)
# x = np.linspace(-2,8,21)
# minutes = (skewnorm.pdf(x,1)*10)+10
# print(minutes)

print(np.random.normal(size = 20))


x = np.linspace(-2,18,1081)
minutes = (skewnorm.pdf(x,1)*5)+10

y = np.linspace(-11,9,1081)
minutes2 = (skewnorm.pdf(y,1)*5)+10
# print(np.array(minutes2).argmax())
# print(minutes2[560])
# plt.plot(minutes)
# plt.show()
minutes3 = ((skewnorm.pdf(x,1) + skewnorm.pdf(y,1))*5)+10

plt.plot(minutes3)
plt.show()



# minutes = skewnorm.pdf(r,0)   
# print(minutes)

# plt.plot(minutes)
# plt.show()

# ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
# ax.set_xlim([x[0], x[-1]])
# ax.legend(loc='best', frameon=False)
# plt.show()