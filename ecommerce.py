### Attempt at dynamic pricing

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# print(os.getcwd())
path = os.getcwd() + "/e-commerce"
# filenames = glob.glob(path + "/*.csv")
order_items = pd.read_csv(path + "/olist_order_items_dataset.csv")

# Create Pivot table to show top 10 products ordered
pivot = order_items.pivot_table(index= ['product_id'], aggfunc= 'size')
pivot = pivot.sort_values(ascending = False)
# print(pivot)
# print(pivot[0:10])
# print(pivot.index[0:10])


# # # print(order_items)
print(order_items["product_id"].mode()[0])
common_0 = order_items[order_items["product_id"] == order_items["product_id"].mode()[0]]

common_0 = common_0.sort_values(by=['order_item_id'], ascending=False)


## Demand Curve (ish)
common_0_0 = common_0.drop_duplicates(subset = ['order_id', 'product_id'])
# print(common_0)
common_0_0.plot(x = 'order_item_id', y = 'price', kind = 'scatter')

### seller id
pivot_1 = common_0.pivot_table(index= ['seller_id'], aggfunc= 'size')
pivot_1 = pivot_1.sort_values(ascending = False)

print(pivot_1)

common_0_0.sort_values(by=['shipping_limit_date'], ascending=True).plot(x = 'shipping_limit_date', y = 'price', kind = 'scatter')


# common_1 = order_items[order_items["product_id"] == pivot.index[1]]
# common_1 = common_1.sort_values(by=['shipping_limit_date'], ascending=True)
# # print(common_0)

# plt.figure()
# common_0.plot(x = 'shipping_limit_date',y = 'price',kind = 'scatter')
# common_1.plot(x = 'shipping_limit_date',y = 'price',kind = 'scatter')
# plt.show()


# for i in range(11):
#     product = order_items[order_items["product_id"] == pivot.index[i]]
#     product = product.sort_values(by=['shipping_limit_date'], ascending=True)
#     plt.figure()
#     # product.plot(x = 'shipping_limit_date',y = 'price',kind = 'scatter', title = pivot.index[i]+ ' mode:'+pivot[i])
#     product.plot(x = 'shipping_limit_date',y = 'price',kind = 'scatter')
#     plt.show()