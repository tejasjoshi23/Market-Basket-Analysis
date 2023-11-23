import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from itertools import combinations
# from collections import Counter
import os

def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]

april_sales_dataframe = pd.read_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\Sales_Data\\Sales_April_2019.csv')
# print(april_sales_dataframe.head())

files = [file for file in os.listdir('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\Sales_Data')]

all_months_sales_dataframe = pd.DataFrame()

for file in files:
    dataframe = pd.read_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\Sales_Data\\' + file)
    all_months_sales_dataframe = pd.concat([all_months_sales_dataframe, dataframe])

all_months_sales_dataframe.to_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\12 months sales.csv', index = False)

complete_sales_dataframe = pd.read_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\12 months sales.csv')
complete_sales_dataframe = complete_sales_dataframe.dropna(how = 'all')
complete_sales_dataframe = complete_sales_dataframe[complete_sales_dataframe['Order Date'].str[0:2] != 'Or']

complete_sales_dataframe['Month'] = complete_sales_dataframe['Order Date'].str[0:2]
complete_sales_dataframe['Month'] = complete_sales_dataframe['Month'].astype('int32')
# print(complete_sales_dataframe.head())

complete_sales_dataframe['Price Each'] = pd.to_numeric(complete_sales_dataframe['Price Each'])
complete_sales_dataframe['Quantity Ordered'] = pd.to_numeric(complete_sales_dataframe['Quantity Ordered'])

complete_sales_dataframe['Sales'] = complete_sales_dataframe['Price Each'] * complete_sales_dataframe['Quantity Ordered']
# print(complete_sales_dataframe.head())

months = range(1, 13)
sales_months = complete_sales_dataframe.groupby('Month').sum()
# plt.bar(months, sales_months['Sales'], color = plt.cm.rainbow(np.linspace(0, 1, 12)))
# y_label, y_tick = plt.yticks()
# plt.xticks(months)
# plt.yticks(y_label, (y_label / 1000000).astype(int))
# plt.xlabel('Months')
# plt.ylabel('Sales in Million USD')
# plt.tight_layout()
# plt.show()

complete_sales_dataframe['City'] = complete_sales_dataframe['Purchase Address'].apply(lambda x: get_city(x) + " " + get_state(x))
# print(complete_sales_dataframe.head())

sales_city = complete_sales_dataframe.groupby('City').sum()
# print(sales_city)

cities = [city for city, x in complete_sales_dataframe.groupby('City')]

# plt.bar(cities, sales_city['Sales'], color = plt.cm.rainbow(np.linspace(0, 1, 12)))
# y_label, y_tick = plt.yticks()
# plt.xticks(cities, rotation = 90)
# plt.yticks(y_label, (y_label / 1000000).astype(int))
# plt.xlabel('Cities')
# plt.ylabel('Sales in Million USD')
# plt.tight_layout()
# plt.show()

complete_sales_dataframe['Order Date'] = pd.to_datetime(complete_sales_dataframe['Order Date'])
complete_sales_dataframe['Hour'] = complete_sales_dataframe['Order Date'].dt.hour
# print(complete_sales_dataframe.head())

hours = [hour for hour, x in complete_sales_dataframe.groupby('Hour')]
order_count = complete_sales_dataframe.groupby(['Hour'])['Quantity Ordered'].count()

# plt.bar(hours, order_count, color = plt.cm.rainbow(np.linspace(0, 1, 24)))
# plt.xticks(hours)
# plt.xlabel('Hour')
# plt.ylabel('Number Of Orders')
# plt.tight_layout()
# plt.show()

quantity_bought = complete_sales_dataframe.groupby('Product')['Quantity Ordered'].sum()
products_bought = [product for product, x in complete_sales_dataframe.groupby('Product')]

# plt.bar(products_bought, quantity_bought, color = plt.cm.rainbow(np.linspace(0, 1, 24)))
# plt.xticks(products_bought, rotation = 90)
# plt.xlabel('Products')
# plt.ylabel('Quantity Ordered')
# plt.tight_layout()
# plt.show()

product_price = complete_sales_dataframe.groupby('Product').mean()['Price Each']

fig, axis_1 = plt.subplots()

axis_2 = axis_1.twinx()
axis_1.bar(products_bought, quantity_bought, color = plt.cm.rainbow(np.linspace(0, 1, len(products_bought))))
axis_2.plot(products_bought, product_price, color = 'green')

axis_1.set_xlabel('Products')
axis_1.set_ylabel('Quantity Ordered')
axis_1.set_xticks(products_bought)
fig.autofmt_xdate(rotation = 60)
axis_2.set_ylabel('Price ($)')
plt.tight_layout()
plt.show()

# no_duplicate_dataframe = complete_sales_dataframe[complete_sales_dataframe['Order ID'].duplicated(keep = False)]
# no_duplicate_dataframe['Product Bundle'] = no_duplicate_dataframe.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
# no_duplicate_dataframe = no_duplicate_dataframe[['Order ID', 'Product Bundle']].drop_duplicates()
# print(no_duplicate_dataframe.head())

# count = Counter()

# for element in no_duplicate_dataframe['Product Bundle']:
#     element_list = element.split(',')
#     count.update(Counter(combinations(element_list, 2)))

# print(count.most_common(5))