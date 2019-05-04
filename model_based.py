from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os

#load the data

oecd_path = "datasets/lifesat/oecd_bli_2015.csv"
gdp_path  = "datasets/lifesat/gdp_per_capita.csv"
# oced_dir = os.path.dirname(oecd_path)
# gdp_dir =os.path.dirname(gdp_path)

oecd_bli = pd.read_csv(oecd_path,thousands=',')
gdp_per_capita = pd.read_csv(gdp_path,thousands=',',
					delimiter='\t',encoding='latin1',na_values="n/a")


# print(oecd_bli.tail())
# print(gdp_per_capita.shape)

#prepare the data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


country_stats = prepare_country_stats(oecd_bli,gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# print(x.shape)
# print(y.shape)


#Visualize the data
country_stats.plot(kind='scatter',x="GDP per capita",y="Life satisfaction")
plt.show()

#select a linear model 
modelLinear = sklearn.linear_model.LinearRegression()
modelKneighbor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)


#train model 
modelLinear.fit(x,y)
modelKneighbor.fit(x,y)

# make predictions for cyprus
x_new = [[22587]] #cyprus gdp per capita
print("Linear Regression: ",modelLinear.predict(x_new))
print("K-Nearest Neighbor: ",modelKneighbor.predict(x_new))

