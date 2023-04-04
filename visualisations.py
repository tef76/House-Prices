#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:46:07 2022

@author: fabienpelletier

regression linéaire
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

plt.style.use('seaborn')

path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/house-prices-advanced-regression-techniques'

train = pd.read_csv(path + '/' + 'train.csv')




columns_nan = []
for c in train.columns : 
    if train[c].isnull().values.any() :
        columns_nan.append(c)
        
y = []

for c in columns_nan : 
    y.append(train[c].isnull().sum())


tmp = zip(y, columns_nan)
tmp = sorted(tmp, reverse=True)
y, columns_nan = list(zip(*tmp))
y = (np.array(y)*100) / len(train.index)

plt.figure(figsize=(15,10)) 
plt.title("Proportion de Nan par colonnes\n (les colonnes n'en contenant pas ne sont pas affichées)")
ax = plt.subplot(111)

ax.bar(columns_nan, y)
plt.xticks(rotation=20)
plt.gca().set_ylabel("Pourcentage de NaN dans la colonnes")



# version regression avce Sale price 


path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_Nan_free'

train = pd.read_csv(path + '/' + 'train.csv')
test = pd.read_csv(path + '/' + 'test.csv')


plt.figure()
ax = plt.subplot(111)
sns.histplot(train['GarageQual'], ax=ax)


plt.figure()
ax = plt.subplot(111)
plt.title("Histogramme des prix \n (SalePrice)")
sns.histplot(train['SalePrice'], ax=ax)




print("Find most important features relative to target after ")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
corrtab = corr.SalePrice
print(corrtab)
print("Il reste :", len(corrtab), "features")




plt.figure() 
plt.title("Sale Price en fonction de OverallQu ")
plt.scatter(train['OverallQual'], train['SalePrice'])


plt.figure(figsize=(15, 5)) 
plt.title("SalePrice en fonction de GrLivArea")
plt.scatter(train['GrLivArea'], train['SalePrice'])



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7,15))
axes = axes.ravel()
sns.histplot(train['Fireplaces'], ax=axes[0])
sns.histplot(train['FireplaceQu'], ax=axes[1])
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7,15))
axes = axes.ravel()
sns.histplot(train['PoolQC'], ax=axes[0])
sns.histplot(train['PoolArea'], ax=axes[1])



path= '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind'
train = pd.read_csv(path + '/' + 'train.csv')
test = pd.read_csv(path + '/' + 'test.csv')



# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
# axes = axes.ravel()
plt.figure()
sns.histplot(train['FireplaceInd'])
plt.show()
plt.figure()
sns.histplot(train['PoolInd'])
plt.show()


###############################################################################
#Lineare REgression 

X_train = train['GrLivArea']
print(X_train)
X_train = np.array(X_train).reshape(-1, 1)
y_train = train['SalePrice']

model = LinearRegression().fit(X_train, y_train)
y_lr = model.predict(X_train)
print(model.coef_)
print("y_lr:", y_lr)


model = Ridge(alpha=6).fit(X_train, y_train)
y_rg = model.predict(X_train)



model = DecisionTreeRegressor().fit(X_train, y_train)
y_dtr = model.predict(X_train)

print("y_dtr : ", y_dtr)

plt.figure(figsize=(10, 5)) 
plt.title("SalePrice en fonction de GrLivArea")
plt.scatter(X_train, y_train)

plt.plot(X_train, y_lr, c="orange", label="LinearRegression")
# plt.plot(X_train, y_rg, c="purple", label="Ridge")
# plt.scatter(X_train, y_dtr, c="green", label="DecisionTreeRegressor")

plt.gca().set_xlabel('GrLivArea')
plt.gca().set_ylabel('SalePrice')
plt.legend()


X_train = train['OverallQual']
print(X_train)
X_train = np.array(X_train).reshape(-1, 1)
y_train = train['SalePrice']

model = LinearRegression().fit(X_train, y_train)
y_lr = model.predict(X_train)
print(model.coef_)
print("y_lr:", y_lr)


model = Ridge(alpha=6).fit(X_train, y_train)
y_rg = model.predict(X_train)



model = DecisionTreeRegressor().fit(X_train, y_train)
y_dtr = model.predict(X_train)

print("y_dtr : ", y_dtr)

plt.figure(figsize=(10, 5)) 
plt.title("SalePrice en fonction de OverallQual")
plt.scatter(X_train, y_train)

plt.plot(X_train, y_lr, c="orange", label="LinearRegression")
# plt.plot(X_train, y_rg, c="purple", label="Ridge")
# plt.scatter(X_train, y_dtr, c="green", label="DecisionTreeRegressor")

plt.gca().set_xlabel('OverallQual')
plt.gca().set_ylabel('SalePrice')
plt.legend()

X_train = train['BsmtInd']
print(X_train)
X_train = np.array(X_train).reshape(-1, 1)
y_train = train['SalePrice']

model = LinearRegression().fit(X_train, y_train)
y_lr = model.predict(X_train)
print(model.coef_)
print("y_lr:", y_lr)


model = Ridge(alpha=6).fit(X_train, y_train)
y_rg = model.predict(X_train)



model = DecisionTreeRegressor().fit(X_train, y_train)
y_dtr = model.predict(X_train)

print("y_dtr : ", y_dtr)

plt.figure(figsize=(10, 5)) 
plt.title("SalePrice en fonction de OverallQual")
plt.scatter(X_train, y_train)


plt.plot(X_train, y_lr, c="orange", label="LinearRegression")
# plt.plot(X_train, y_rg, c="purple", label="Ridge")
plt.scatter(X_train, y_dtr, c="green", label="DecisionTreeRegressor")

plt.gca().set_xlabel('BsmtInd')
plt.gca().set_ylabel('SalePrice')
plt.legend()











###############################################################################
# versions avec les classes: 
    
# path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_20_classif'

# train = pd.read_csv(path + '/' + 'train.csv')
# test = pd.read_csv(path + '/' + 'test.csv')


# sns.set(rc={"figure.figsize":(10, 5)})

# # sns.title("Sale price en fonction de GrLivArea")
# sns.displot(train, x="GrLivArea", y="GarageArea", kind="kde", hue='label')
# sns.displot(train, x="GrLivArea"    , kind="kde", hue='label', palette='cool').set(title="Classe en fonction de GrLivArea")
# sns.displot(train, x="OverallQual"    , kind="hist", hue='label', palette='cool').set(title="Classe en fonction de OverallQual")
# # sns.displot(train, x="OverallQual"    , kind="hist", hue='label', palette='cool').set(title="Classe en fonction de OverallQual")
# sns.displot(train, x="KitchenQual"    , kind="hist", hue='label', palette='cool').set(title="Classe en fonction de OverallQual")
# sns.displot(train, x="GarageType"    , kind="hist", hue='label', palette='cool').set(title="Classe en fonction de OverallQual")
# sns.displot(train, x="Fireplaces"    , kind="hist", hue='label', palette='cool').set(title="Classe en fonction de Fireplaces")
# sns.displot(train, x="FireplaceQu"    , kind="hist", hue='label', palette='cool').set(title="Classe en fonction de FireplaceQu")
# sns.displot(train, x="OverallQual",  y='label'  , kind="hist", hue='label').set(title="Classe en fonction de OverallQual")


# # sns.pairplot(train)
# sns.pairplot(train, hue="OverallQual")
