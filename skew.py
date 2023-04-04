#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:37:02 2022

@author: fabienpelletier

Regarder si les données sont skewed ou pas 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.api import qqplot

LOG_TRANSFORM = True


path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_20'
path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_Nan_free'
path= '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind'
# path= '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind_103'

train = pd.read_csv(path + '/' + 'train.csv')
test = pd.read_csv(path + '/' + 'test.csv')
ids = pd.read_csv(path + '/' +  'id_test.csv')



## NO LOG
# sns.displot(train)
# col_interest = [c for c in train.columns if max(train[c].tolist()) > 1]

# n = len(col_interest) // 6
# m = 6
# fig, axes= plt.subplots(ncols=n, nrows=m, figsize=(50,50))

# for c, ax in zip(col_interest, axes.ravel()): 
#     sns.histplot(train[c], ax=ax)
    
    
# ## LOGED

# log_train = np.log1p(train)

# col_interest = [c for c in log_train.columns if max(log_train[c].tolist()) > 1]

# n = len(col_interest) // 6
# m = 6

# fig, axes= plt.subplots(ncols=n, nrows=m, figsize=(50,50))

# for c, ax in zip(col_interest, axes.ravel()): 
#     sns.histplot(log_train[c], ax=ax)
    

###############################################################################
#   QQPLOTS


plt.figure()
ax = plt.subplot(111)
plt.title("GrLivArea qqplot")
skewed = train['GrLivArea']
qqplot(skewed, line='s', ax=ax)
plt.show()

plt.figure(figsize=(18,10))

ax = plt.subplot(1,2,1)
plt.title("SalePrice qqplot")
skewed = train['SalePrice']
qqplot(skewed, line='s', ax=ax)


###############################################################################
# SalePrice distribution

# plt.figure() 
ax = plt.subplot(1,2,2)

plt.title("Déviation positive")
sns.histplot(train["SalePrice"], kde=True, ax=ax)
plt.show()



plt.figure(figsize=(18,10))

ax = plt.subplot(1,2,1)
plt.title("log(SalePrice qqplot)")
skewed = np.log(train['SalePrice'])
qqplot(skewed, line='s', ax=ax)


ax = plt.subplot(1,2,2)

plt.title("log(SalePrice)")
sns.histplot(np.log(train["SalePrice"]), kde=True, ax=ax)





"""
LEs deux sont right skewed 
"""