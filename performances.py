#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:29:14 2022

@author: fabienpelletier

Etude regression 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import mean_squared_log_error





path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_20'
path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_Nan_free'
path= '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind'
# path= '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind_103'

train = pd.read_csv(path + '/' + 'train.csv')
test = pd.read_csv(path + '/' + 'test.csv')
# ids = pd.read_csv(path + '/' +  'id_test.csv')

# ids = pd.DataFrame(test['Id'])
# del test['Id']
# del train['Id']


print(test)

print(train.shape)


LOG_TRANSFORM = True



def means_over_n_run(n, model, X_train, y_train, X_test, y_test): 
    
    score_list = []
    for i in range(n) : 
        model_train = clone(model)
        model_train.fit(X_train, y_train)
        score_list.append(model_train.score(X_test, y_test))
    return np.mean(score_list)


y_train = train.pop('SalePrice').values.tolist()
X_train = train.values.tolist()
print(len(X_train[0]))

import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# def rmsle(y, y_pred):
# 	assert len(y) == len(y_pred)
# 	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
# 	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


def rmsle(y, y_pred) : 
    return np.sqrt(mean_squared_log_error(y, y_pred))

# X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25)

y_train_log = [np.log(y) for y in y_train]



# print(X_test)

# if LOG_TRANSFORM : 
#     y_train = [np.log(y) for y in y_train]

# print("NAN ? ", test.isnull().values.any())

# for i,ind in enumerate(X_test) : 
#     for j, x in enumerate(ind) : 
        
#         if np.isnan(x): 
#             print(i, "," , j)

# print(test.loc['660'])


cv_nb = 5
"""""""""""""""""""""""""""""""""""
Ordinary least squares Linear Regression.
"""""""""""""""""""""""""""""""""""

model = LinearRegression()
y_pred = cross_val_predict(model, X_train, y_train, cv=cv_nb)
# y_pred = abs(y_pred)
y_pred = [y if y > 0 else 0 for y in y_pred]
score_lr = rmsle(y_train, y_pred)
model = LinearRegression()
y_pred = cross_val_predict(model, X_train, y_train_log, cv=cv_nb)
# y_pred = abs(y_pred)
# y_pred = [y if y > 0 else 0 for y in y_pred]
y_pred = [np.exp(y) for y in y_pred]
score_lr_log = rmsle(y_train, y_pred)


"""""""""""""""""""""""""""""""""""
Linear least squares with l2 regularization.
"""""""""""""""""""""""""""""""""""
alpha = 9

model = Ridge(alpha=alpha).fit(X_train, y_train)
y_pred = cross_val_predict(model, X_train, y_train, cv=cv_nb)
# y_pred = abs(y_pred)
y_pred = [y if y > 0 else 0 for y in y_pred]
score_rg = rmsle(y_train, y_pred)
model = Ridge(alpha=alpha).fit(X_train, y_train_log)
y_pred = cross_val_predict(model, X_train, y_train_log, cv=cv_nb)
# y_pred = abs(y_pred)
y_pred = y_pred = [np.exp(y) for y in y_pred]
score_rg_log = rmsle(y_train, y_pred)
"""""""""""""""""""""""""""""""""""
A decision tree regressor.
"""""""""""""""""""""""""""""""""""

model = DecisionTreeRegressor()
y_pred = cross_val_predict(model, X_train, y_train, cv=cv_nb)
score_dtr = rmsle(y_train, y_pred)

model = DecisionTreeRegressor()
y_pred = cross_val_predict(model, X_train, y_train_log, cv=cv_nb)
y_pred = [np.exp(y) for y in y_pred]
score_dtr_log = rmsle(y_train, y_pred)

"""""""""""""""""""""""""""""""""""
A random forest regressor.
"""""""""""""""""""""""""""""""""""

model = RandomForestRegressor(n_estimators=90, max_depth=330, min_samples_leaf=2)
y_pred = cross_val_predict(model, X_train, y_train, cv=cv_nb)
score_rfr = rmsle(y_train, y_pred)
model = RandomForestRegressor(n_estimators=90, max_depth=330, min_samples_leaf=2)
y_pred = cross_val_predict(model, X_train, y_train_log, cv=cv_nb)
y_pred = [np.exp(y) for y in y_pred]
score_rfr_log = rmsle(y_train, y_pred)
"""""""""""""""""""""""""""""""""""
Linear Model trained with L1 prior as regularizer (aka the Lasso).
"""""""""""""""""""""""""""""""""""
alphalasso = 0.0005591836734693878
model = Lasso(alpha=alphalasso, max_iter=30000)
y_pred = cross_val_predict(model, X_train, y_train, cv=cv_nb)
y_pred = [y if y > 0 else 0 for y in y_pred]
score_la = rmsle(y_train, y_pred)
model = Lasso(alpha=alphalasso, max_iter=30000)
y_pred = cross_val_predict(model, X_train, y_train_log, cv=cv_nb)
y_pred = y_pred = [np.exp(y) for y in y_pred]
score_la_log = rmsle(y_train, y_pred)
###############################################################################

name_list = ['LinearRegression', "Ridge Regression", 
             "DecisionTreeRegressor", "RandomForestRegressor", "Lasso Regressor"]

no_log = [score_lr, score_rg, score_dtr, score_rfr, score_la]
log = [score_lr_log, score_rg_log, score_dtr_log, score_rfr_log, score_la_log]
x = np.array([i for i in range(len(log))])

bar_width = 0.4



plt.figure(figsize=(9,8))
ax = plt.subplot(111)
plt.title("Erreur RMSLE de modèle de regression")



ax.bar(x, no_log, width=bar_width, label="no log")

ax.bar(x + bar_width, log, width=bar_width, label="with target log")

ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(name_list)
ax.set_ylabel("RMSLE, cross validation " + str(cv_nb) + " runs")
plt.xticks(rotation=15)
plt.legend()
