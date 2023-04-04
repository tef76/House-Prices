#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:56:19 2022

@author: fabienpelletier

Regression 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
LOG_TRANSFORM = True


path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_20'
path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_Nan_free'
path= '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind'
# path= '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind_103'

train = pd.read_csv(path + '/' + 'train.csv')
test = pd.read_csv(path + '/' + 'test.csv')
ids = pd.read_csv(path + '/' +  'id_test.csv')

# ids = pd.DataFrame(test['Id'])
# del test['Id']
# del train['Id']


print(test)

print(train)


def means_over_n_run(n, model, X_train, y_train, X_test, y_test): 
    
    score_list = []
    for i in range(n) : 
        model_train = clone(model)
        model_train.fit(X_train, y_train)
        score_list.append(model_train.score(X_test, y_test))
    return np.mean(score_list)






y_train = train.pop('SalePrice').values.tolist()
X_train = train.values.tolist()

X_test = test.values.tolist()

# print(X_test)

# if LOG_TRANSFORM : 
#     y_train = [np.log1p(y) for y in y_train]
    

y_train_log = [np.log(y) for y in y_train]

print("NAN ? ", test.isnull().values.any())

for i,ind in enumerate(X_test) : 
    for j, x in enumerate(ind) : 
        
        if np.isnan(x): 
            print(i, "," , j)

# print(test.loc['660'])


"""""""""""""""""""""""""""""""""""
Ordinary least squares Linear Regression.
"""""""""""""""""""""""""""""""""""

# model = LinearRegression().fit(X_train, y_train)

# score = model.score(X_train, y_train)

# print(score)
# y_pred = model.predict(X_test)

# ids.insert(1, "SalePrice",  y_pred)
# submission = ids


"""""""""""""""""""""""""""""""""""
Linear least squares with l2 regularization.
"""""""""""""""""""""""""""""""""""


param_list = {'alpha' : [float(i) for i in range(1, 15)]}

# model = Ridge()
# print(model.get_params().keys())
# GSCV = GridSearchCV(estimator=model, param_grid=param_list, cv=5, verbose=4)
# GSCV.fit(X_train, y_train)
# model = GSCV.best_estimator_
# print(model)




# model = Ridge(alpha=9.0).fit(X_train, y_train)

# score = model.score(X_train, y_train)

# # alpha = 9 
# # Meilleure score : 0.13066



# print(score)
# y_pred = model.predict(X_test)

# if LOG_TRANSFORM: 
#     y_pred = [np.exp(y) for y in y_pred]

# ids.insert(1, "SalePrice",  y_pred)
# submission = ids

"""""""""""""""""""""""""""""""""""
Linear Model trained with L1 prior as regularizer (aka the Lasso).
"""""""""""""""""""""""""""""""""""



param_list = {'alpha' : [float(i) for i in np.linspace(0.0001, 0.001)]}

# best alpha :  0.0005591836734693878
# model = Lasso()
# print(model.get_params().keys())
# GSCV = GridSearchCV(estimator=model, param_grid=param_list, cv=5, verbose=4)
# GSCV.fit(X_train, y_train)
# model = GSCV.best_estimator_
# print(model)
# print(GSCV.best_params_)


model = Lasso(alpha= 0.0005591836734693878).fit(X_train, y_train_log)

score = model.score(X_train, y_train_log)

# alpha =  0.0005591836734693878

# best score : 0.12945


print(score)
y_pred = model.predict(X_test)
# y_pred = [y if y > 0 else 0 for y in y_pred]
print(y_pred)
y_pred = [np.exp(y) for y in y_pred]
print(y_pred)


ids.insert(1, "SalePrice",  y_pred)
submission = ids


"""""""""""""""""""""""""""""""""""
A random forest regressor.
"""""""""""""""""""""""""""""""""""

param_list = {'n_estimators' : [50, 100, 200, 500], 
              'criterion' : ["squared_error", "absolute_error", "poisson"],
              'max_depth' : [None, 3, 5, 10, 100, 500],
              'bootstrap' : [True, False]}

param_list = {'max_depth' : [1,5, 10, 100]}
param_list = {'max_depth' : [ 100, 150, 200, 300, 500, 1000]}
param_list = {'max_depth' : [ 250, 260, 270, 300, 310, 320, 330, 400]}
# param_list = {'min_samples_leaf' : [i / 100 for i in range(1, 51)]}
param_list = {'min_samples_leaf' : [i for i in range(1, 5)], 
              'max_depth' : [300, 310, 320, 330, 340],
              'n_estimators' : [80, 90, 100, 110, 120]}


# max depth 330
# n_estimator 90
# min_sample_leaf = 2

# Meilleur score : 0.14274


# model = RandomForestRegressor()
# print(model.get_params().keys())
# GSCV = GridSearchCV(estimator=model, param_grid=param_list, cv=5, verbose=4)
# GSCV.fit(X_train, y_train)
# model = GSCV.best_estimator_
# print(model)

# print(GSCV.best_params_)

# score = model.score(X_train, y_train)

# print(score)
# y_pred = model.predict(X_test)

# if LOG_TRANSFORM: 
#     y_pred = [np.exp(y) for y in y_pred]

# ids.insert(1, "SalePrice",  y_pred)
# submission = ids




# model = RandomForestRegressor(n_estimators=90, max_depth=330, min_samples_leaf=2).fit(X_train, y_train)

# score = model.score(X_train, y_train)

# print(score)
# y_pred = model.predict(X_test)

# if LOG_TRANSFORM: 
#     y_pred = [np.exp(y) for y in y_pred]

# ids.insert(1, "SalePrice",  y_pred)
# submission = ids




###############################################################################
path2sub = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/submissions'



def get_nb_sub(path2sub) : 
    nb_list = []
    for name in listdir(path2sub): 
        try :
            nb_list.append(int(name[-6:-4]))
        except :
            nb_list.append(int(name[-5]))
    return max(nb_list) + 1 
    

nb = str(get_nb_sub(path2sub))
submission.to_csv(path2sub + '/' + 'submission' + nb +  '.csv', index=False)
