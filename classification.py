#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:29:41 2022

@author: fabienpelletier

classifications
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from sklearn import tree
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from perceptron_non import PERCEPTRON


path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind_classif'

X_train = pd.read_csv(path + '/' + 'train.csv')
y_train = X_train.pop('label').values.tolist()
X_train =X_train.values.tolist()

X_test = pd.read_csv(path + '/' + 'test.csv')
y_test = X_test.pop('label').values.tolist()
X_test = X_test.values.tolist()

print("Decision tree classifieur")
score_list = []
hyperp = ["entropy", "gini"]
for c in hyperp : 
    clf = tree.DecisionTreeClassifier(criterion=c)
    
    scored = cross_val_score(clf, X_train, y_train, cv=5)
    score_list.append(np.mean(scored))

best_c = hyperp[np.argmax(score_list)]
print("Meilleurs critère :", best_c, " avec :", max(score_list))

clf = tree.DecisionTreeClassifier(criterion=best_c)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("score jeux de test :", score )


print("Random forest Classifieur")
score_list = []
hyperp = ["entropy", "gini"]
for c in hyperp : 
    clf = RandomForestClassifier(criterion=c)
    
    scored = cross_val_score(clf, X_train, y_train, cv=5)
    score_list.append(np.mean(scored))

best_c = hyperp[np.argmax(score_list)]
print("Meilleurs critère :", best_c, " avec :", max(score_list))

clf = tree.DecisionTreeClassifier(criterion=best_c)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("score jeux de test :", score )



print("ONE VERSUS ONE")
model = PERCEPTRON(one_vers_all=False,shuffle=True, title="")
model.fit(X_train, y_train)
model.score(X_test, y_test, display=False)



