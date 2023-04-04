#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:50:52 2022

@author: fabienpelletier

clustering 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split




path = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind_22'

train = pd.read_csv(path + '/' + 'train.csv')

# train = train[train['GrLivArea'] < 4000 ]
train = train[train['SalePrice'] < 500000 ]



y = train.pop('SalePrice').values.tolist()
y = (np.array(y)).reshape(-1, 1)


print(y)

km = KMeans(n_clusters=4, n_init=1000).fit(y)


labels = km.labels_

val, count = np.unique(labels, return_counts=True)

print(km.inertia_)

print(val) 
print(count)

# plt.figure() 
# plt.plot(sorted(y))

plt.figure() 

plt.scatter(y, [labels[i] for i in range(len(y))], c=labels)

plt.legend()




def rename_class(saleprice, labels) : 
    # means
    means = []
    for i in np.unique(labels):
        # print("i:", i)
        mean = np.mean([price  for price, l in zip(saleprice, labels) if l == i ])
        means.append((mean, i))
        # assert len([price  for price, l in zip(saleprice, labels) if l == i ]) == len(y)
    
    means.sort()
    labeltoreplace = [x[1] for x in means]
    
    final = [-1 for i in range(len(labels))]
    for good, bad in enumerate(labeltoreplace):
        
        for i,l in enumerate(labels) :
            if l == bad : 
                final[i] = good
    return final
    

labels = rename_class(y, labels)
    



plt.figure() 
plt.gca().set_xlabel('SalePrice ($)')
plt.gca().set_yticklabels(["", "w1","", "w2","", "w3","", "w4"])
plt.title("Répartition des classes")
plt.scatter(y, [labels[i] for i in range(len(y))], c=labels)




X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)



X_train.insert(len(X_train.columns), "label", y_train)
X_test.insert(len(X_test.columns), "label", y_test)


path2save = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind_22_classif'

X_train.to_csv(path2save + '/' + 'train.csv', index=False)
X_test.to_csv(path2save + '/' + 'test.csv', index=False)










# inertia = []
# x = [k for k in range(2, 11)]


# for k in x : 
    
    
#     km = KMeans(n_clusters=k, n_init=1000).fit(y)
#     inertia.append(km.inertia_)
    

# plt.figure()
# plt.plot(x, inertia)


# labels = km.labels_

# plt.figure() 

# plt.scatter(y, [labels[i] for i in range(len(y))], c=labels)

# plt.legend()

