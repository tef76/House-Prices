#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:16:42 2022

@author: fabienpelletier
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA

path = '/home/fabienpelletier/Documents/Jeux de donnees/house-prices-advanced-regression-techniques'

train = pd.read_csv(path + '/' + 'train.csv')
test = pd.read_csv(path + '/' + 'test.csv')

# print(test.drop('Id'))
# print(test["SalePrice"])

all_data = pd.concat([train, test], keys= ["train", "test"])




# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"
all_data.loc[:, "Alley"] = all_data.loc[:, "Alley"].fillna("None")
# BsmtQual etc : data description says NA for basement features is "no basement"
all_data.loc[:, "BsmtQual"] = all_data.loc[:, "BsmtQual"].fillna("NA")
all_data.loc[:, "BsmtCond"] = all_data.loc[:, "BsmtCond"].fillna("NA")
all_data.loc[:, "BsmtExposure"] = all_data.loc[:, "BsmtExposure"].fillna("NA")
all_data.loc[:, "BsmtFinType1"] = all_data.loc[:, "BsmtFinType1"].fillna("NA")
all_data.loc[:, "BsmtFinType2"] = all_data.loc[:, "BsmtFinType2"].fillna("NA")
# CentralAir : NA most likely means No
all_data.loc[:, "CentralAir"] = all_data.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
all_data.loc[:, "Condition1"] = all_data.loc[:, "Condition1"].fillna("Norm")
all_data.loc[:, "Condition2"] = all_data.loc[:, "Condition2"].fillna("Norm")
# External stuff : NA most likely means average
all_data.loc[:, "ExterCond"] = all_data.loc[:, "ExterCond"].fillna("TA")
all_data.loc[:, "ExterQual"] = all_data.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
all_data.loc[:, "Fence"] = all_data.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
all_data.loc[:, "FireplaceQu"] = all_data.loc[:, "FireplaceQu"].fillna("NA")
# Functional : data description says NA means typical
all_data.loc[:, "Functional"] = all_data.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
all_data.loc[:, "GarageType"] = all_data.loc[:, "GarageType"].fillna("NA")
all_data.loc[:, "GarageFinish"] = all_data.loc[:, "GarageFinish"].fillna("NA")
all_data.loc[:, "GarageQual"] = all_data.loc[:, "GarageQual"].fillna("NA")
all_data.loc[:, "GarageCond"] = all_data.loc[:, "GarageCond"].fillna("NA")
# HeatingQC : NA most likely means typical
all_data.loc[:, "HeatingQC"] = all_data.loc[:, "HeatingQC"].fillna("TA")
# KitchenQual : NA most likely means typical
all_data.loc[:, "KitchenQual"] = all_data.loc[:, "KitchenQual"].fillna("TA")
# LotShape : NA most likely means regular
all_data.loc[:, "LotShape"] = all_data.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
all_data.loc[:, "MasVnrType"] = all_data.loc[:, "MasVnrType"].fillna("None")
# MiscFeature : data description says NA means "no misc feature"
all_data.loc[:, "MiscFeature"] = all_data.loc[:, "MiscFeature"].fillna("NA")
# PavedDrive : NA most likely means not paved
all_data.loc[:, "PavedDrive"] = all_data.loc[:, "PavedDrive"].fillna("N")
# SaleCondition : NA most likely means normal sale
all_data.loc[:, "SaleCondition"] = all_data.loc[:, "SaleCondition"].fillna("Normal")
all_data.loc[:, "PoolQC"] = all_data.loc[:, "PoolQC"].fillna("NA")
# Utilities : NA most likely means all public utilities
all_data.loc[:, "Utilities"] = all_data.loc[:, "Utilities"].fillna("AllPub")

all_data.loc[:, "MSZoning"] = all_data.loc[:, "MSZoning"].fillna("C (all)")


# BedroomAbvGr : NA most likely means 0
all_data.loc[:, "BedroomAbvGr"] = all_data.loc[:, "BedroomAbvGr"].fillna(0)
all_data.loc[:, "BsmtFullBath"] = all_data.loc[:, "BsmtFullBath"].fillna(0)
all_data.loc[:, "BsmtFinSF1"] = all_data.loc[:, "BsmtFinSF1"].fillna(0)
all_data.loc[:, "BsmtFinSF2"] = all_data.loc[:, "BsmtFinSF2"].fillna(0)
all_data.loc[:, "TotalBsmtSF"] = all_data.loc[:, "TotalBsmtSF"].fillna(0)
all_data.loc[:, "BsmtHalfBath"] = all_data.loc[:, "BsmtHalfBath"].fillna(0)
all_data.loc[:, "BsmtUnfSF"] = all_data.loc[:, "BsmtUnfSF"].fillna(0)
# EnclosedPorch : NA most likely means no enclosed porch
all_data.loc[:, "EnclosedPorch"] = all_data.loc[:, "EnclosedPorch"].fillna(0)
all_data.loc[:, "Fireplaces"] = all_data.loc[:, "Fireplaces"].fillna(0)
all_data.loc[:, "GarageArea"] = all_data.loc[:, "GarageArea"].fillna(0)
all_data.loc[:, "GarageCars"] = all_data.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
all_data.loc[:, "HalfBath"] = all_data.loc[:, "HalfBath"].fillna(0)
# KitchenAbvGr : NA most likely means 0
all_data.loc[:, "KitchenAbvGr"] = all_data.loc[:, "KitchenAbvGr"].fillna(0)
# LotFrontage : NA most likely means no lot frontage
all_data.loc[:, "LotFrontage"] = all_data.loc[:, "LotFrontage"].fillna(0)
all_data.loc[:, "MasVnrArea"] = all_data.loc[:, "MasVnrArea"].fillna(0)
all_data.loc[:, "MiscVal"] = all_data.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
all_data.loc[:, "OpenPorchSF"] = all_data.loc[:, "OpenPorchSF"].fillna(0)

all_data.loc[:, "PoolArea"] = all_data.loc[:, "PoolArea"].fillna(0)
all_data.loc[:, "ScreenPorch"] = all_data.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
all_data.loc[:, "TotRmsAbvGrd"] = all_data.loc[:, "TotRmsAbvGrd"].fillna(0)
# WoodDeckSF : NA most likely means no wood deck
all_data.loc[:, "WoodDeckSF"] = all_data.loc[:, "WoodDeckSF"].fillna(0)


# on surpprime car trop de NaN et impossible à remplacer 
del all_data['GarageYrBlt']


# Pour les colonnes qui contiennent des valeurs ordonnées on le rempli à la mains 
# car Label Encoder pourrai ne pas donner de valuers cohérentes. 
all_data = all_data.replace({
                      "LandContour" : {"Lvl" : 0, "Bnk" : 1, "HLS" : 2, "Low" : 3},
                      "Utilities"   : {"ELO" : 0, "NoSeWa" : 1, "NoSewr" : 2, "AllPub" : 3},
                      "LandSlope"   : {"Gtl" : 0, "Mod" : 1, "Sev" : 2},
                      "LotShape"    : {"IR3" : 0, "IR2" : 1, "IR1" : 2, "Reg" : 3},
                      "ExterQual"   : {"Po" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                      "ExterCond"   : {"Po" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4}, 
                      "BsmtQual"    : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},  
                      "BsmtCond"    : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}, 
                      "BsmtExposure" : {"NA" : 0, "No" : 0, "Mn" : 1, "Av" : 2, "Gd" : 3}, 
                      "BsmtFinType1" : {"NA" : 0, "Unf": 1, "LwQ": 1, "Rec" : 2, "BLQ": 3, "ALQ": 4, "GLQ": 5 },
                      "BsmtFinType2" : {"NA" : 0, "Unf": 1, "LwQ": 1, "Rec" : 2, "BLQ": 3, "ALQ": 4, "GLQ": 5 },
                      "HeatingQC"   : {"Po" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                      "CentralAir"  : {"N" : 0, "Y": 1}, 
                      "KitchenQual" : {"Po" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                      "Functional"  : {"Sal" : 0, "Sev" : 1, "Maj2" : 2, "Maj1" : 3, "Mod" : 4, "Min2" : 5, "Min1" : 5, "Typ" : 6},
                      "FireplaceQu" : {"NA": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                      "GarageFinish" : {"NA" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},
                      "GarageQual"  : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}, 
                      "GarageCond"  : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}, 
                      "PoolQC"      : {"NA" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4}, 
                      "Fence"       : {"No" : 0, "NA" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv" : 3, "GdPrv" : 4},
                      })





# Colonnes que non dont le valeurs ne sont pas ordonnée, on laisse LabelEncoder dessider.

# cols = ('MSZoning', "Street", "Alley", "LotConfig", "Neighborhood", "Condition1", "Condition2",
#         "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
#         "MasVnrType", "Foundation", "Heating" , "Electrical", "GarageType", "PavedDrive",
#         "MiscFeature", "SaleType", "SaleCondition")

# #
# # process columns, apply LabelEncoder to categorical features
# for c in cols:
#     lbl = LabelEncoder() 
#     lbl.fit(list(all_data[c].values))
#     all_data[c] = lbl.transform(list(all_data[c].values))








# test = train.loc['test']
# del test["SalePrice"]
# train = train.loc['train']


# if train.isnull().values.any() :
#     print("NaN dans TRAIN")
#     for i,ind in enumerate(train.values.tolist()) : 
#         for j, x in enumerate(ind) : 
#             if np.isnan(x): 
#                 print(i, "," , j)
                
                
# if test.isnull().values.any() :
#     print("Nan dans TEST")
#     for i,ind in enumerate(test.values.tolist()) : 
#         for j, x in enumerate(ind) : 
#             if np.isnan(x): 
#                 print(i, "," , j)







# path2save = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_Nan_free'

# train.to_csv(path_or_buf= path2save + '/' +  'train.csv', index=False) 
# test.to_csv(path_or_buf= path2save + '/' +  'test.csv', index=False) 




###############################################################################
###############################################################################












# pca = PCA()
# pca.fit(train)   

# explained= pca.explained_variance_ratio_
# print(explained)


# plt.figure() 
# plt.plot(explained)
# plt.title("Variance expliquée par n composantes")



    
# print("Find most important features relative to target")
# corr = train.corr()
# corr.sort_values(["SalePrice"], ascending = False, inplace = True)
# corrtab = corr.SalePrice
# print(corrtab)


# def hpart_(all_data) : 
"""
Fonction qui extrait un dataset dans lequelle on va essayer de conserver le 
plus de feature possible, mais en utilisant One Hot Encoder
et en "concaténant" les collonnes TrucQual et Truc de manière à avoir
un indicateur et plus deux.
"""


#ExterQual et ExterCond
#BsmtQual et BsmtCond
#Fireplace et fireplaceQu
#BsmtFinType1 et BsmtFinSF1
#BsmtFinType2 et BsmtFinSF2
#kitchen KitchenQual
#GarageCars et GarageQual
#PoolArea et PoolQC
# print(all_data.columns)

all_data['ExterInd'] = all_data['ExterQual'] * all_data["ExterCond"]

all_data['BsmtInd'] = all_data['BsmtQual'] * all_data["BsmtCond"]
all_data['FireplaceInd'] = all_data['Fireplaces'] * all_data["FireplaceQu"]
all_data['BsmtType1Ind'] = all_data['BsmtFinType1'] * all_data["BsmtFinSF1"]
all_data['BsmtType2Ind'] = all_data['BsmtFinType2'] * all_data["BsmtFinSF2"]
all_data['KitchenInd'] = all_data['KitchenAbvGr'] * all_data["KitchenQual"]
all_data['GarageInd'] = all_data['GarageCars'] * all_data["GarageQual"]
all_data['PoolInd'] = all_data['PoolArea'] * all_data["PoolQC"]

delete_list = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'Fireplaces', 
             'FireplaceQu', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
             'BsmtFinSF2', 'KitchenAbvGr',  'KitchenQual', 'GarageCars', 
             'GarageQual', 'PoolArea', 'PoolQC']

for c in delete_list : 
    del all_data[c]

del all_data['GarageArea'] # redondant avec GarageInd


print(all_data.columns)

all_data = all_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })





# One hot encoder
cols = ("MSSubClass", "MoSold", 'MSZoning', "Street", "Alley", "LotConfig", "Neighborhood", "Condition1", "Condition2",
        "BldgType", "HouseStyle", "RoofStyle", "RoofMatl","Exterior1st", "Exterior2nd",
        "MasVnrType", "Foundation", "Heating" , "Electrical", "GarageType", "PavedDrive",
        "MiscFeature", "SaleType", "SaleCondition")

total_columns = 0
for c in cols :
    data2enc =  all_data.pop(c)
    # print(data2enc)
    data_enc = pd.get_dummies(data2enc, prefix = c + "_")
    if c + "_None" in data_enc.columns:
        del data_enc['None']
    total_columns += len(data_enc.columns)
    all_data = pd.concat([all_data, data_enc], axis= 1)






    
print("Find most important features relative to target")
corr = all_data.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
corrtab = corr.SalePrice
print(corrtab)

# on enlève tout ce qui n'est pas  < 0,1

for colname, correlation in zip(corrtab.index, list(corrtab)) : 
    if colname == 'Id' :
        continue
    if abs(correlation) < 0.4 :
        del all_data[colname]



print("IL Y A : ", len(all_data.columns), "COLONNES")
print(all_data.columns)
# gerer "Exterior1st", "Exterior2nd"

# ext1 = all_data.pop("Exterior1st")
# ext2 = all_data.pop("Exterior2nd")
# nb_ligne = len(ext1.index)
# empty_col = np.zeros(nb_ligne)
# columns_name = ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard","ImStucc","MetalSd","Other",
# "Plywood", "PreCast", "Stone	", "Stucco", "VinylSd","Wd Sdng", "WdShing"]	
# columns_name = ["Exterior_" + c for c in columns_name]
# void_df = pd.DataFrame(columns=columns_name)
# for c in columns_name : 
#     void_df[c] = empty_col                 
# print(void_df)
# ext1 = pd.get_dummies(ext1, prefix="Exterior_")
# ext2 = pd.get_dummies(ext2, prefix="Exterior_")
# ext1 = pd.concat([void_df, ext1], join='outer')
# ext2 = pd.concat([void_df, ext2], join='outer')
# print(ext1)
# print(ext2)
# ext = ext1 + ext2
# # ext = pd.concat([ext1, ext2], axis=0).groupby().sum().reset_index()
# print(ext)

test = all_data.loc['test']
del test["SalePrice"]
train = all_data.loc['train']
idlist = test.pop("Id")
del train['Id']
    
path2save = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_OHE_Ind_22'

train.to_csv(path_or_buf= path2save + '/' + 'train.csv', index=False) 
test.to_csv (path_or_buf= path2save + '/' + 'test.csv',  index=False) 
idlist.to_csv (path_or_buf= path2save + '/' + 'id_test.csv',  index=False) 


def hpart_20(train, test) : 

    print("Find most important features relative to target after ")
    corr = train.corr()
    corr.sort_values(["SalePrice"], ascending = False, inplace = True)
    corrtab = corr.SalePrice
    print(corrtab)
    print("Il a :", len(corrtab), "features")
    
    """
    On ne va garder que les features qui sont le plus correller à la variable SalePrice
    On enlevera les eventuelles 'doublons' (fireplaceQuality et fireplaces sont redondant)
    """
    print(test.columns)
    idlist = test["Id"]
    
    keep = ["OverallQual", "GrLivArea", "ExterQual", "KitchenQual",  "GarageCars" ,"TotalBsmtSF", "1stFlrSF", 
            "BsmtQual", "FullBath", "GarageFinish", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd", "MasVnrArea",
            "Fireplaces", "HeatingQC", "BsmtFinSF1", "GarageType"]
    
    # on enlève tout ce qui n'a pas d'impact  < 0,5 
    
    # for colname, correlation in zip(corrtab.index, list(corrtab)) : 
    #     if abs(correlation) < 0.2 :
    #         del train[colname]
    #         del test[colname]
    
    for colname in test.columns.tolist():
        if not colname in keep : 
            del train[colname]
            del test[colname]
    
    print("feature conserver : ", len(keep), keep)
    
    path2save = '/home/fabienpelletier/Seafile/Ma bibliothèque/ProjetM1/Jeux de donnees/hpart_20'
    
    train.to_csv(path_or_buf= path2save + '/' + 'train.csv', index=False) 
    test.to_csv (path_or_buf= path2save + '/' + 'test.csv',  index=False) 
    idlist.to_csv (path_or_buf= path2save + '/' + 'id_test.csv',  index=False) 

