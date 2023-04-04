#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 15:31:47 2021

@author: fabien

AA1 PERCEPTRON
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# print(plt.style.available)
plt.style.use('seaborn')


# X = np.array([[0,1], [1,0], [2,2], [0,0], [-1,0], [-2, 0]])
# LABELS = np.array([1,1,1,2,2,2])




"""
Trouver un moyen de faire converger l'algo quand le prob n'est pas linéairement
séparable - autrement qu'avec la limite d'itération => SVM

Faire TOP1 TOP2

Matrice de confusion

"""


def transform(x, labels, i,j=None, one_vers_all=False):
    """


    Parameters
    ----------
    x : array
        données.
    labels : array
        labels des x.
    i : du type des élémetns de labels
        le labels de la premières classe.
    j : du type des éléments de labels, optional
        si one_versus_all vaut false
        le label de la deuxieme classe dans un one versus one. The default is None.
    one_vers_all : bool, optional
        suivre la logique one versus all et non one versus one. The default is False.

    Returns
    -------
    un tableau x comme :
        si one_vers_all == False :
            contient les x de la classe i : (x 1)
            et les x de la classe j : (x, -1)
        si one_vers_all == True :
            contient les x de la classe i : (x, 1)
            les x des classe j != i : (x, -1)


    """
    if j == None and not one_vers_all:
        raise Exception("J is missing in a one versus one")

    new_x = []
    if one_vers_all:

        for k in range(len(x)):
            if labels[k] == i:
                new_x.append(np.append(x[k], 1))
            else:
                new_x.append(np.append(-x[k], -1))

    else:

        for k in range(len(x)):
            if labels[k] == i:
                new_x.append(np.append(x[k], 1))
            if labels[k] == j:
                 new_x.append(np.append(-x[k], -1))

    return np.array(new_x)



def oneVone(x, labels, epoch, title="", display=False):
    # country_list renvois une liste de tuple de la forme (i, [wj,  j != i])
    classes = np.unique(labels)
    front_list = []
    country_list = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            # print(classes[i],classes[j])
            y = transform(x, labels, classes[i], classes[j])
            # on itère poiur trouver une séparatrice
            a = np.zeros(y[1].shape)
            well_classed_best = 0
            a_best = np.copy(a)

            for e in range(epoch):
                well_classed = 0
                np.random.shuffle(y)
                for yi in y:
                    if yi.T@a <= 0:
                        a = yi + a
                for yi in y :
                    if yi.T@a > 0:
                       well_classed += 1
                if well_classed > well_classed_best:
                    a_best = np.copy(a)
            front_list.append(a_best)
            country_list.append((classes[i], [classes[j]]))


    if display == True : 
        plt.figure(figsize=(15,10))
        plt.title(title)
        a = [z[0] for z in x]
        b = [z[1] for z in x]
        plt.scatter(a,b, c=labels,cmap='cool')
        i = 0
        for a1,a2, a0 in front_list:
            # droite d'équation a1*x1 + a2*x2 + a0 = 0:
            # Affichage
            random_color = tuple(np.random.choice(range(255),size=3) / 255)
            plt.axline((0, -a0/a2), xy2=(a[-1], -(a1*a[-1] + a0)/a2), c=random_color, label=country_list[i])
            i = i + 1
            # plt.axline((a[0], -(a1*a[0] + a0)/a2), xy2=(a[-1], -(a1*a[-1] + a0)/a2))
        x_max = max(a)
        y_max = max(b)
        plt.gca().set_xlim(-x_max, x_max)
        plt.gca().set_ylim(-y_max, y_max)
        plt.legend()
    # x_max = np.argmax(a)
    # y_max = np.argmax(b)
    # plt.gca().set_xlim(-x_max, x_max)
    # plt.gca().set_ylim(-y_max, y_max)
    return front_list, country_list


def oneVall(x, labels, epoch, title="", display=False):
    # country_list est de la forme liste de (i , [wj j != i])
    classes = np.unique(labels)
    front_list = []
    country_list = []

    for i in classes:
        y = transform(x, labels, i, one_vers_all=True)
        # on itère poiur trouver une séparatrice
        a = np.zeros(y[1].shape)
        a_best = np.copy(a)
        well_classed_best = 0
        # while not cond :
        for e in range(epoch):
            well_classed = 0
            for yi in y:
                if yi.T@a <= 0:
                    a = yi + a
            for yi in y:
                if yi.T@a > 0:
                    well_classed += 1

            if well_classed > well_classed_best:
                a_best = np.copy(a)
        # si on  pas modifié la vaeur de a alors cond vaut True et on
        #  sort
        front_list.append(a_best)
        country_list.append((i, [j for j in classes if j != i]))
        
    if display == True : 
        # Affichage
        plt.figure(figsize=(15,10))
        plt.title(title)
        a = [z[0] for z in x]
        b = [z[1] for z in x]
        plt.scatter(a,b, c=labels, cmap='cool')
        # plt.legend(classes)
        i = 0
        for a1,a2,a0 in front_list:
            # droite d'équation a1*x1 + a2*x2 + a0 = 0:
            random_color = tuple(np.random.choice(range(255),size=3) / 255)
            plt.axline((0, -a0/a2), xy2=(a[-1], -(a1*a[-1] + a0)/a2), c=random_color, label=str(country_list[i]))
            i = i + 1
            # plt.axline((a[0], -(a1*a[0] + a0)/a2), xy2=(a[-1], -(a1*a[-1] + a0)/a2))
        x_max = max(a)
        y_max = max(b)
        plt.gca().set_xlim(-x_max, x_max)
        plt.gca().set_ylim(-y_max, y_max)
        plt.legend()
        # x_max = np.argmax(a)
        # y_max = np.argmax(b)
    
        # plt.gca().set_xlim(-x_max, x_max)
        # plt.gca().set_ylim(-y_max, y_max)
    return front_list, country_list




def evaluate(X, front_list, country_list, classes):
    predict_labels = []
    for x in X:
        votes = {clef : 0 for clef in classes}
        x = np.append(x, 1)
        for wi in range(len(front_list)):
            c = x.T@front_list[wi]
            if c > 0:
                votes[country_list[wi][0]] += 1
            elif c < 0:
                for j in country_list[wi][1]: votes[j] += 1

        voted = sorted(votes.items(), key = lambda x:x[1], reverse=True)
        # print(voted)
        top1_vote = voted[0][1]
        top2_vote = voted[1][1]
        if top1_vote == top2_vote :
            # On ne classifie pas
            predict_labels.append(None)
        else :
            predict_labels.append([a for a,b in voted])


    return predict_labels



def get_score(y_pred, y_reel):
    count_good = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_reel[i]:
            count_good += 1
    return count_good / len(y_pred)



def get_x_label_from(path):
    tab = np.loadtxt(path)
    x = tab[:,1:]
    labels = tab[:,0]
    return x,labels



class PERCEPTRON():
    def __init__(self,epoch=100, one_vers_all=False, shuffle=False, title=""):
        # self.x_app = X
        self.ova = one_vers_all
        # self.labels = LABELS
        self.front_list = None
        self.count_list = None
        self.classes = None
        # self.y = None
        self.epoch = epoch
        self.shuffle = shuffle
        self.title = title

    def fit(self, X, y):
        if self.shuffle :
            tmp = list(zip(np.copy(X),np.copy(y)))
            np.random.shuffle(tmp)
            X,y = zip(*tmp)

        self.classes = np.unique(y)
        # self.y = y
        self.front_list, self.count_list = oneVall(X, y, self.epoch, self.title) \
            if self.ova else oneVone(X, y, self.epoch, self.title)

    def evaluate(self, X):
        # renvois un array avec les labels prédit par notre modèle
        predict_labels = evaluate(X, self.front_list, self.count_list, self.classes)



        return predict_labels

    def score(self, X,y, display=False):

        y_preds = self.evaluate(X)
        # y_pred_top1 = [i[0] for i in y_preds]
        top1 = 0
        top2 = 0
        # print(y_preds)
        tab = np.zeros(len(self.classes))
        no_classed = 0
        confusion = pd.DataFrame({i:tab for i in self.classes}, index=self.classes)
        for i in range(len(y_preds)):
            yi_tab = y_preds[i]
            # print(yi_tab)
            # top1
            # print(yi_tab[0])
            if yi_tab == None :
                no_classed += 1
            else :
                if yi_tab[0] == y[i] :
                    top1 += 1
                    top2 += 1
                elif yi_tab[1] == y[i]:
                    top2 += 1
                # confusion[y[i]][yi_tab[0]] += 1
                confusion[yi_tab[0]][y[i]] += 1


        print("\n Classification top 1 : ", top1 / len(X))
        print("Classification top 2 : ", top2 / len(X))
        print("Nombre de point non classés: ", no_classed)
        print("\nMATRICE DE CONFUSION : \n", confusion)
        if display :
            plt.figure(figsize=(15,10))
            plt.title("Test " + self.title)
            a = [z[0] for z in X]
            b = [z[1] for z in X]
            plt.scatter(a,b, c=y,cmap='cool')
            i = 0
            for a1,a2,a0 in self.front_list:
                # droite d'équation a1*x1 + a2*x2 + a0 = 0:
                random_color = tuple(np.random.choice(range(255),size=3) / 255)
                plt.axline((0, -a0/a2), xy2=(a[-1], -(a1*a[-1] + a0)/a2), c=random_color, label=self.count_list[i])
                i = i + 1
            x_max = max(a)
            y_max = max(b)
            plt.gca().set_xlim(-x_max, x_max)
            plt.gca().set_ylim(-y_max, y_max)
            plt.legend()




        # print("\n Classification Rate : ", get_score(y_pred, y))


        # return get_score(y_pred, y)




def main () : 

    PATH = "../data"
    
    
    # X = np.array([[0,1], [1,0], [2,2], [0,0], [-1,0], [-2, 0]])
    # LABELS = np.array([1,1,1,2,2,2])
    
    # X = np.array([[0,1], [1,1], [1,0], [0,0]])
    # LABELS = np.array([1,1,1,2])
    
    
    # X = X[100:300]
    # y = y[100:300]
    
    # X_test = X_test[100:300]
    # y_test = y_test[100:300]
    # X = X[300:500]
    # y = y[300:500]
    
    # X_test = X_test[300:500]
    # y_test = y_test[300:500]
    
    
    
    
    
    # print("#########  JEUX DE DONNEES 1 ########\n")
    
    # Xall,yall = get_x_label_from(PATH + '/' + 'data_tp1_app.txt')
    # X_test_all, y_test_all = get_x_label_from(PATH + '/' + 'data_tp1_dec.txt')
    
    
    # X = [x for i,x in enumerate(Xall) if yall[i] != 5]
    # y = [yi for yi in yall if yi != 5]
    
    # X_test = [x for i,x in enumerate(X_test_all) if y_test_all[i] != 5]
    # y_test = [yi for yi in y_test_all if yi != 5]
    
    
    
    # print("ONE VERSUS ONE")
    # model = PERCEPTRON(one_vers_all=False, shuffle=True,title="1v1 Jeux de Données 1 ")
    # model.fit(X,y)
    # model.score(X_test, y_test, display=True)
    
    # print("ONE VERSUS ALL")
    # model = PERCEPTRON(one_vers_all=True,shuffle=True, title="1vall Jeux de Données 1")
    # model.fit(X,y)
    # model.score(X_test, y_test, display=True)
    
    
    
    
    # print("#########  JEUX DE DONNEES 2 ########\n")
    
    # Xall,yall = get_x_label_from(PATH + '/' + 'data_tp2_app.txt')
    # X_test_all, y_test_all = get_x_label_from(PATH + '/' + 'data_tp2_dec.txt')
    
    # X = [x for i,x in enumerate(Xall) if yall[i] != 3]
    # y = [yi for yi in yall if yi != 3]
    
    # X_test = [x for i,x in enumerate(X_test_all) if y_test_all[i] != 3]
    # y_test = [yi for yi in y_test_all if yi != 3]
    
    
    
    
    # print("ONE VERSUS ONE")
    # model = PERCEPTRON(one_vers_all=False,shuffle=True, title="1v1 Jeux de Données 2")
    # model.fit(X,y)
    # model.score(X_test, y_test, display=True)
    
    # print("ONE VERSUS ALL")
    # model = PERCEPTRON(one_vers_all=True,shuffle=True, title="1vall Jeux de Données 2")
    # model.fit(X,y)
    # model.score(X_test, y_test, display=True)
    
    
    
    
    
    print("#########  JEUX DE DONNEES 3 ########\n")
    
    Xall,yall = get_x_label_from(PATH + '/' + 'data_tp3_app.txt')
    X_test_all, y_test_all = get_x_label_from(PATH + '/' + 'data_tp3_dec.txt')
    
    X = [x for i,x in enumerate(Xall) if yall[i] != 1]
    y = [yi for yi in yall if yi != 1]
    
    X_test = [x for i,x in enumerate(X_test_all) if y_test_all[i] != 1]
    y_test = [yi for yi in y_test_all if yi != 1]
    
    
    
    
    print("ONE VERSUS ONE")
    model = PERCEPTRON(one_vers_all=False,shuffle=True, title="1v1 Jeux de Données 3")
    model.fit(X,y)
    model.score(X_test, y_test, display=True)
    
    print("ONE VERSUS ALL")
    model = PERCEPTRON(one_vers_all=True,shuffle=True, title="1vall Jeux de Données 3")
    model.fit(X,y)
    model.score(X_test, y_test, display=True)
    
    
    
    
    # classes = [1,2,3]
    # tab = np.zeros(3)
    # confusion = pd.DataFrame({i:tab for i in classes}, index=classes)
    # print(confusion)
    # confusion[1][2] += 1
    # print(confusion)
    
    


if __name__  == '__main__' : 
    main()