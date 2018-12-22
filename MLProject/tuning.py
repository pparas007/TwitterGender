# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 19:08:13 2018

@author: 007Paras
"""

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def grid_search(X,y):
    C_array = [0.001, 0.01, 0.1, 1, 10]
    gamma_array = [0.001, 0.01, 0.1, 1]
    hyperparameters = {'C': C_array, 'gamma' : gamma_array}
    grid_search = GridSearchCV(SVC(kernel = 'rbf'), hyperparameters, cv=10)
    grid_search.fit(X, y)
    return grid_search.best_params_.get('C'),grid_search.best_params_.get('gamma')

def postModelStats(X_train,y_train):
    #linear_model = SVC(kernel = 'linear', gamma = 0.5, random_state = 0)
    linear_model = SVC(kernel = 'linear')
    linear_model.fit(X_train, y_train)
    
    #print(linear_model.coef_)
    selector = RFE(linear_model, 30, step=1)
    selector = selector.fit(X_train, y_train)
    
    top_features=np.where(selector.support_)[0]
    print(top_features)
    #print(selector.ranking_)
    return top_features

def crossValidate(X, y, model):
    k_folds = 5
    scores = cross_val_score(model, X, y, cv = k_folds)
    return scores