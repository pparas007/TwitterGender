import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import plotly.plotly as py
import plotly.tools as tls
import matplotlib.mlab as mlab
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def plot(X, y, feature_names, index_list):
    #for i in range(0, len(X[0, :])):
    
    #Not very useful
    for i in index_list:
        print("Plotting X[:,", i, "] vs y")
        print("Skipping plotting X vs Y")
        #plotXY(X, y, i, feature_names)
    

    #for i in range(0, len(X[0, :])):
    for i in index_list:
        #for j in range(i + 1, len(X[0, :])):
        for j in index_list:
            if j <= i:
                continue
            print("Plotting X[:,", i, "] vs X[:,", j , "] vs y")
            print("Skipping plotting X vs X vs Y")
            #plotXXY(X, y, i, j, feature_names)

def plotXY(X, y, indexX, feature_names):
    # plots the data points with o for the positive examples and x for the negative examples. output is saved to file graph.png
    x = X[:, indexX]
    fig, ax = plt.subplots(figsize = (12, 8))

    index = random.sample(range(0, len(y)), 100)
    yTemp = y[index]
    xTemp = x[index]

    ax.scatter(xTemp, yTemp, c = 'r', marker = 'x', label = 'Data')
    ax.set_title(feature_names[indexX] + ' vs. Gender')
    ax.set_xlabel(feature_names[indexX])
    ax.set_ylabel('Gender')
    plt.show()

def plotXXY(X, y, indexX, indexY, feature_names):
    x1 = X[:, indexX]
    x2 = X[:, indexY]
    # plots the data points with o for the positive examples and x for the negative examples. output is saved to file graph.png
    fig, ax = plt.subplots(figsize = (12, 8))
    
    index = random.sample(range(0, len(y)), 100)
    yTemp = y[index]
    x1Temp = x1[index]
    x2Temp = x2[index]

    indexPos = yTemp > 0
    indexNeg = yTemp == 0

    ax.scatter(x1Temp[indexPos], x2Temp[indexPos], c = 'b', marker = '+', label = 'Male')
    ax.scatter(x1Temp[indexNeg], x2Temp[indexNeg], c = 'r', marker = 'x', label = 'Female')
    ax.set_title(feature_names[indexX] + ' vs. ' + feature_names[indexY])
    ax.set_xlabel(feature_names[indexX])
    ax.set_ylabel(feature_names[indexY])
    plt.show()

def trainingVsAccuracySVC(X, y, method):
    i=0
    accuracy=np.zeros((20, 2))
    for trainingSize in range (500,len(X),500):
        testSize = len(X) - trainingSize
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = trainingSize, test_size = testSize, random_state=0)
        if method == 'svc_linear':
            svc = SVC(kernel = 'linear')
        elif method == 'svc_rbf':
            svc = SVC(kernel = 'rbf', gamma = 'auto')
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        
        accuracy[i][0]=trainingSize
        accuracy[i][1]=metrics.accuracy_score(y_test, y_pred)
        i+=1
        
    index = np.argwhere(accuracy[:,0]<=1)
    accuracy = np.delete(accuracy, index, axis = 0)
    
    fig, ax = plt.subplots(figsize = (12, 8))
    #x1,x2,y1,y2=fig.axis()
    plt.axis((500,10000,0.10,1.00))
    plt.plot(accuracy[:,0], accuracy[:,1])
    ax.set_title('Training size vs. Accuracy')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Accuracy')
    plt.show()

def trainingVsAccuracyLogReg(X, y):
    i = 0
    accuracy = np.zeros((20, 2))
    for trainingSize in range (500, len(X), 500):
        testSize = len(X) - trainingSize
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = trainingSize, test_size = testSize, random_state = 0)
        logisticRegr = LogisticRegression(solver = 'lbfgs')
        logisticRegr.fit(X_train, y_train)
        y_pred = logisticRegr.predict(X_test)
        
        accuracy[i][0] = trainingSize
        accuracy[i][1] = metrics.accuracy_score(y_test, y_pred)
        i += 1
    index = np.argwhere(accuracy[:, 0] <= 1)
    accuracy = np.delete(accuracy, index, axis = 0)
    
    fig, ax = plt.subplots(figsize = (12, 8))
    #x1,x2,y1,y2=fig.axis()
    plt.axis((500, 10000, 0.10, 1.00))
    plt.plot(accuracy[:, 0], accuracy[:, 1])
    ax.set_title('Training size vs. Accuracy')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Accuracy')
    plt.show()