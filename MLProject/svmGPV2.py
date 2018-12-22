import preprocessing
import utils
import stats
import plotting

import numpy as np
print("numpy version:", np.__version__)
import matplotlib.pyplot as plt
import pandas as pd
print("pandasversion:", pd.__version__)
import sklearn
print("sklearn version:", sklearn.__version__)
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder, StandardScaler
import colorsys
from collections import Counter
from colors import rgb, hsv, hex
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics #, grid_search
from sklearn.model_selection import GridSearchCV

def grid_search(X,y):
    C_array = [0.001, 0.01, 0.1, 1, 10]
    gamma_array = [0.001, 0.01, 0.1, 1]
    hyperparameters = {'C': C_array, 'gamma' : gamma_array}
    grid_search = GridSearchCV(SVC(kernel='rbf'), hyperparameters, cv=10)
    grid_search.fit(X, y)
    return grid_search.best_params_.get('C'),grid_search.best_params_.get('gamma')

"""
def processColor2(color,most_fequent_color):
    #this function first processes different length hex values to make them 6-length
    
    #this function need to be improved to properly understand & handle 2,5,length values
    # currently all doubtful values are replaced with most frequently occuring color 
    return_color=''
    if(len(color) == 6):
        return_color=color
    elif(len(color) == 2):
        return_color=color[0]+color[0]+color[0]+color[1]+color[1]+color[1]
    elif(len(color) == 3 or len(color) == 4):
        return_color=color[0]+color[0]+color[1]+color[1]+color[2]+color[2]
    else:
        return_color=most_fequent_color
    
    #separate r,g,b and convert them to integer from hex
    r,g,b=int(return_color[0:2],16),int(return_color[2:4],16),int(return_color[4:6],16)
    
    #convert rgb to hsv:  copied from internet
    h, s, v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
    #h, s, v = colorsys.rgb_to_hls(r/255., g/255., b/255.)
    h, s, v = 360 * h, 100 * s, 100 * v
    
    return h, s, v
   

def processColor(color,most_fequent_color):
    #this function first processes different length hex values to make them 6-length

    #separate r,g,b and convert them to integer from hex
    r,g,b=int(return_color[0:2],16),int(return_color[2:4],16),int(return_color[4:6],16)
    
    #convert rgb to hsv:  copied from internet
    h, l, s = colorsys.rgb_to_hls(r/255., g/255., b/255.)
    h, l, s = 360 * h,  100 * l, 100 * s
    
    return h, l, s

def colorCode2(column):
    # where there is no color value in column, replace it with the most common color value
    #print(column)
    #print('column.shape:', column.shape)
    #print('column.type:', type(column))
    #print('column[0].shape:', column[0].shape)
    #print('column[0] type:',type(column[0]))
    #print('column[0][0] type:',type(column[0][0]))
    # commenting it out, since elements are already str. If used, elements become numpy.str_
    #column=column.astype(str)
    ##print('after column.shape:', column.shape)
    #print('after column.type:', type(column))
    #print('after column[0].shape:', column[0].shape)
    #print('after column[0] type:',type(column[0]))
    #print('after column[0][0] type:',type(column[0][0]))
    #print(column)
    
    for i in range(0,(len(column))):
        h,l, s = processColor2(column[i])
        #hue value ranges from 0-360
        #divide it into 3 parts and put 1, 2 or 3 in color column 
        if(h<=120):
            column[i]=1.
        elif(h<=240):
            column[i]=2.
        elif(h<=360):
            column[i]=3.
        #else:
         #   column[i]=4.
    column=np.reshape(column,(len(column),1))
    return (column)
"""

"""
def colorCode(column):
    # where there is no color value in column, replace it with the most common color value
    #print(column)
    #print('column.shape:', column.shape)
    #print('column.type:', type(column))
    #print('column[0].shape:', column[0].shape)
    #print('column[0] type:',type(column[0]))
    #print('column[0][0] type:',type(column[0][0]))
    # commenting it out, since elements are already str. If used, elements become numpy.str_
    #column=column.astype(str)
    ##print('after column.shape:', column.shape)
    #print('after column.type:', type(column))
    #print('after column[0].shape:', column[0].shape)
    #print('after column[0] type:',type(column[0]))
    #print('after column[0][0] type:',type(column[0][0]))
    #print(column)

    map(str.strip,column)
    column=column[:,0]
    most_fequent_color=Counter(column).most_common(1)
    most_fequent_color=most_fequent_color[0][0]
    
    for i in range(0,(len(column))):
        h,s,v=processColor(column[i],most_fequent_color)
        #hue value ranges from 0-360
        #divide it into 3 parts and put 1, 2 or 3 in color column 
        if(h<=120):
            column[i]=1.
        elif(h<=240):
            column[i]=2.
        elif(h<=360):
            column[i]=3.
        #else:
         #   column[i]=4.
    column=np.reshape(column,(len(column),1))
    return (column)

def plotDataSimple(X,y):
    Xtemp = X.copy()
    ytemp = y.copy()
    #Xtemp = X.copy().values
    #ytemp = y.copy().values
    print("yplotdata", y)
    #index=np.argwhere(np.logical_and(ytemp!='male',ytemp!='female'))
    #ytemp=np.delete(ytemp,index,axis=0)
    #Xtemp=np.delete(Xtemp,index,axis=0)
    #ytemp=LabelEncoder().fit_transform(ytemp)

    # plots the data points with o for the positive examples and x for the negative examples. output is saved to file graph.png
    fig, ax = plt.subplots(figsize=(12,8))

    ## Using conditions
    male = ytemp>0
    female = ytemp<=0
    #ax.scatter(Xtemp[male,4], Xtemp[male,6], c='b', marker='o', label='Male')
    #ax.scatter(Xtemp[female,4], Xtemp[female,6], c='r', marker='x', label='Female')
    ax.scatter(Xtemp[:,2], ytemp, c='r', marker='x', label='Data')

    ax.set_xlabel('#tweets')
    ax.set_ylabel('gender')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12,8))
    #ax.scatter(Xtemp[male,4], Xtemp[male,6], c='b', marker='o', label='Male')
    #ax.scatter(Xtemp[female,4], Xtemp[female,6], c='r', marker='x', label='Female')
    ax.scatter(Xtemp[:,4], ytemp, c='b', marker='o', label='Data')

    ax.set_xlabel('#tweets')
    ax.set_ylabel('gender')
    plt.show()

def plotData(X,y):
    Xtemp = X.copy()
    ytemp = y.copy()
    #Xtemp = X.copy().values
    #ytemp = y.copy().values
    print("yplotdata", y)
    #index=np.argwhere(np.logical_and(ytemp!='male',ytemp!='female'))
    #ytemp=np.delete(ytemp,index,axis=0)
    #Xtemp=np.delete(Xtemp,index,axis=0)
    #ytemp=LabelEncoder().fit_transform(ytemp)

    # plots the data points with o for the positive examples and x for the negative examples. output is saved to file graph.png
    fig, ax = plt.subplots(figsize=(12,8))

    ## Using conditions
    male = ytemp>0
    female = ytemp<=0
    #ax.scatter(Xtemp[male,4], Xtemp[male,6], c='b', marker='o', label='Male')
    #ax.scatter(Xtemp[female,4], Xtemp[female,6], c='r', marker='x', label='Female')
    
    ax.scatter(Xtemp[male,2], Xtemp[male,4], c='b', marker='o', label='Male')
    ax.scatter(Xtemp[female,2], Xtemp[female,4], c='r', marker='x', label='Female')

    ax.set_xlabel('Test 1')
    ax.set_ylabel('Test 2')  
    plt.show()
    #fig.savefig('graph.png') 

def utilsData(X, y):
    Xtemp = X
    ytemp = y
    #dataset = pd.read_csv('dataset.csv',encoding = "latin1", usecols=(5,6,8,11,13,17,18,21))
    #divide into dependent and independent variables 
    #X = dataset.iloc[:, 1:].values
    #y = dataset.iloc[:, 0].values
    
    Xtemp = X.copy()
    ytemp = y.copy()

    #There are 20050 records
    print("Stats for y: ")
    print(ytemp.describe(include='all'))
    print(ytemp.value_counts(dropna=False))
    print(" ")
    print("Stats for X: ")
    print(Xtemp.describe(include='all'))
    #print(Xtemp.unique())
    for i in range(0,(len(Xtemp.columns))):
        print("column", Xtemp.iloc[:,i].name, ", NaN count:", Xtemp.iloc[:,i].isna().sum())
        if i == 3 or i == 5:
            #print(Xtemp.iloc[:,i].unique())
            print("skipping")
        else:
            print(Xtemp.iloc[:,i].value_counts(bins=10))
        #if pd.api.types.is_categorical_dtype(Xtemp.iloc[:,i]):
        #    print("is categorical")
        #    print(Xtemp.iloc[:,i].unique())
        #else:
        #    print("is NOT categorical")
        #    print(Xtemp.iloc[:,i].value_counts(bins=10))
        #print(Xtemp.iloc[:,i].unique())
        #print(Xtemp.iloc[:,i].value_counts(dropna=False))
    #num retweets
    #gpv_count, gpv_division = np.histogram(Xtemp.iloc[:,i], bins = [0,1,2,3,4,5,6,7,8,9])
    #Xtemp.iloc[:,i].hist(bins=gpv_division)
    #print(gpv_division)

    #print(type(Xtemp.iloc[:,3].hist(bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    #print(type(Xtemp.iloc[:,i].hist(bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).figure))
    #Xtemp.iloc[:,3].hist(bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).figure
    
    cuts = pd.cut(Xtemp.iloc[:,4], [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(cuts.value_counts())
    print("cuts end")
    #print(type(cuts.value_counts()))
    #print(type(cuts.value_counts().plot(kind='bar')))

    #.figure.show()
    #Xtemp.iloc[:,4].plot.hist(grid=True, bins= [0, 10], rwidth=0.9, color='#607c8e')
    Xtemp.iloc[:,4].plot.hist(grid=True, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rwidth=0.9, color='#607c8e')
    plt.title('Title')
    plt.xlabel('#retweets')
    plt.ylabel('count')
    plt.grid(axis='y', alpha=1)
    #plt.show()
    

    #column = column.astype('str')
    #mask = (column.str.len() != 6)

    #lengths = column.str.len()
    #print(lengths.value_counts())
    column = Xtemp.iloc[:,3]
    linkColorStats = column.str.len().value_counts().to_frame(name='count')
    linkColorStats.rename_axis("link_color_len", inplace=True)
    total = linkColorStats ['count'].sum()
    linkColorStats ['perc'] = linkColorStats ['count'] * 100 / total
    print(linkColorStats)

    column = Xtemp.iloc[:,5]
    sidebarColorStats = column.str.len().value_counts().to_frame(name='count')
    sidebarColorStats.rename_axis("sidebar_color_len", inplace=True)
    total = sidebarColorStats['count'].sum()
    sidebarColorStats ['perc'] = sidebarColorStats ['count'] * 100 / total
    print(sidebarColorStats)
    #plotData(Xtemp, ytemp)
"""

def main():
    #5:gender, 6:gender_confidence, 8:confidence in profile, 10:description, 11:no of favourited tweets,
    #13:link color, 14:name, 17:retweet count, 18:sidebar color, 21:tweet count
    #                                                                  (5, 6, 8, 11, 13, 17, 18, 21))
    dataset = pd.read_csv('dataset.csv',encoding = "latin1", usecols = (5, 6, 8, 10, 11, 13, 17, 18, 21))
    #divide into dependent and independent variables 
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]

    stats.stats(X, y)

    #print("help")
    #print(type(y.values))
    #print(type(X.values))
    print("printing")

    X, y = preprocessing.preprocessData(X.values, y.values)

    plotting.plot(X, y)

    """
    utilsData(X, y)
    
    #turning it into numpy ndarrays
    X = X.values
    y = y.values

    # data preprocessing to clean and arrange data.
    X,y=preprocessData(X,y)
    #plotData(X, y)

    plotDataSimple(X, y)
    """
    #after preprocessing: split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    #fit the data to the classifier
    svc = SVC(kernel = 'rbf',gamma=0.5, random_state = 0)
    svc.fit(X_train, y_train)

    # predict the test data using the model
    y_pred = svc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    #find accuracy percentage
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    #hyperparameter tuning through grid search 
    #best_C, best_gamma = grid_search(X, y)
    #print('Predicted best hyperparameters through hyperparameter-tuning', best_C, best_gamma)

if __name__ == '__main__':
  main()


