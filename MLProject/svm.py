import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder, StandardScaler
import colorsys
from collections import Counter
from colors import rgb, hsv, hex
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics, grid_search
from sklearn.grid_search import GridSearchCV

def grid_search(X,y):
    C_array = [0.001, 0.01, 0.1, 1, 10]
    gamma_array = [0.001, 0.01, 0.1, 1]
    hyperparameters = {'C': C_array, 'gamma' : gamma_array}
    grid_search = GridSearchCV(SVC(kernel='rbf'), hyperparameters, cv=10)
    grid_search.fit(X, y)
    return grid_search.best_params_.get('C'),grid_search.best_params_.get('gamma')

def processColor(color,most_fequent_color):
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
    h, s, v = 360 * h, 100 * s, 100 * v
    
    return h, s, v
   
def colorCode(column):
    # where there is no color value in column, replace it with the most common color value
    column=column.astype(str)
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

def preprocessData(X,y):
    #remove rows from both X and y, where gender is not specified 
    #index=np.argwhere(np.logical_and(np.logical_and(y!='male',y!='female'),y!='brand'))
    index=np.argwhere(np.logical_and(y!='male',y!='female'))
    y=np.delete(y,index,axis=0)
    X=np.delete(X,index,axis=0)
    

    #LabelEncoder is used to transform categorical variable(i.e 'male','female' & 'brand'), to 
    # numerical variable (0, 1 & 2)
    y=LabelEncoder().fit_transform(y)
    
    #remove rows from X & y with gender confidence less than some threshold
    #Here the assumption is that such rows might affect the model badly
    #need to look into this ...
    gender_confidence_threshold=0.6
    index=np.argwhere(X[:,0]<gender_confidence_threshold) 
    y=np.delete(y,index,axis=0)
    X=np.delete(X,index,axis=0)

    #remove rows from X and y with confidence in profile less than some threshold
    #Here the assumption is that such rows might affect the model badly
    #need to look into this ...
    profile_confidence_threshold=0.6
    index=np.argwhere(X[:,1]<profile_confidence_threshold) 
    y=np.delete(y,index,axis=0)
    X=np.delete(X,index,axis=0)
    
    # reomove unusefull column like profile confidence and gender confidence now whcih 
    # do not add any value as model features 
    X=np.delete(X,0,axis=1) #gender confidence column removed
    X=np.delete(X,0,axis=1) #profile confidence column removed
    
    #color coding
    #colorCode function assigns values between 1,2,3,4 according to its hue value
    #colorCode function needs to be improved in future
    X[:,1:2]= colorCode(X[:,1:2])
    X[:,3:4]= colorCode(X[:,3:4])
    #at this point we have color columns' values in category 1,2,3,4 according to their hue values
    #such categorical data needs to be handled by adding dummy columns to represent each category
    #OneHotEncoder is he class which converts single column into 3(or 4) different columns for each category
    #.. containing value 0 or 1.
    encoder_color=OneHotEncoder(categorical_features=[1,3])
    X=encoder_color.fit_transform(X).toarray()
    
    #Scaling
    #fit all the features between range of -1 & 1, to avoid overemphasize on a particular feature
    standardScalar=StandardScaler()
    X=standardScalar.fit_transform(X)
    
    #preprocessing completed
    return X,y

    
def main():
    #5:gender, 6:gender_confidence, 8:confidence in profile, 10:description, 11:no of favourited tweets,
    #13:link color, 14:name, 17:retweet count, 18:sidebar color, 21:tweet count
    
    dataset = pd.read_csv('dataset.csv',encoding = "latin1", usecols=(5,6,8,11,13,17,18,21))
    #divide into dependent and independent variables 
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values
    # data preprocessing to clean and arrange data.
    X,y=preprocessData(X,y)
    
    #after preprocessing: split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    
    #fit the data to the classifier
    svc = SVC(kernel = 'rbf',gamma=0.5, random_state = 0)
    svc.fit(X_train, y_train)
    
    # predict the test data using the model
    y_pred = svc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    #find accuracy percentage
    print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
    
    #hyperparameter tuning through grid search 
    best_C,best_gamma=grid_search(X,y)
    print('Predicted best hyperparameters through hyperparameter-tuning',best_C,best_gamma)
    
if __name__ == '__main__':
  main()

