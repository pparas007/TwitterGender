import preprocessing
import stats
import plotting
import tuning
import numpy as np
print("numpy version:", np.__version__)
import pandas as pd
print("pandasversion:", pd.__version__)
import scikitplot as skplt
import matplotlib.pyplot as plt

import sklearn
print("sklearn version:", sklearn.__version__)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics #, grid_search
from sklearn.metrics import classification_report
from sklearn.svm.libsvm import predict_proba
from sklearn.linear_model import LogisticRegression

def process(name, method, X, y, X_train, y_train, X_test, y_test):
    print('\n\n#################### ' + name + ' - Report ####################\n')
    #print('\n\n#################### Support Vector Machines - Report ####################\n')
    clf = None
    
    if method == 'lr':
        ### Logistic Regression
        clf = LogisticRegression(solver = 'lbfgs')
    elif method == 'svc_linear':
        #fit the data to the classifier
        clf = LinearSVC(C = 1, max_iter=5000)
    elif method == 'svc_rbf':
        #fit the data to the classifier
        clf = SVC(kernel = 'rbf', gamma = 0.01, C = 1, probability = True)
    elif method == 'knc':
        #fit the data to the classifier
        clf = KNeighborsClassifier(n_neighbors = 3)

    scores = tuning.crossValidate(X_train, y_train, clf)
    print("K-fold Cross-Validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf.fit(X_train, y_train)

    # predict the test data using the model
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    #find accuracy percentage
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    
          #find f1 score
    print("F1 score: ",metrics.f1_score(y_test, y_pred, average="macro"))
    
    skplt.metrics.plot_roc_curve(y_test, y_pred)
    plt.show()
 
    print('\nAverage confidence in prediction: ')

    if method == 'knc':
        conf = clf.score(X_test, y_test)
        print(conf)
    else:
        conf = clf.decision_function(X_test)
        print(np.sum(abs(conf)) / len(conf))
    print('\nConfusion Matrix:')
    print('\tMale\tFemale\nMale\t',cm[0, 0],'\t',cm[0, 1],'\t\nFemale\t',cm[1, 0],'\t',cm[1, 1],'\n')
    
    target_names = ['male','female']
    print(classification_report(y_test, y_pred, target_names = target_names))

    if method == 'lr':
        plotting.trainingVsAccuracyLogReg(X, y)
    elif method == 'svc_linear' or method == 'svc_rbf':
        plotting.trainingVsAccuracySVC(X, y, method)
    #elif method == 'knc':
        #plotting.trainingVsAccuracy???(X, y)

    #hyperparameter tuning through grid search 
    #best_C, best_gamma = tuning.grid_search(X, y)
    #print('Predicted best hyperparameters through hyperparameter-tuning', best_C, best_gamma)

def main():
    #14:name
    names_dataset = dataset = pd.read_csv('dataset.csv', encoding = "latin1", usecols = (14,))
    #5:gender, 6:gender_confidence, 8:confidence in profile, 10:description, 11:no of favourited tweets,
    #13:link color, 17:retweet count, 18:sidebar color, 19:tweet text, 21:tweet count
    dataset = pd.read_csv('dataset.csv', encoding = "latin1", usecols = (5, 6, 8, 10, 11, 13, 17, 18, 19, 21))
    
    #words = pd.read_csv('manually_filtered_stats.csv', encoding = "latin1", usecols = (0,))
    #divide into dependent and independent variables 
    #6:gender_confidence, 8:confidence in profile, 10:description, 11:no of favourited tweets,
    #13:link color, 17:retweet count, 18:sidebar color, 19:tweet text, 21:tweet count
    X = dataset.iloc[:, 1:]
    #5:gender
    y = dataset.iloc[:, 0]
    
    #10:description, 19:tweet text
    description_and_tweet = pd.read_csv('dataset.csv', encoding = "latin1", usecols = (10, 19))
    description_and_tweet = description_and_tweet.replace(np.nan, '', regex=True)
    x1 = description_and_tweet.iloc[:, 0].values
    x2 = description_and_tweet.iloc[:, 1].values
    description_and_tweet_combined = x1+' '+x2
    
    #swap # of favorite tweets and link_color column
    #link_color_col = numpy.copy(X[:, 1])
    #X[:, 1] = X[:, 0]
    #X[:, 0] = link_color_col

    #swap # of favorite tweets and sidebar_color column
    #sidebar_color_col = numpy.copy(X[:, 3])
    #X[:, 3] = X[:, 1]
    #X[:, 1] = sidebar_color_col

    #Might need to be updated/reviewed because of change of columns
    stats.stats(X, y)

    X, y = preprocessing.preprocessData(X.values, y.values, names_dataset.values, description_and_tweet_combined)
    
    """
    #84 columns
    X_feature_names = [
        16 dummy features extracted from link color     0-15
        16 dummy features extracted from sidebar color  16-31
        '# of favorite tweets',                         32      
        '# of retweets',                                33
        '# of tweets',                                  34
        5 features extracted from description,          35-39
        5 features extracted from tweet text,           40-44
        3 features extracted from name (dummy),         45-47
        36 features extracted from tweet text           48-83
    ]
    """
    
    #Might need to be updated/reviewed because of change of columns
    #It doesn't seem it is affected by order of columns, but with dummy variables, it might generate too many plots
    
    feature_names = ['lk_red', 'lk_red-orange', 'lk_orange-brown', 'lk_orange-yellow', 'lk_yellow', 'lk_yellow-green', 'lk_green', 
        'lk_green-cyan', 'lk_cyan', 'lk_cyan-blue', 'lk_blue', 'lk_blue-magenta', 'lk_magenta', 'lk_magenta-pink', 'lk_pink', 'lk_pink-red',
        'sb_red', 'sb_red-orange', 'sb_orange-brown', 'sb_orange-yellow', 'sb_yellow', 'sb_yellow-green', 'sb_green', 
        'sb_green-cyan', 'sb_cyan', 'sb_cyan-blue', 'sb_blue', 'sb_blue-magenta', 'sb_magenta', 'sb_magenta-pink', 'sb_pink', 'sb_pink-red',
        '# of favorite tweets', '# of retweets', '# of tweets',
        '# of hashtags in description', 'URLs present in description', '# of emoticons used in description', 'length of description', '# of mentions in description',
        '# of hashtags in tweet text', 'URLs present in tweet text', '# of emoticons used in tweet text', 'length of tweet text', '# of mentions in tweet text',
        'feature 1 from name', 'feature 2 from name', 'feature 3 from name',
        'women word_freq', 'bitch word_freq', 'nation word_freq', 'tec  word_freq', 'season word_freq',
        'hair word_freq', 'dad word_freq', 'player word_freq', 'cat word_freq', 'polit word_freq',
        'blogger word_freq', 'radio word_freq', 'pushawardslizquen word_freq', 'boy word_freq', 'author word_freq',
        'footbal word_freq', 'kid word_freq', 'travel word_freq', 'social word_freq', 'heart word_freq',
        'vote word_freq', 'food word_freq', 'guy word_freq', 'beauti word_freq', 'lover word_freq',
        'via word_freq', 'writer word_freq', 'artist word_freq', 'man word_freq', 'sport word_freq',
        'fuck word_freq', 'girl word_freq', 'fan word_freq', 'game word_freq', 'love word_freq',
        'weather word_freq'
        ]

    #[ 0  8 11 12 14 15 16 22 24 25 32 34 35 38 44 46 47 48 53 60 63 69 71 72 76 77 79 80 81 82]
    index_temp = [0, 8, 11, 12, 14, 15, 16, 22, 24, 25, 32, 34, 35, 38, 44, 46, 47, 48, 53, 60, 63, 69, 71, 72, 76, 77, 79, 80, 81, 82]
    print("first line: ", X[0, :])
    plotting.plot(X, y, feature_names, index_temp)

    #Might need to be updated/reviewed because of change of columns
    #This is happening over the entire dataset and should only happen on the continuous variables
    X = preprocessing.scale(X)
    
    #select top features using Reverse Feature Elimination
    #not affected by order of columns
    top_features = tuning.postModelStats(X, y)

    #print("top_features:", top_features)
    #[ 0  8 11 12 14 15 16 22 24 25 32 34 35 38 44 46 47 48 53 60 63 69 71 72 76 77 79 80 81 82]
    
    X = X[:, top_features]
    
    #after preprocessing: split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    #Each run will take approximately 10 to 15 minutes given the number of features used in the model
    process('Support Vector Classifier - RBF Kernel', 'svc_rbf', X, y, X_train, y_train, X_test, y_test)
    process('Logistic Regression', 'lr', X, y, X_train, y_train, X_test, y_test)
    process('Support Vector Classifier - Linear Kernel', 'svc_linear', X, y, X_train, y_train, X_test, y_test)
    process('K nearest Classifier', 'knc', X, y, X_train, y_train, X_test, y_test)
    

if __name__ == '__main__':
  main()