import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder, StandardScaler
import colorsys
from collections import Counter
from colors import rgb, hsv, hex
from sklearn.svm import SVC
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import re
#for outlier detection, box plot
import seaborn as sns
import gender_guesser.detector as gender

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
    
def correctColorFormat(X, col_index):
    #this function first processes different length hex values to make them 6-length
    lenVectorized = np.vectorize(len)

    #this function need to be improved to properly understand & handle 2,5,length values
    # currently all doubtful values are replaced with most frequently occuring color 
    index = np.argwhere(np.logical_or(lenVectorized(X[:, col_index]) == 3, lenVectorized(X[:, col_index]) == 4))
    if index.size > 0:
        for i in np.nditer(index):
            X[i, col_index] = X.item((i, col_index))[0] * 2 + X.item((i, col_index))[1] * 2 + X.item((i, col_index))[2] * 2
            
    index = np.argwhere(lenVectorized(X[:, col_index]) == 2)
    if index.size > 0:
        for i in np.nditer(index):
            X[i, col_index] = X.item((i, col_index))[0] * 3 + X.item((i, col_index))[1] * 3
    
    index = np.argwhere(lenVectorized(X[:, col_index]) == 1)
    if index.size > 0:
        for i in np.nditer(index):
            X[i, col_index] = X.item((i, col_index))[0] * 6
        
    return X

def convertToHLS(color):
    #this function first processes different length hex values to make them 6-length
    
    #separate r,g,b and convert them to integer from hex
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    
    """
    #convert rgb to hsv:  copied from internet
    h, s, v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
    h, s, v = 360 * h, 100 * s, 100 * v
    """

    #convert to hls, a different color space model than hsv
    h, l, s = colorsys.rgb_to_hls(r / 255., g / 255., b / 255.)
    h, l, s = 360 * h,  100 * l, 100 * s
    return h, l, s

def colorCode(column):
    #Simple wheel
    #color_borders = [60, 180, 300]
    #red, green, blue

    #Complex wheel
    #http://www.workwithcolor.com/red-color-hue-range-01.htm
    color_borders = [10, 20, 40, 50, 60, 80, 140, 169, 200, 220, 240, 280, 320, 330, 345, 355]
    #red, red-orange, orange-brown, orange-yellow, yellow,
    #yellow-green, green, green-cyan, cyan, cyan-blue
    #blue, blue-magenta, magenta, magenta-pink, pink,
    #pink-red

    for i in range(0, (len(column))):
        h, l, s = convertToHLS(column[i])
        h = int(h)
        column[i] = h
        #hue value ranges from 0-360
        #divide it into 3 parts and put 1, 2 or 3 in color column 
        
        #General logic for any color wheel; replaces code commented below
        if h > color_borders[-1]:
            column[i] = 0
        else:
            j = 0
            for border in color_borders:
                if h <= border:
                    column[i] = j
                    break
                j += 1
        
        """
        #Simple
        if (h <= 60 or h > 300):
            column[i] = 0
        elif (h <= 180):
            column[i] = 1
        elif (h <= 300):
            column[i] = 2
        """
    #column=np.reshape(column,(len(column),1))
    return (column)

#dealing with inconsistent data
def cleanData(X, y, names, description_and_tweet_combined):
    #remove rows from both X and y, where gender is not specified
    total = len(y)
    index = np.argwhere(np.logical_and(y != 'male', y != 'female'))
    print("# rows removed because of gender is not male nor female:", len(index), ", percentage removed: ", round(len(index) * 100 / total, 2), "%")
    y = np.delete(y, index, axis = 0)
    X = np.delete(X, index, axis = 0)
    #remove rows from the other two feature groups too
    names = np.delete(names, index, axis = 0)
    description_and_tweet_combined = np.delete(description_and_tweet_combined, index, axis = 0)
    print("# rows remaining:", len(y))
    
    lenVectorized = np.vectorize(len)
    #remove rows from X and y with link color that have invalid values
    index = np.argwhere(np.logical_or(lenVectorized(X[:, 4]) < 6, lenVectorized(X[:, 4]) > 6))
    print("# rows that could be removed because their link color length is different than 6:", len(index), ", percentage potentially removed: ", round(len(index) * 100 / total, 2), "%")
    
    #salvaging some records by correcting the hex RGB color format
    correctColorFormat(X, 4)

    lenVectorized = np.vectorize(len)
    #remove rows from X and y with link color that have invalid values
    index = np.argwhere(np.logical_or(lenVectorized(X[:, 4]) < 6, lenVectorized(X[:, 4]) > 6))
    print("# rows removed because their link color length is different than 6:", len(index), ", percentage removed: ", round(len(index) * 100 / total, 2), "%")
    y = np.delete(y, index, axis = 0)
    X = np.delete(X, index, axis = 0)
    #remove rows from the other two feature groups too
    names = np.delete(names, index, axis = 0)
    description_and_tweet_combined = np.delete(description_and_tweet_combined, index, axis = 0)
    print("# rows remaining:", len(y))

    index = np.argwhere(np.logical_or(lenVectorized(X[:, 6]) < 6, lenVectorized(X[:, 6]) > 6))
    print("# rows that could be removed because their sidebar color length is different than 6:", len(index), ", percentage potentially removed: ", round(len(index) * 100 / total, 2), "%")
    
    #salvaging some records by correcting the hex RGB color format
    correctColorFormat(X, 6)

    index = np.argwhere(np.logical_or(lenVectorized(X[:, 6]) < 6, lenVectorized(X[:, 6]) > 6))
    print("# rows removed because their sidebar color length is different than 6:", len(index), ", percentage removed: ", round(len(index) * 100 / total, 2), "%")
    y = np.delete(y, index, axis = 0)
    X = np.delete(X, index, axis = 0)
    #remove rows from the other two feature groups too
    names = np.delete(names, index, axis = 0)
    description_and_tweet_combined = np.delete(description_and_tweet_combined, index, axis = 0)
    print("# rows remaining:", len(y))

    return X, y, names, description_and_tweet_combined

#filter data
def filterData(X, y, names, description_and_tweet_combined):
    total = len(y)
    
    #remove rows from X & y with gender confidence less than some threshold
    #Here the assumption is that such rows might affect the model badly
    #need to look into this ...
    gender_confidence_threshold = 0.6
    index = np.argwhere(X[:, 0] < gender_confidence_threshold)
    print("# rows removed because their gender confidence is below threshold of", gender_confidence_threshold, ": ", len(index), ", percentage removed: ", round(len(index) * 100 / total, 2), "%")
    y = np.delete(y, index, axis = 0)
    X = np.delete(X, index, axis = 0)
    #remove rows from the other two feature groups too
    names = np.delete(names, index, axis = 0)
    description_and_tweet_combined = np.delete(description_and_tweet_combined, index, axis = 0)
    print("# rows remaining:", len(y))

    #remove rows from X and y with confidence in profile less than some threshold
    #Here the assumption is that such rows might affect the model badly
    #need to look into this ...
    profile_confidence_threshold = 0.6
    index = np.argwhere(X[:, 1] < profile_confidence_threshold) 
    print("# rows removed because their profile confidence is below threshold of:", profile_confidence_threshold, ": ", len(index), ", percentage removed: ", round(len(index) * 100 / total, 2), "%")
    y = np.delete(y, index, axis = 0)
    X = np.delete(X, index, axis = 0)
    #remove rows from the other two feature groups too
    names = np.delete(names, index, axis = 0)
    description_and_tweet_combined = np.delete(description_and_tweet_combined, index, axis = 0)
    print("# rows remaining:", len(y))
    
    #remove irrelevant columns like profile confidence and gender confidence which 
    #do not add any value as model features 
    X = np.delete(X, 0, axis = 1) #gender confidence column removed
    X = np.delete(X, 0, axis = 1) #profile confidence column removed

    #process description and tweet columns
    new_columns1 = processTextColumn(X[:, 0])
    new_columns2 = processTextColumn(X[:, 5])
    #adding 5 colums, such as count of hashtag used, as features extracted from description
    X = np.concatenate((X, new_columns1), axis = 1)
    #adding 5 colums, such as count of hashtag used, as features extracted from tweet text
    X = np.concatenate((X, new_columns2), axis = 1)
    X = np.delete(X, 0, axis = 1) #description column removed
    X = np.delete(X, 4, axis = 1) #tweet text column removed
    
    """
    #15 columns
    X_feature_names = [
        '# of favorite tweets',                 
        'Link color hex value',
        '# of retweets',                
        'Sidebar color hex value',
        '# of tweets',                          
        '# of hashtags in description',         
        'URLs present in description',          
        '# of emoticons used in description',   
        'length of description',                
        '# of mentions in description'          
        '# of hashtags in tweet text',         
        'URLs present in tweet text',          
        '# of emoticons used in tweet text',   
        'length of tweet text',                
        '# of mentions in tweet text'          
    ]
    """
    return X, y, names, description_and_tweet_combined


def processTextColumn(column):
    emoticons = [line.rstrip('\n') for line in open('emoticons.txt')]
    
    #0:count of hashtag used, 1:are urls used, 2:count of emoticons used, 3:length of profile description,
    #4:count of @ used
    new_columns = np.ndarray(shape = (len(column), 5), dtype = int)
    
    for i in range(0, (len(column))):
        description_string = str(column[i])
        if (len(description_string) != 0):
            hashTags = description_string.count('#')
            new_columns[i, 0] = hashTags
            urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', description_string)
            if (len(urls) != 0):
                new_columns[i, 1] = 1
            else:
                new_columns[i, 1] = 0
            emoticons_count = 0
            for emoticon in emoticons:
                if(emoticon in description_string):
                    emoticons_count += 1
            new_columns[i, 2] = emoticons_count
            new_columns[i, 3] = len(description_string)
            at = description_string.count('@')
            new_columns[i, 4] = at
    return new_columns

def processNamesColumn(column):
    new_columns = np.ndarray(shape = (len(column), 1), dtype = int)
    gender_detector = gender.Detector()
    
    for i in range(0, (len(column))):
        a_name = str(column[i])
        started=False;
        a_word=""
        new_columns[i, 0] = 0
        
        regex = re.compile('[^a-zA-Z]')
        a_name=regex.sub(' ', a_name)
        a_name=re.sub(r'([A-Z])', r' \1', a_name)
        words=a_name.split()
        for word in words:
            prediction=gender_detector.get_gender(word)
            #print(word,' ',prediction)
            if(prediction=='mostly_female'):
                new_columns[i, 0] = 2
                continue
            elif(prediction=='mostly_male'):
                new_columns[i, 0] = 1
                continue
            elif(prediction=='female'):
                new_columns[i, 0] = 2
                break
            elif(prediction=='male'):
                new_columns[i, 0] = 1
                break
        #print(words,'  ',new_columns[i, 0])
        
    return new_columns

def processDescriptionAndTweetCombined(description_and_tweet_combined):
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    map={}
    max_freq = 0
    dict={}
    #with open('manually_filtered_stats.txt') as f:
        #lines = f.readlines()
    with open('manually_filtered_stats_advanced.txt') as f:
        c=0
        for line in f:
            line=line.strip().split(' ')[0]
            dict[line]=c
            c=c+1

    new_columns = np.ndarray(shape = (len(description_and_tweet_combined), len(dict)), dtype = int)
    
    for i in range(0,len(description_and_tweet_combined)):
        description_and_tweet_combined[i]=description_and_tweet_combined[i].lower()
        regex = re.compile('[^a-zA-Z]')
        description_and_tweet_combined[i]=regex.sub(' ', description_and_tweet_combined[i])
        token_words=word_tokenize(description_and_tweet_combined[i])

        for word in token_words:
            word=porter.stem(word)
            if(dict.get(word)!=None):
                new_columns[i][dict.get(word)]=new_columns[i][dict.get(word)]+1
    
    
    #for i in range(0,len(description_and_tweet_combined)):
     #   for j in range(0,len(dict)):
      #      print(new_columns[i][j], end = '')
        #print()
    return new_columns

def transformData(X, y, names, description_and_tweet_combined):
    print('Converting link color and sidebar color to categories of hue')
    #color coding
    #colorCode transforms a RGB hex value into a color index, given a color wheel
    #1:link color, 3:sidebar color
    X[:, 1] = colorCode(X[:, 1])
    X[:, 3] = colorCode(X[:, 3])

    """
    #15 columns
    X_feature_names = [
        '# of favorite tweets',                 
        'Link color index',
        '# of retweets',                
        'Sidebar color index',
        '# of tweets',                          
        '# of hashtags in description',         
        'URLs present in description',          
        '# of emoticons used in description',   
        'length of description',                
        '# of mentions in description'          
        '# of hashtags in tweet text',         
        'URLs present in tweet text',          
        '# of emoticons used in tweet text',   
        'length of tweet text',                
        '# of mentions in tweet text'          
    ]
    """

    #names
    gender_Columns_deduced_from_names = processNamesColumn(names[:, 0])
    encoder_gender = OneHotEncoder(categorical_features = [0])
    #Becomes a 3 columns
    gender_Columns_deduced_from_names = encoder_gender.fit_transform(gender_Columns_deduced_from_names).toarray()
    #print("shape: gender_Columns_deduced_from_names: ", gender_Columns_deduced_from_names.shape)
    X = np.concatenate((X, gender_Columns_deduced_from_names), axis = 1)
    
    #description_and_tweet_combined
    #Becomes a 36 columns, representing the word frequency/word count for each of the words listed 
    #on manually_filtered_stats_advanced.txt from the description and tweet text combined
    words_columns = processDescriptionAndTweetCombined(description_and_tweet_combined)
    #print("shape: words_columns: ", words_columns.shape)
    X = np.concatenate((X, words_columns), axis = 1)
    
    """
    #54 columns
    X_feature_names = [
        '# of favorite tweets',                     0      
        'Link color hex value',                     1
        '# of retweets',                            2
        'Sidebar color hex value',                  3
        '# of tweets',                              4
        5 features extracted from description,      5-9
        5 features extracted from tweet text,       10-14
        3 features extracted from name (dummy),     15-17
        36 features extracted from tweet text       18-53
    ]
    """
    return X, y

def boxplot_metrics(X, col_index):
    Q1 = np.quantile(X[:, col_index], 0.25)
    Q3 = np.quantile(X[:, col_index], 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound, IQR, Q1, Q3

def removeOutliers(X, y, cols, feature_names):
    total = len(y)
    for col_index in cols:
        sns.boxplot(x=X[:, col_index].astype(float))
        lower_bound, upper_bound, IQR, Q1, Q3 = boxplot_metrics(X, col_index)
        print(feature_names[col_index])
        print("lower_bound: %f, upper_bound: %f, IQR: %f, Q1: %f, Q3: %f" % (lower_bound, upper_bound, IQR, Q1, Q3))
        index = np.argwhere((X[:, col_index] < (Q1 - 1.5 * IQR)) | (X[:, col_index] > (Q3 + 1.5 * IQR)))
        print("# rows removed because the", feature_names[col_index], "is outside the boxplot whiskers [", lower_bound, ": ", upper_bound, "]:", len(index), ", percentage removed: ", round(len(index) * 100 / total, 2), "%")
        y = np.delete(y, index, axis = 0)
        X = np.delete(X, index, axis = 0)
        print("# rows remaining:", len(y))
        plt.show()
    return X, y

#Convert categorical features to dummy variables
#This operation affects the order of the numpy ndarray columns
def encodeData(X, y):
    #Receives X with 54 columns
    
    y = LabelEncoder().fit_transform(y)

    #Depending on the colorCode() method, at this point we could have hue color values in columns 1 and 3, or categorical values that represent different colors
    #If we have categorical values we need to encode them with dummy variables
    #OneHotEncoder converts a categorical feature/column into n binary variables, where n is the number of categories
    #OneHotEncoder doesn't generate n-1 variables, it generates n variables
    encoder_color = OneHotEncoder(categorical_features = [1, 3])
    
    X = encoder_color.fit_transform(X).toarray()

    #new order of columns?
    #Might be possible to avoid by moving dummy variable columns to the end
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
    return X, y

# Receives a ndarray, not 'pandas.core.series.Series' object
def preprocessData(X, y, names, description_and_tweet_combined):
    if not isinstance(y, np.ndarray):
        print("y argument needs to be of type ndarray, but it currently is", type(y))
    if not isinstance(X, np.ndarray):
        print("X argument needs to be of type ndarray, but it currently is", type(X))

    print("Cleaning data")
    X, y, names, description_and_tweet_combined = cleanData(X, y, names, description_and_tweet_combined)
    print("Filtering data")
    X, y, names, description_and_tweet_combined = filterData(X, y, names, description_and_tweet_combined)
    print("Transforming data")
    X, y = transformData(X, y, names, description_and_tweet_combined)
    
    #54 columns
    feature_names = [
        '# of favorite tweets',                 #0
        'Link color hue',                       #1
        '# of retweets',                        #2
        'Sidebar color hue',                    #3
        '# of tweets',                          #4
        '# of hashtags in description',         #5
        'URLs present in description',          #6
        '# of emoticons used in description',   #7
        'length of description',                #8
        '# of mentions in description'          #9
        #5 features extracted from tweet text,       10-14
        #3 features extracted from name (dummy),     15-17
        #36 features extracted from tweet text       18-53
    ]
    
    #Outlier identification and removal
    #Link color (col 1): One of the link colors (0084B4) is so common (50% of entire dataset) than the IQR becomes 0, and thus only one value would remain after outlier removal. Hence, avoiding removal
    #No retweet or retweet count (col 2): It does vary but 0 is the most frequent occurrence, making IQR equal to 0 and the only value that will remain after outlier removal would be 0
    #Sidebar color (col 3): Should we do it?
    #No of hashtags (col 5): IQR becomes 0, and # of hashtags after outlier removal becomes 0 for all rows
    #No of emoticons (col 7): IQR becomes 0, and # of hashtags after outlier removal becomes 0 for all rows
    #Length of description (col 8): don't think we should remove
    #No of @, or ats count (col 9): IQR becomes 0
    
    #X, y = removeOutliers(X, y, [0, 1, 2, 3, 4, 5, 7, 8, 9], feature_names)
    #X, y = removeOutliers(X, y, [0, 2, 3, 4, 8], feature_names)
    X, y = removeOutliers(X, y, [0, 2, 4, 8], feature_names)
    #X, y = removeOutliers(X, y, [0, 2, 4], feature_names)

    #after this method, feature_names variable will no longer be accurate if dummy encoding occurred in the method below
    X, y = encodeData(X, y)
    
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
    #preprocessing completed
    return X, y


def scale(X):
    #Scaling
    #fit all the features between range of -1 & 1, to avoid overemphasize on a particular feature
    standardScalar = StandardScaler()
    #need to scale to speed up recursive feature elimination
    X = standardScalar.fit_transform(X)
    return X