# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:28:41 2018

@author: 007Paras
"""

import preprocessing
import stats
import plotting
import tuning
import numpy as np
print("numpy version:", np.__version__)
import pandas as pd
print("pandasversion:", pd.__version__)
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
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re

def process(x1):
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    map={}
    max_freq=0
    f= open("stats.txt","w")
    for i in range(0,len(x1)):
        x1[i]=x1[i].lower()
        regex = re.compile('[^a-zA-Z]')
        x1[i]=regex.sub(' ', x1[i])
        token_words=word_tokenize(x1[i])

        stem_sentence=""
        for word in token_words:
            word=porter.stem(word)
            if word not in stop_words: 
                if(len(word)>2):
                    if(map.get(word)==None):
                        map[word]=1
                    else:
                        map[word]=(map.get(word)+1)
                        if(max_freq<map[word]):
                            max_freq=map[word]
 
    print(max_freq)
    for i in range(3,max_freq+1):
        for key in map:
            if(i==map[key]):
                print(key+' '+str(map[key])+'\n')
                f.write(key+' '+str(map[key])+'\n')
    f.close()    
def main():
    dataset = pd.read_csv('dataset.csv', encoding = "latin1", usecols = (5,10,19))
    dataset = dataset.replace(np.nan, '', regex=True)
    x1 = dataset.iloc[:, 1].values
    x2 = dataset.iloc[:, 2].values
    y = dataset.iloc[:, 0].values
    x1=x1+' '+x2
    process(x1)
    
    
if __name__ == '__main__':
  main()