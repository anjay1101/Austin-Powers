#!/usr/bin/env python
# coding: utf-8


# Time the total execution.
total_time = datetime.now()

## Imports
import numpy as np
import pandas as pd
import sys

from Preprocessing import prepare_data
from Preprocessing import topics_to_num

import time
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import balanced_accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 



## Pre-processing.

# Get the training data.
train_data = pd.read_csv("training.csv")

# Pre-process the training data.
train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()

# Pre-process for counts
vect = CountVectorizer()

X_train_vect = vect.fit_transform(train_data['article_words'])
y_train = train_data['topic'].apply(lambda x: topic_map[x])

X_train_vect, X_test_vect, y_train, y_test = train_test_split(X_train_vect,y_train,test_size=0.2,random_state=42)


##  Feature Selection using Mutual Information 

def mutual_info_select(X_train,y_train,X_test,k):
    selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train,y_train)
    return selector.transform(X_train),selector.transform(X_test)



#selects top 10k best features based on mutual_info metric
X_train_mic, X_test_mic = mutual_info_select(X_train_vect,y_train,X_test_vect,10000)

#selects top 20k best features based on mutual_info metric
X_train_mic_2, X_test_mic_2 = mutual_info_select(X_train_vect,y_train,X_test_vect,20000)

#selects top 5k best features based on mutual_info metric
X_train_mic_3, X_test_mic_3 = mutual_info_select(X_train_vect,y_train,X_test_vect,5000)

#selects top 2.5k best features based on mutual_info metric
X_train_mic_4, X_test_mic_4 = mutual_info_select(X_train_vect,y_train,X_test_vect,2500)

#selects top 1k best features based on mutual_info metric
X_train_mic_5, X_test_mic_5 = mutual_info_select(X_train_vect,y_train,X_test_vect,1000)


##  Results 

def results(name, model,X_train,y_train,X_test,y_test,f1): #f1 is a bool representing whether 
    pred = model.fit(X_train,y_train).predict(X_test)
    train_result = model.score(X_train,y_train)
    test_result = accuracy_score(y_test, pred)
    b_acc_score = balanced_accuracy_score(y_test,pred)
    
    print(name)
    print("The final score for the training database on the MNB classifier is: ", round(train_result, 4))
    print("The final score for the test database on the MNB classifier is: ", round(test_result, 4))
    print("The final balance accuracy score for the test set on the MNB classifier is: ", round(b_acc_score, 4), '\n')
    
    if(f1):
        print("\tPrecision: ", precision_score(y_test,pred,average=None,zero_division=0))
        print("\tRecall: ", recall_score(y_test,pred,average=None,zero_division=0))
        print("\tF1: ", f1_score(y_test,pred,average=None,zero_division=0),"\n")
    
    return 
    

# Print Results

mnb = MultinomialNB()

#1. Implement Multinomial Naive Bayes (MNB) using TFIDF scaling
results("Multinomal Naive Bayes using TFIDF scaling",mnb,train_X,train_labels,val_X,val_labels,0)

#2. Implement Multinomial Naive Bayes (MNB) using counts
results("Multinomal Naive Bayes using counts",mnb,X_train_vect,y_train,X_test_vect,y_test,0)

#3. Implement Multinomial Naive Bayes (MNB) using counts and top 10k mutual info features
results("Multinomal Naive Bayes using counts and top 10k mutual info features",
        mnb,X_train_mic,y_train,X_test_mic,y_test,0)

#4. Implement Multinomial Naive Bayes (MNB) using counts and top 20k mutual info features
results("Multinomal Naive Bayes using counts and top 20k mutual info features",
        mnb,X_train_mic_2,y_train,X_test_mic_2,y_test,0)

#5.Implement Multinomial Naive Bayes (MNB) using counts and top 5k mutual info features
results("Multinomal Naive Bayes using counts and top 5k mutual info features",
        mnb,X_train_mic_3,y_train,X_test_mic_3,y_test,0)

#6.Implement Multinomial Naive Bayes (MNB) using counts and top 2.5k mutual info features
results("Multinomal Naive Bayes using counts and top 2.5k mutual info features",
        mnb,X_train_mic_4,y_train,X_test_mic_4,y_test,1)

#7.Implement Multinomial Naive Bayes (MNB) using counts and top 1k mutual info features
results("Multinomal Naive Bayes using counts and top 1k mutual info features",
        mnb,X_train_mic_5,y_train,X_test_mic_5,y_test,0)


print('\n\nTotal execution time: ', datetime.now() - total_time)






