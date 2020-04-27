#!/usr/bin/env python
# coding: utf-8



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
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE

# Time the total execution.
total_time = datetime.now()


##  Feature Selection using Mutual Information

def mutual_info_select(X_train,y_train,X_test,k):
    selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train,y_train)
    return selector.transform(X_train),selector.transform(X_test)


##  Results

def results(name, model, X_train, y_train, X_test, y_test, f1):  # f1 is a bool representing whether
    pred = model.fit(X_train, y_train).predict(X_test)
    train_result = model.score(X_train, y_train)
    test_result = accuracy_score(y_test, pred)
    b_acc_score = balanced_accuracy_score(y_test, pred)

    print(name)
    print("The final score for the training database on the MNB classifier is: ", round(train_result, 4))
    print("The final score for the test database on the MNB classifier is: ", round(test_result, 4))
    print("The final balance accuracy score for the test set on the MNB classifier is: ", round(b_acc_score, 4), '\n')

    if (f1):
        print("\tPrecision: ", precision_score(y_test, pred, average=None, zero_division=0))
        print("\tRecall: ", recall_score(y_test, pred, average=None, zero_division=0))
        print("\tF1: ", f1_score(y_test, pred, average=None, zero_division=0), "\n")

    return

def counts_results():
    ## Pre-processing.
    # Pre-process the training data.
    train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()

    # Get the training data.
    train_data = pd.read_csv("training.csv")

    # Pre-process for counts
    vect = CountVectorizer()

    X_train_vect = vect.fit_transform(train_data['article_words'])
    y_train = train_data['topic'].apply(lambda x: topic_map[x])

    #for CV use
    X_full = X_train_vect
    y_full = y_train

    mnb = MultinomialNB()
    X_train_vect, X_test_vect, y_train, y_test = train_test_split(X_train_vect,y_train,test_size=0.2,random_state=42)

    # Implement Multinomial Naive Bayes (MNB) using all counts.
    results("Multinomal Naive Bayes using counts",mnb,X_train_vect,y_train,X_test_vect,y_test,0)


    #selects top 20k best features based on mutual_info metric
    X_train_mic_2, X_test_mic_2 = mutual_info_select(X_train_vect,y_train,X_test_vect,20000)
    # Implement Multinomial Naive Bayes (MNB) using counts and top 20k mutual info features.
    results("Multinomal Naive Bayes using counts and top 20k mutual info features",
            mnb,X_train_mic_2,y_train,X_test_mic_2,y_test,0)
            
            
    #selects top 10k best features based on mutual_info metric
    X_train_mic, X_test_mic = mutual_info_select(X_train_vect,y_train,X_test_vect,10000)
    # Implement Multinomial Naive Bayes (MNB) using counts and top 10k mutual info features.
    results("Multinomal Naive Bayes using counts and top 10k mutual info features",
            mnb,X_train_mic,y_train,X_test_mic,y_test,0)


    #selects top 5k best features based on mutual_info metric
    X_train_mic_3, X_test_mic_3 = mutual_info_select(X_train_vect,y_train,X_test_vect,5000)
    # Implement Multinomial Naive Bayes (MNB) using counts and top 5k mutual info features.
    results("Multinomal Naive Bayes using counts and top 5k mutual info features",
            mnb,X_train_mic_3,y_train,X_test_mic_3,y_test,0)


    #selects top 2.5k best features based on mutual_info metric
    X_train_mic_4, X_test_mic_4 = mutual_info_select(X_train_vect,y_train,X_test_vect,2500)
    #Implement Multinomial Naive Bayes (MNB) using counts and top 2.5k mutual info features
    results("Multinomal Naive Bayes using counts and top 2.5k mutual info features",
            mnb,X_train_mic_4,y_train,X_test_mic_4,y_test,0)


    #selects top 1k best features based on mutual_info metric
    X_train_mic_5, X_test_mic_5 = mutual_info_select(X_train_vect,y_train,X_test_vect,1000)
    #7.Implement Multinomial Naive Bayes (MNB) using counts and top 1k mutual info features
    results("Multinomal Naive Bayes using counts and top 1k mutual info features",
            mnb,X_train_mic_5,y_train,X_test_mic_5,y_test,0)

    # Now tune alpha for this model
    max_alph = 0
    max_score = 0
    for i in np.arange(0.0, 1.0, 0.001):
        i = round(i, 4)
        mnb = MultinomialNB(alpha=i).fit(X_train_mic_4, y_train)
        pred = mnb.predict (X_test_mic_4)
        best_score = balanced_accuracy_score(y_test, pred)
        # print("alpha = ", i, "balanced score = ", b_acc_score)
        if best_score > max_score:
            max_score = best_score
            max_alph = i
    
    X_mic, _ = mutual_info_select(X_full,y_full,X_test_vect,2500) #full X dataset with mutual_info applied 
    max_CV_score = cross_val_score(MultinomialNB(alpha=max_alph),X_mic,y_full,scoring='balanced_accuracy',cv=5)
    print("The best CV score with k=2500 was = ", max_CV_score, " when alpha = ", max_alph)



def TFIDF_results():
    ## Pre-processing.
    # Pre-process the training data.
    train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()

    sm = SMOTE(random_state=2)
    train_X, train_labels = sm.fit_sample(train_X, train_labels)

    # Get the training data.
    train_data = pd.read_csv("training.csv")

    mnb = MultinomialNB().fit(train_X, train_labels)
    pred = mnb.predict(val_X)
    before = balanced_accuracy_score(val_labels, pred)

    # exact score changes every time due to shuffling. The following should be used as an approximate reference.
    # changed by adjusting TOP_K in Preprocesssing.py

    # balanced score with k of 20000 = 0.2222678941339102
    print("balanced score with k of 20000 = 0.2222678941339102")
    # balanced score with k of 10000 = 0.2281496580290365
    print("balanced score with k of 10000 = 0.2281496580290365")
    # balanced score with k of 5000 = 0.25767887570312037
    print("balanced score with k of 5000 = 0.25767887570312037")
    # balanced score with k of 2500 = 0.2766391128164138
    print("balanced score with k of 2500 = 0.2766391128164138")
    # balanced score with k of 1000 = 0.2663739785838589
    print("balanced score with k of 1000 = 0.2663739785838589")

    sys.exit(1)
    # Now tune alpha for this model
    max_alph = 0
    max_score = 0
    for i in np.arange(0.0, 1.0, 0.001):
        i = round(i, 4)
        mnb = MultinomialNB(alpha=i).fit(train_X, train_labels)
        pred = mnb.predict(val_X)
        best_score = balanced_accuracy_score(val_labels, pred)
        if best_score > max_score:
            max_score = best_score
            max_alph = i
    print("The best score with k=2500 BEFORE alpha tuning was = ", before, " when alpha = ", 1)
    print("The best score with k=2500 AFTER alpha tuning was = ", max_score, " when alpha = ", max_alph)






counts_results()
TFIDF_results()







