import numpy as np
import pandas as pd
import sys

from Preprocessing import vectorize

import time
from datetime import datetime

# Time the total execution.
total_time = datetime.now()

# Pre-processing.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

def shuffle(data):
    # reorder the data randomly, Google recommends this as a best practice
    return data.sample(frac=1).reset_index(drop=True)

def vectorize(data):
    data = shuffle(data)
    v = TfidfVectorizer(
        preprocessor=lambda x: x,  # the preprocessor is set to be the identity function (it does nothing)
        tokenizer=lambda x: x.split(','),
        # the tokenizer (which converts a string into individual words) splits a string at ','
        ngram_range=NGRAM_RANGE,
        analyzer=TOKEN_MODE,
        min_df=MIN_DOCUMENT_FREQUENCY)  # we decide to use unigrams and bigrams as the google guide suggests

    X = v.fit_transform(data['article_words'])  # create a column for each word and compute the tfidf score for each word/document combination

    # an array representing all words,
    # if you want to know what feature the nth column of X represents just access the nth element in the array
    feature_names = v.get_feature_names()

    selector = SelectKBest(f_classif, k=min(TOP_K, X.shape[1]))  # selects n best features based on f_classif metric
    selector.fit(X, data['topic'])

    X = selector.transform(X)

    # creates a mapping from new column to old column
    # the ith column in the X created by the new selector is the col_map[i]th column in the old one
    col_map = selector.get_support(indices=True)

    # get the new feature names using the old feature names and col_map
    new_feature_names = [feature_names[col_map[i]] for i in range(min(TOP_K, X.shape[1]))] # CHANGED SOMETHING HERE!

    return X, new_feature_names


# go to the cell bellow for an explanation of the data we gathered
def print_explanation(X, feature_names):
    print("X is our input matrix it has %s rows and %s columns" % X.shape)
    print("it holds TFIDF scores (visit http://tfidf.com/ for more info)")
    print()
    print("each row represents a document")
    print("each column represents a word")
    print("thus, the value at the ith row and jth column is the tfidf score for the jth word in the ith document")
    print()
    print("feature_names tells us what word the jth column represents")
    print("to figure out what the jth word is you access feature_names[j]")
    print()
    print("hence, we use feature names to see that the 50th word is \"%s\"" % feature_names[50])
    print("very cool")
    print("now we can use X to figure out that the tfidf score of %s in the 6060th document is %s" % (feature_names[50], X[6060, 50]))
    print("nice")


# Fits the test matrix to the same size as the training matrix.
def matrix_fit(test_X, test_feature_names, train_X, train_feature_names):
    matrix_fit_time = datetime.now()
    num_test_rows = test_X.shape[0]
    new_matrix = [[0 for i in range(train_X.shape[1])] for j in range(num_test_rows)]
    # populate new_matrix.
    print("Populating matrix . . . ")
    for feature in train_feature_names:
        if feature in test_feature_names:
            #Fill the corresponding column with the tfidf's.
            train_feature_index = train_feature_names.index(feature)
            test_feature_index = test_feature_names.index(feature)
            for this_row in range(num_test_rows):
                new_matrix[this_row][train_feature_index] = test_X[this_row, test_feature_index]
        else:
            #leave that column all zeros.
            pass
    print("Populated matrix! It took ", datetime.now() - matrix_fit_time, " minutes.\n")
    return new_matrix

# Implement the models.
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Get the training data.
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("test.csv")

# Pre-process the training data.
train_X, test_X, train_feature_names = vectorize(train_data, test_data)

# 1) Support Vector Classification (SVC)
svc_time = datetime.now()
svc = SVC(gamma='auto')
# Train the SVC.
svc.fit(train_X, train_data['topic'])
# Test our SVC results.
train_result = svc.score(train_X, train_data['topic'])
print("The final score for the training database on the SVC classifier is: ", round(train_result, 4))
test_result = svc.score(test_X, test_data['topic'])
print("The final score for the test database on the SVC classifier is: ", round(test_result, 4))
print("To classify with SVC it took ", datetime.now() - svc_time, '\n')


# 2) Random Forest Classifier (RFC)
rfc = RandomForestClassifier(max_depth=2, random_state=0)
# Train the RFC.
rfc.fit(train_X, train_data['topic'])
# Test our RFC results.
train_result = rfc.score(train_X, train_data['topic'])
print("The final score for the training database on the RFC classifier is: ", round(train_result, 4))
test_result = rfc.score(test_X, test_data['topic'])
print("The final score for the test database on the RFC classifier is: ", round(test_result, 4), '\n')


# 3) Multinomial Naive Bayes (MNB)
mnb = MultinomialNB()
# Train the MNB classifier.
mnb.fit(train_X, train_data['topic'])
# Test our MNB results.
train_result = mnb.score(train_X, train_data['topic'])
print("The final score for the training database on the MNB classifier is: ", round(train_result, 4))
test_result = mnb.score(test_X, test_data['topic'])
print("The final score for the test database on the MNB classifier is: ", round(test_result, 4), '\n')


print('\n\nTotal execution time: ', datetime.now() - total_time)
