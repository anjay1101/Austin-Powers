#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# initializing dataset
data_train = pd.read_csv("training.csv")
data_test = pd.read_csv("test.csv")


# preliminary analysis
def sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}


def add_col(df, name):
    # add a new column with zeroes
    df[name] = pd.Series(0, index=df.index)
    return df


def top_entries(d, n):
    # return n top entries in dictionary
    d = sort_dict(d)
    return list(d.items())[:20]


def normalize_occurences(word_occurences, articles):
    # convert total occurences to percentage occurence
    return {key: round(value / articles.shape[0], 2) for (key, value) in word_occurences.items()}


def get_top_words(data_train, topic, n):
    articles = data_train[data_train['topic'] == topic]

    word_count = {}  # map each word to how often it appears in the topic's articles

    word_occurences = {}  # map each word to how many articles it appears in

    unique_words = set()

    for i in articles.index:
        words = data_train['article_words'][i].split(',')

        for word in words:
            word_count.setdefault(word, 0)
            word_count[word] += 1

        unique_words = set(words)
        unique_words = unique_words.union(unique_words)

        for word in unique_words:
            word_occurences.setdefault(word, 0)
            word_occurences[word] += 1

    word_occurences = normalize_occurences(word_occurences, articles)

    return top_entries(word_count, n), top_entries(word_occurences, n), unique_words


topics = set(data_train.topic.values)

all_unique_words = set()

# getting the top words by topic
for topic in topics:
    top_words_by_count, top_words_by_occurences, unique_words = get_top_words(data_train, topic, 20)

    all_unique_words = all_unique_words.union(unique_words)

# pre-processing

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


def shuffle():
    # reorder the data_train randomly, Google recommends this as a best practice
    return data_train.sample(frac=1).reset_index(drop=True)


def vectorize():
    max_features = 300
    min_df = 10
    max_df = 1.
    ngram_range = (1, 2)

    v = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=True)

    train = v.fit_transform(data_train['article_words'])
    features = v.get_feature_names()

    selector = SelectKBest(f_classif, k=min(TOP_K, train.shape[1]))  # selects n best features based on f_classif metric
    selector.fit(train, data_train['topic'])

    train = selector.transform(train)

    test = v.transform(data_test['article_words'])

    return train, test, features


x_train, x_test, feature_names = vectorize()
y_train = data_train['topic']
y_test = data_test['topic']

sm = SMOTE(random_state=2)
x_data_train, y_data_train = sm.fit_sample(x_train, y_train)

#### svm

# Best: 0.724737 using {'probability': True, 'kernel': 'linear', 'gamma': 0.0001, 'degree': 3, 'C': 0.1}

# svc = svm.SVC(random_state=8)
#
# C = [0.001, 0.1, 1, 10]
#
# gamma = [0.0001, 0.001, 0.01, 0.1]
#
# degree = [1, 2, 3, 4]
# kernel = ['linear', 'rbf', 'poly']
#
# probability = [True]
#
# random_grid = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree, 'probability': probability}
#
# random_search = RandomizedSearchCV(estimator=svc, param_distributions=random_grid, n_iter=50, scoring='accuracy', cv=3,
#                                    verbose=1, random_state=8)
#
# random_search.fit(x_train, y_train)
#
# print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))

svc = svm.SVC(random_state=8)

svc.fit(x_data_train, y_data_train)

svc_pred = svc.predict(x_test)

print(balanced_accuracy_score(y_test, svc_pred))
print(precision_score(y_test, svc_pred, average='weighted'))
print(recall_score(y_test, svc_pred, average='weighted'))
print(f1_score(y_test, svc_pred, average='weighted'))