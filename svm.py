#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import imblearn
from collections import Counter
import altair as alt
alt.renderers.enable('altair_viewer')
alt.data_transformers.disable_max_rows()
import sys
# pre-processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# from tqdm import tqdm  # for showing progress

import time

# In[36]:


# initializing dataset
data_train = pd.read_csv("training.csv")
data_test = pd.read_csv("test.csv")
#
# # counter = Counter(data_train['topic'])
# # print(counter)
#
# under_sample = RandomUnderSampler()
#
# x_data_train, y_data_train = under_sample.fit_resample(data_train['article_words'].reshape(1, -1), data_train['topic'])
#
# # counter = Counter(y_data_train)
# #
# # print(counter)

# Data explore

# data_train['id'] = 1
# df2 = pd.DataFrame(data_train.groupby('topic').count()['id']).reset_index()
#
# bars = alt.Chart(df2).mark_bar(size=10).encode(
#     x=alt.X('topic'),
#     y=alt.Y('PercentOfTotal:Q', axis=alt.Axis(format='.0%', title='% of Articles')),
#     color='topic'
# ).transform_window(
#     TotalArticles='sum(id)',
#     frame=[None, None]
# ).transform_calculate(
#     PercentOfTotal="datum.id / datum.TotalArticles"
# )
#
# text = bars.mark_text(
#     align='center',
#     baseline='bottom',
#     #dx=5  # Nudges text to right so it doesn't appear on top of the bar
# ).encode(
#     text=alt.Text('PercentOfTotal:Q', format='.1%')
# )
#
# (bars + text).interactive().properties(
#     height=300,
#     width=700,
#     title = "% of articles in each category",
# )
# bars.show()
#




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

#     print()
#     print(topic)
#     print("top word counts")
#     print(top_words_by_count)
#
#     print()
#     print("top occurences")
#     print(top_words_by_occurences)
#     print()
#
# print("total words:", len(all_unique_words))

# In[75]:


# pre-processing

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif

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
    # v = TfidfVectorizer(
    #     preprocessor=lambda x: x,  # the preprocessor is set to be the identity function (it does nothing)
    #     tokenizer=lambda x: x.split(','),
    #     # the tokenizer (which converts a string into individual words) splits a string at ','
    #     ngram_range=NGRAM_RANGE,
    #     analyzer=TOKEN_MODE,
    #     min_df=MIN_DOCUMENT_FREQUENCY)  # we decide to use unigrams and bigrams as the google guide suggests

    ngram_range = (1, 2)
    min_df = 10
    max_df = 1.
    max_features = 300

    v = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        # norm='l2',
                        sublinear_tf=True)

    train = v.fit_transform(data_train['article_words'])
    features = v.get_feature_names()

    selector = SelectKBest(f_classif, k=min(TOP_K, train.shape[1]))  # selects n best features based on f_classif metric
    selector.fit(train, data_train['topic'])

    train = selector.transform(train)

    test = v.transform(data_test['article_words'])
    #
    # category_codes = ["ARTS CULTURE ENTERTAINMENT", "BIOGRAPHIES PERSONALITIES PEOPLE", "DEFENCE", "DOMESTIC MARKETS",
    #                   "FOREX MARKETS", "HEALTH", "MONEY MARKETS", "SCIENCE AND TECHNOLOGY", "SHARE LISTINGS", "SPORTS"]
    #
    # for Product in category_codes:
    #     features_chi2 = chi2(train, data_train['topic'] == Product)
    #     indices = np.argsort(features_chi2[0])
    #     feature_names_array = np.array(v.get_feature_names())[indices]
    #     unigrams = [v for v in feature_names_array if len(v.split(' ')) == 1]
    #     bigrams = [v for v in feature_names_array if len(v.split(' ')) == 2]
    #     print("# '{}' category:".format(Product))
    #     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    #     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    #     print("")

    return train, test, features


x_train, x_test, feature_names = vectorize()
y_train = data_train['topic']
y_test = data_test['topic']

# counter = Counter(y_train)
# print(counter)

under_sample = RandomOverSampler()

x_data_train, y_data_train = under_sample.fit_resample(x_train, y_train)

# counter = Counter(y_data_train)
#
# print(counter)

# print("x_train")
# print(x_train.shape)
# print("x_test")
# print(x_test.shape)
# print("names")
# print(feature_names)

# Use grid search for hyper-parameter tuning
# logistic = LogisticRegression()
#
# penalty = ['l1', 'l2']
#
# dual = [True, False]
#
# max_iter = [100, 110, 120, 130, 140]
#
# # c = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
#
# C = np.logspace(0, 4, 10)
#
# param_grid = dict(max_iter=max_iter, C=C, penalty=penalty)
#
# # grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1)
#
# grid = GridSearchCV(logistic, param_grid, cv=5, verbose=0)
#
# grid_result = grid.fit(x_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#### svm

# Best: 0.724737 using {'probability': True, 'kernel': 'linear', 'gamma': 0.0001, 'degree': 3, 'C': 0.1}

# svc = svm.SVC(random_state=8)

# C = [0.0001, 0.001, 0.1]
#
# gamma = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#
# degree = [1, 2, 3, 4, 5]
# kernel = ['linear', 'rbf', 'poly']
#
# probability = [True]
#
# random_grid = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree, 'probability': probability}
#
# svm = svm.SVC(random_state=8)
#
# random_search = RandomizedSearchCV(estimator=svc, param_distributions=random_grid, n_iter=50, scoring='accuracy', cv=3,
#                                    verbose=1, random_state=8)
#
# random_search.fit(x_train, y_train)
#
# print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))

svc = svm.SVC(C=0.1, kernel='linear', degree=3, gamma=0.0001)

svc.fit(x_data_train, y_data_train)

svc_pred = svc.predict(x_test)

print("Training accuracy score is: ")
print(accuracy_score(y_data_train, svc.predict(x_data_train)))

print("Test accuracy score is: ")
print(accuracy_score(y_test, svc_pred))
