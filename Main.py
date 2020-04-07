#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sys
from tqdm import tqdm # for showing progress

import time


# In[36]:


# initializing dataset
data = pd.read_csv("training.csv")


# In[9]:


#preliminary analysis


def sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse = True)}

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
    return {key : round(value / articles.shape[0], 2) for (key, value) in word_occurences.items()}

def get_top_words(data, topic, n):
    articles = data[data['topic'] == topic]
    
    word_count = {} #map each word to how often it appears in the topic's articles
    
    word_occurences = {} # map each word to how many articles it appears in
    
    unique_words = set()
    
    for i in articles.index:
        words = data['article_words'][i].split(',')
        
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

topics = set(data.topic.values)

all_unique_words = set()

# getting the top words by topic
for topic in topics:
    top_words_by_count, top_words_by_occurences, unique_words = get_top_words(data, topic, 20)
    
    all_unique_words = all_unique_words.union(unique_words)
    
    print()
    print(topic)
    print("top word counts")
    print(top_words_by_count)

    print()
    print("top occurences")
    print(top_words_by_occurences)
    print()



print("total words:", len(all_unique_words))


# In[75]:


#pre-processing

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
    v = TfidfVectorizer(
            preprocessor = lambda x: x, #the preprocessor is set to be the identity function (it does nothing)
            tokenizer = lambda x: x.split(','), #the tokenizer (which converts a string into individual words) splits a string at ','
            ngram_range = NGRAM_RANGE,
            analyzer = TOKEN_MODE,
            min_df = MIN_DOCUMENT_FREQUENCY) # we decide to use unigrams and bigrams as the google guide suggests

    X =  v.fit_transform(data['article_words'])
    feature_names = v.get_feature_names()
    
    selector = SelectKBest(f_classif, k=min(TOP_K, X.shape[1])) #selects n best features based on f_classif metric
    selector.fit(X, data['topic'])
    
    X = selector.transform(X)

    return X, feature_names

    
x_train, feature_names = vectorize(data)
x_train.shape


# In[2]:





# In[ ]:




