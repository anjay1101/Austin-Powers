#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import sys
from tqdm import tqdm # for showing progress


# In[10]:
data = pd.read_csv("training.csv")

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
#     print(word_occurences)
    print(articles.shape[0])
    return {key : round(value / articles.shape[0], 2) for (key, value) in word_occurences.items()}

topics = set(data.topic.values)

all_unique_words = set()

# getting the top words by topic
for topic in topics:
    articles = data[data['topic'] == topic]

    word_count = {} #map each word to how often it appears in the topic's articles

    word_occurences = {} # map each word to how many articles it appears in

    for i in articles.index:
        words = data['article_words'][i].split(',')

        for word in words:
            word_count.setdefault(word, 0)
            word_count[word] += 1

        unique_words = set(words)
        all_unique_words = all_unique_words.union(unique_words)

        for word in unique_words:
            word_occurences.setdefault(word, 0)
            word_occurences[word] += 1

    word_occurences = normalize_occurences(word_occurences, articles)

    word_count = sort_dict(word_count)

    print()
    print(topic)
    print("top word counts")
    print(top_entries(word_count, 20))
    print("average word count:", np.mean(list(word_count.values())) )

    print()
    print("top occurences")
    print(top_entries(word_occurences, 20))
    print()



print("total words:", len(all_unique_words))
