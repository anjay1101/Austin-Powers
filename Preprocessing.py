import pandas as pd
import numpy as np
import tensorflow as tf

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

def vectorize(train_data, test_data):
    v = TfidfVectorizer(
            preprocessor = lambda x: x, #the preprocessor is set to be the identity function (it does nothing)
            tokenizer = lambda x: x.split(','), #the tokenizer (which converts a string into individual words) splits a string at ','
            ngram_range = NGRAM_RANGE,
            analyzer = TOKEN_MODE,
            min_df = MIN_DOCUMENT_FREQUENCY) # we decide to use unigrams and bigrams as the google guide suggests

    X_train =  v.fit_transform(train_data['article_words']) # create a column for each word and compute the tfidf score for each word/document combination
    X_test = v.transform(test_data['article_words'])

    # an array representing all words,
    # if you want to know what feature the nth column of X represents just access the nth element in the array
    feature_names = v.get_feature_names()

    selector = SelectKBest(f_classif, k=min(TOP_K, X_train.shape[1])) #selects n best features based on f_classif metric
    selector.fit(X_train, train_data['topic'])

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    # creates a mapping from new column to old column
    # the ith column in the X created by the new selector is the col_map[i]th column in the old one
    col_map = selector.get_support(indices=True)

    # get the new feature names using the old feature names and col_map
    new_feature_names = [feature_names[col_map[i]] for i in range(len(col_map))]

    return X_train, X_test, new_feature_names

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


def topics_to_num(data):
    topics = list(set(data.topic.values))

    # maps each topic to a number
    topic_map = {topics[i]:i for i in range(len(topics))}

    data['topic'] = data['topic'].apply(lambda x: topic_map[x])

    return data, topic_map

def split_train_validate(X, data, validation_percent):
    n_rows = X.shape[0]
    train_rows = int(n_rows * (1 - validation_percent))

    train_X = X[:train_rows, :]
    val_X = X[train_rows:, :]

    train_labels = data.topic[:train_rows]
    val_labels = data.topic[train_rows:]

    return train_X, val_X, train_labels, val_labels

# percentage of data that is converted to validation data
validation_percent = 0.2

def prepare_data(validation_percent=validation_percent):
    train_data = pd.read_csv("training.csv")
    train_data = shuffle(train_data)

    train_data, topic_map = topics_to_num(train_data)

    test_data = pd.read_csv("test.csv")
    test_data, _ = topics_to_num(test_data)

    X, test_X, feature_names = vectorize(train_data, test_data)
    X = X.toarray() # make dense rather than sparse

    train_X, val_X, train_labels, val_labels = split_train_validate(X, train_data, validation_percent)

    num_classes = len(set(train_data.topic))

    return train_X, val_X, train_labels, val_labels, num_classes, topic_map
