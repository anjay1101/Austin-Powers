import pandas as pd


#preliminary analysis

# initializing dataset
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
