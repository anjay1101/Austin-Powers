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


# Time the total execution.
total_time = datetime.now()


# Pre-processing.

# Get the training data.
train_data = pd.read_csv("training.csv")
# test_data = pd.read_csv("test.csv")


# Pre-process the training data.
train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()

# Pre-process for counts
vect = CountVectorizer()

X_train_vect = vect.fit_transform(train_data['article_words'])
y_train = train_data['topic'].apply(lambda x: topic_map[x])

X_train_vect, X_test_vect, y_train, y_test = train_test_split(X_train_vect,y_train,test_size=0.2,random_state=42)

# X_test_vect = vect.transform(test_data['article_words'])
# y_test = test_data['topic'].apply(lambda x: topic_map[x])


'''
#selects top 10k best features based on mutual_info metric
selector_1 = SelectKBest(mutual_info_classif, k=min(10000, X_train_vect.shape[1]))
selector_1.fit(X_train_vect, y_train)
#Transform features
X_train_mic = selector_1.transform(X_train_vect)
X_test_mic = selector_1.transform(X_test_vect)
'''
'''
#selects top 20k best features based on mutual_info metric
selector_2 = SelectKBest(mutual_info_classif, k=min(20000, X_train_vect.shape[1]))
selector_2.fit(X_train_vect, y_train)
#Transform features
X_train_mic_2 = selector_2.transform(X_train_vect)
X_test_mic_2 = selector_2.transform(X_test_vect)
'''
'''
#selects top 5k best features based on mutual_info metric
selector_3 = SelectKBest(mutual_info_classif, k=min(5000, X_train_vect.shape[1]))
selector_3.fit(X_train_vect, y_train)
#Transform features
X_train_mic_3 = selector_3.transform(X_train_vect)
X_test_mic_3 = selector_3.transform(X_test_vect)

#selects top 2.5k best features based on mutual_info metric
selector_4 = SelectKBest(mutual_info_classif, k=min(2500, X_train_vect.shape[1]))
selector_4.fit(X_train_vect, y_train)
#Transform features
X_train_mic_4 = selector_4.transform(X_train_vect)
X_test_mic_4 = selector_4.transform(X_test_vect)
'''

'''
#selects top 1k best features based on mutual_info metric
selector_5 = SelectKBest(mutual_info_classif, k=min(1000, X_train_vect.shape[1]))
selector_5.fit(X_train_vect, y_train)
#Transform features
X_train_mic_5 = selector_5.transform(X_train_vect)
X_test_mic_5 = selector_5.transform(X_test_vect)
'''



##Results


# Implement Multinomial Naive Bayes (MNB) using TFIDF scaling
'''
mnb = MultinomialNB()
# Train the MNB classifier.
mnb.fit(train_X, train_labels)
# Test our MNB results.
train_result = mnb.score(train_X, train_labels)
print("The final score for the training database on the MNB classifier is: ", round(train_result, 4))
test_result = mnb.score(val_X, val_labels)
pred = mnb.predict(val_X)
b_acc_score = balanced_accuracy_score(val_labels,pred)
print("The final score for the test database on the MNB classifier is: ", round(test_result, 4))
print("The final balance accuracy score for the test set on the MNB classifier is: ", round(b_acc_score, 4), '\n')
'''
'''
# Implement Multinomial Naive Bayes (MNB) using counts
count_mnb = MultinomialNB().fit(X_train_vect,y_train)
train_result = count_mnb.score(X_train_vect,y_train)
print("The final score for the training database on the count MNB classifier is: ", round(train_result, 4))
test_result = count_mnb.score(X_test_vect,y_test)
print("The final score for the test database on the count MNB classifier is: ", round(test_result, 4))
pred = count_mnb.predict(X_test_vect)
b_acc_score = balanced_accuracy_score(y_test,pred)
print("The final balance accuracy score for the test set on the count MNB classifier is: ", round(b_acc_score, 4), '\n')
'''

'''
# Implement Multinomial Naive Bayes (MNB) using counts and top 10k mutual info features
mic_mnb = MultinomialNB().fit(X_train_mic,y_train)
train_result = mic_mnb.score(X_train_mic,y_train)
print("The final training score on the top 10k mutual info feature selection MNB classifier is: ", round(train_result, 4))
test_result = mic_mnb.score(X_test_mic,y_test)
print("The final test score on the top 10k mutual info feature selection MNB classifier is: ", round(test_result, 4))
pred = mic_mnb.predict(X_test_mic)
b_acc_score = balanced_accuracy_score(y_test,pred)
print("The final balance accuracy score on the top 10k mutual info feature selection MNB classifier is: ", round(b_acc_score, 4), '\n')
'''

'''
# Implement Multinomial Naive Bayes (MNB) using counts and top 20k mutual info features
mic2_mnb = MultinomialNB().fit(X_train_mic_2,y_train)
train_result = mic2_mnb.score(X_train_mic_2,y_train)
print("\nThe final training score on the top 20k mutual info feature selection MNB classifier is: ", round(train_result, 4))
test_result = mic2_mnb.score(X_test_mic_2,y_test)
print("The final test score on the top 20k mutual info feature selection MNB classifier is: ", round(test_result, 4))
pred = mic2_mnb.predict(X_test_mic_2)
b_acc_score = balanced_accuracy_score(y_test,pred)
print("The final balance accuracy score on the top 20k mutual info feature selection MNB classifier is: ", round(b_acc_score, 4), '\n')
'''

'''
# Implement Multinomial Naive Bayes (MNB) using counts and top 5k mutual info features
mic3_mnb = MultinomialNB().fit(X_train_mic_3,y_train)
train_result = mic3_mnb.score(X_train_mic_3,y_train)
print("\nThe final training score on the top 5k mutual info feature selection MNB classifier is: ", round(train_result, 4))
test_result = mic3_mnb.score(X_test_mic_3,y_test)
print("The final test score on the top 5k mutual info feature selection MNB classifier is: ", round(test_result, 4))
pred = mic3_mnb.predict(X_test_mic_3)
b_acc_score = balanced_accuracy_score(y_test,pred)
print("The final balance accuracy score on the top 5k mutual info feature selection MNB classifier is: ", round(b_acc_score, 4), '\n')
'''

'''
# Implement Multinomial Naive Bayes (MNB) using counts and top 2.5k mutual info features
mic4_mnb = MultinomialNB().fit(X_train_mic_4,y_train)
train_result = mic4_mnb.score(X_train_mic_4,y_train)
print("\nThe final training score on the top 2.5k mutual info feature selection MNB classifier is: ", round(train_result, 4))
test_result = mic4_mnb.score(X_test_mic_4,y_test)
print("The final test score on the top 2.5k mutual info feature selection MNB classifier is: ", round(test_result, 4))
pred = mic4_mnb.predict(X_test_mic_4)
b_acc_score = balanced_accuracy_score(y_test,pred)
print("The final balance accuracy score on the top 2.5k mutual info feature selection MNB classifier is: ", round(b_acc_score, 4), '\n')
'''

'''
print("Precision: ", precision_score(y_test,pred,average=None))
print("Recall: ", recall_score(y_test,pred,average=None,zero_division=0))
print("F1: ", f1_score(y_test,pred,average=None))
'''

'''
# Implement Multinomial Naive Bayes (MNB) using counts and top 1k mutual info features
mic5_mnb = MultinomialNB().fit(X_train_mic_5,y_train)
train_result = mic5_mnb.score(X_train_mic_5,y_train)
print("\nThe final training score on the top 1k mutual info feature selection MNB classifier is: ", round(train_result, 4))
test_result = mic5_mnb.score(X_test_mic_5,y_test)
print("The final test score on the top 1k mutual info feature selection MNB classifier is: ", round(test_result, 4))
pred = mic5_mnb.predict(X_test_mic_5)
b_acc_score = balanced_accuracy_score(y_test,pred)
print("The final balance accuracy score on the top 1k mutual info feature selection MNB classifier is: ", round(b_acc_score, 4), '\n')
'''


## The k for SelectkBest which maximised balanced_accuracy_score was 2,500. balanced_score = 0.7593



#selects top 2.5k best features based on mutual_info metric
selector_4 = SelectKBest(mutual_info_classif, k=min(2500, X_train_vect.shape[1]))
selector_4.fit(X_train_vect, y_train)
#Transform features
X_train_mic_4 = selector_4.transform(X_train_vect)
X_test_mic_4 = selector_4.transform(X_test_vect)


# Implement Multinomial Naive Bayes (MNB) using counts and top 2.5k mutual info features
mic4_mnb = MultinomialNB().fit(X_train_mic_4,y_train)
train_result = mic4_mnb.score(X_train_mic_4,y_train)
print("\nThe final training score on the top 2.5k mutual info feature selection MNB classifier is: ", round(train_result, 4))
test_result = mic4_mnb.score(X_test_mic_4,y_test)
print("The final test score on the top 2.5k mutual info feature selection MNB classifier is: ", round(test_result, 4))
pred = mic4_mnb.predict(X_test_mic_4)
b_acc_score = balanced_accuracy_score(y_test,pred)
print("The final balance accuracy score on the top 2.5k mutual info feature selection MNB classifier is: ", round(b_acc_score, 4), '\n')

# Now tune the alpha for this model.

max_alph = 0
max_b_score = 0
for i in np.arange(0.0, 1.0, 0.001):
    mic4_mnb = MultinomialNB(alpha = i).fit(X_train_mic_4, y_train)
    pred = mic4_mnb.predict(X_test_mic_4)
    b_acc_score = balanced_accuracy_score(y_test, pred)
    #print("alpha = ", i, "balanced score = ", b_acc_score)
    if b_acc_score > max_b_score:
        max_b_score = b_acc_score
        max_alph = i
print("The best balanced score with k=2500 was = ", max_b_score, " when alpha = ", max_alph)





print('\n\nTotal execution time: ', datetime.now() - total_time)

