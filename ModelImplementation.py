import numpy as np
import pandas as pd
import sys

from Preprocessing import vectorize

import time
from datetime import datetime

# Time the total execution.
total_time = datetime.now()

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
