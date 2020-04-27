#!/usr/bin/env python
# coding: utf-8

from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

from Preprocessing import prepare_data

x_train, x_test, y_train, y_test, num_classes, topic_map = prepare_data()

sm = SMOTE(random_state=2)
x_data_train, y_data_train = sm.fit_sample(x_train, y_train)

#### svm

# Best: 0.724737 using {'probability': True, 'kernel': 'linear', 'gamma': 0.0001, 'degree': 3, 'C': 0.1}

svc = svm.SVC(random_state=8)

C = [0.001, 0.1, 1, 10]
gamma = [0.0001, 0.001, 0.01, 0.1]
degree = [1, 2, 3, 4]
kernel = ['linear', 'rbf', 'poly']
probability = [True]

random_grid = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree, 'probability': probability}

random_search = RandomizedSearchCV(estimator=svc, param_distributions=random_grid, n_iter=50, scoring='accuracy', cv=3,
                                   verbose=1, random_state=8)

random_search.fit(x_train, y_train)

print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))

svc = svm.SVC(random_state=8)

svc.fit(x_data_train, y_data_train)

svc_pred = svc.predict(x_test)

print(balanced_accuracy_score(y_test, svc_pred))
print(precision_score(y_test, svc_pred, average='weighted'))
print(recall_score(y_test, svc_pred, average='weighted'))
print(f1_score(y_test, svc_pred, average='weighted'))