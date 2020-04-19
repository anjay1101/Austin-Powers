#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Preprocessing import prepare_data
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
#from sklearn.classifiers import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier


# In[2]:


start_time = time.time()
train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()

#split training data in test and train set 
X_train, X_test, y_train, y_test = train_test_split(train_X,train_labels,test_size=0.3)

#implement basic decision tree
dtc = DecisionTreeClassifier(max_depth = 50, min_samples_leaf = 3)
dt = dtc.fit(X_train,y_train)
pred_dt = dt.predict(X_test)

#implement basic random forests 

#rfc with no parameters
rfc_1 = RandomForestClassifier() #max_depth = 10)
test_pred_1 = rfc_1.fit(X_train,y_train).predict(X_test)

#rfc with random parameters
rfc_2 = RandomForestClassifier(max_depth = 20,n_estimators = 200)
test_pred_2 = rfc_2.fit(X_train,y_train).predict(X_test)

#rfc with random parameters
rfc_3 = RandomForestClassifier(max_depth = 50,n_estimators = 50)
test_pred_3 = rfc_3.fit(X_train,y_train).predict(X_test)

print("--- %s seconds ---" % (time.time() - start_time))

#Usually takes about 2 minutes


# In[3]:


#Hyperparameter Tuning of RF using CV
start_time = time.time()

#Step 1: Use randomized search to find generally parameters 

# Create the random grid
random_grid = {'n_estimators': [50,100,150,200],
               'max_features': ['auto','sqrt'],
               'max_depth': [10,25,50,75,100],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,2,4],
               'bootstrap': [True,False]}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

best_random = rf_random.best_estimator_
pred_random = best_random.predict(X_test)
best_random_acc = accuracy_score(pred,y_test)
print("Accuracy of best random: ", best_random_acc)

print("--- %s seconds ---" % (time.time() - start_time))

#5-10 minutes on a slow computer


# In[4]:


#Step 2: Use Grid Search CV to find the best hyperparameters 
start_time = time.time()

# Create the parameter grid based on the results of random search (i.e. choose ones close to those that did best)
param_grid = {
    'bootstrap': [False],
    'max_depth': [50,75,100],
    'max_features': ['sqrt'],
    'min_samples_leaf': [2],
    'min_samples_split': [10],
    'n_estimators': [50,100]
}

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
pred = best_grid.predict(X_test)
best_grid_acc = accuracy_score(pred,y_test)
print("Accuracy of best grid: ", best_grid_acc)

print("--- %s seconds ---" % (time.time() - start_time))
#about 5 minutes 


# In[5]:


#Results

start_time = time.time()

#Decision Tree
print("Basic decision tree accuracy: ", accuracy_score(pred_dt,y_test))

#Model 1
print("Model 1: No parameters")
print("Accuracy Score: ", rfc_1.score(X_test,y_test))
print("Precision: ", precision_score(test_pred_1,y_test,average=None))
print("Recall: ", recall_score(test_pred_1,y_test,average=None,zero_division=1)) 
print("F1: ", f1_score(test_pred_1,y_test,average=None))
print("\n")


#Model 2
print("Model 2: max_depth 20, n_est = 200")
print("Accuracy Score: ", rfc_2.score(X_test,y_test))
print("Precision: ", precision_score(test_pred_2,y_test,average=None))
print("Recall: ", recall_score(test_pred_2,y_test,average=None,zero_division=1))
print("F1: ", f1_score(test_pred_2,y_test,average=None))
print("\n")


#Model 3
print("Model 3: max_depth 50, n_est = 50")
print("Accuracy Score: ", rfc_3.score(X_test,y_test))
print("Precision: ", precision_score(test_pred_3,y_test,average=None))
print("Recall: ", recall_score(test_pred_3,y_test,average=None,zero_division=1))
print("F1: ", f1_score(test_pred_3,y_test,average=None))
print("\n")

#Best Hyperparameters based on CV
print("\nModel 4: CV tuned with Random-Search ")
print("Parameters :", rf_random.best_params_)
random_pred = best_random.predict(X_test)
print("Accuracy Score: ", rf_random.score(X_test,y_test))
print("Precision: ", precision_score(random_pred,y_test,average=None, zero_division=0))
print("Recall: ", recall_score(random_pred,y_test,average=None,zero_division=0))
print("F1: ", f1_score(random_pred,y_test,average=None,zero_division=0))
print("\n")



print("\nModel 5: CV tuned with Grid-Search ")
print("Parameters :", grid_search.best_params_)
print("Accuracy Score: ", best_grid.score(X_test,y_test))
print("Precision: ", precision_score(pred,y_test,average=None, zero_division=0))
print("Recall: ", recall_score(pred,y_test,average=None,zero_division=0))
print("F1: ", f1_score(pred,y_test,average=None,zero_division=0))
print("\n")

print("--- %s seconds ---" % (time.time() - start_time))


# In[6]:


##Results

# Basic decision tree accuracy:  0.6894736842105263
# Model 1: No parameters
# Accuracy Score:  0.737280701754386
# Precision:  [0.14893617 0.17857143 0.20895522 0.72048193 0.26666667 0.
#  0.93021201 0.16666667 0.02631579 0.96691176 0.06122449]
# Recall:  [0.63636364 1.         0.875      0.62421712 0.8        1.
#  0.75321888 0.39506173 0.5        0.93928571 1.        ]
# F1:  [0.24137931 0.3030303  0.3373494  0.6689038  0.4        0.
#  0.83241107 0.23443223 0.05       0.95289855 0.11538462]


# Model 2: max_depth 20, n_est = 200
# Accuracy Score:  0.6978070175438597
# Precision:  [0.         0.         0.04477612 0.6313253  0.         0.
#  0.95141343 0.0625     0.         0.87132353 0.        ]
# Recall:  [1.         1.         1.         0.6313253  1.         1.
#  0.68424396 0.3        1.         0.95564516 1.        ]
# F1:  [0.         0.         0.08571429 0.6313253  0.         0.
#  0.79600887 0.10344828 0.         0.91153846 0.        ]


# Model 3: max_depth 50, n_est = 50
# Accuracy Score:  0.7315789473684211
# Precision:  [0.12765957 0.03571429 0.14925373 0.69156627 0.26666667 0.
#  0.93816254 0.21875    0.02631579 0.9375     0.        ]
# Recall:  [0.85714286 1.         0.90909091 0.64785553 1.         1.
#  0.7375     0.40776699 1.         0.94444444 1.        ]
# F1:  [0.22222222 0.06896552 0.25641026 0.66899767 0.42105263 0.
#  0.82581649 0.28474576 0.05128205 0.94095941 0.        ]



# Model 4: CV tuned with Random-Search 
# Parameters : {'n_estimators': 150, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': False}
# Accuracy Score:  0.7403508771929824
# Precision:  [0.14893617 0.07142857 0.26865672 0.74698795 0.26666667 0.
#  0.93021201 0.14583333 0.07894737 0.95955882 0.04081633]
# Recall:  [0.77777778 1.         0.9        0.62626263 1.         0.
#  0.76193922 0.34567901 1.         0.92553191 1.        ]
# F1:  [0.25       0.13333333 0.4137931  0.68131868 0.42105263 0.
#  0.83770883 0.20512821 0.14634146 0.94223827 0.07843137]



# Model 5: CV tuned with Grid-Search 
# Parameters : {'bootstrap': False, 'max_depth': 75, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 50}
# Accuracy Score:  0.7403508771929824
# Precision:  [0.19148936 0.07142857 0.2238806  0.71084337 0.26666667 0.
#  0.92932862 0.23958333 0.05263158 0.95588235 0.06122449]
# Recall:  [0.69230769 1.         0.88235294 0.64270153 1.         0.
#  0.75466284 0.43809524 0.66666667 0.92857143 1.        ]
# F1:  [0.3        0.13333333 0.35714286 0.67505721 0.42105263 0.
#  0.83293745 0.30976431 0.09756098 0.94202899 0.11538462]


##Essentially, CV only gives 1% improvement on no parameters model and the best accuracy is 0.74

:




