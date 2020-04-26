#!/usr/bin/env python
# coding: utf-8

# In[13]:


from Preprocessing import prepare_data
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier


# In[14]:


start_time = time.time()
train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()

full_train_X = np.concatenate((train_X,val_X),axis=0)
full_train_labels = np.concatenate((train_labels,val_labels),axis=0)


# In[15]:


#implement basic decision tree
dtc = DecisionTreeClassifier(max_depth = 50, min_samples_leaf = 3)
dt = dtc.fit(train_X,train_labels)
pred_dt = dt.predict(val_X)

#implement basic random forests 

#rfc with no parameters
rfc_1 = RandomForestClassifier() #max_depth = 10)
test_pred_1 = rfc_1.fit(train_X,train_labels).predict(val_X)

#rfc with random parameters
rfc_2 = RandomForestClassifier(max_depth = 20,n_estimators = 200)
test_pred_2 = rfc_2.fit(train_X,train_labels).predict(val_X)

#rfc with random parameters
rfc_3 = RandomForestClassifier(max_depth = 50,n_estimators = 50)
test_pred_3 = rfc_3.fit(train_X,train_labels).predict(val_X)

print("--- %s seconds ---" % (time.time() - start_time))

#Usually takes about 2 minutes


# In[16]:


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
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = "balanced_accuracy", n_iter = 10, cv = 3, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_X, train_labels)

print(rf_random.best_params_)

best_random = rf_random.best_estimator_
random_pred = best_random.predict(val_X)
best_random_acc = accuracy_score(val_labels,random_pred)
print("Accuracy of best random: ", best_random_acc)

print("--- %s seconds ---" % (time.time() - start_time))



#5-10 minutes on a slow computer


# In[17]:


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
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, scoring = "balanced_accuracy",
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_X,train_labels)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_pred = best_grid.predict(val_X)
best_grid_acc = accuracy_score(val_labels,grid_pred)
print("Accuracy of best grid: ", best_grid_acc)

print("--- %s seconds ---" % (time.time() - start_time))
#about 5 minutes 


# In[18]:


#Results

start_time = time.time()

#Decision Tree
print("Basic decision tree accuracy: ", accuracy_score(val_labels,pred_dt))

#Model 1
print("Model 1: No parameters")
print("Balanced Accuracy Score: ", balanced_accuracy_score(val_labels,test_pred_1))
print("Accuracy Score: ", rfc_1.score(val_X,val_labels))
print("Precision: ", precision_score(val_labels,test_pred_1,average=None))
print("Recall: ", recall_score(val_labels,test_pred_1,average=None,zero_division=0)) 
print("F1: ", f1_score(val_labels,test_pred_1,average=None))
print("\n")


#Model 2
print("Model 2: max_depth 20, n_est = 200")
print("Balanced Accuracy Score: ", balanced_accuracy_score(val_labels,test_pred_2))
print("Accuracy Score: ", rfc_2.score(val_X,val_labels))
print("Precision: ", precision_score(val_labels,test_pred_2,average=None))
print("Recall: ", recall_score(val_labels,test_pred_2,average=None,zero_division=0))
print("F1: ", f1_score(val_labels,test_pred_2,average=None))
print("\n")


#Model 3
print("Model 3: max_depth 50, n_est = 50")
print("Balanced Accuracy Score: ", balanced_accuracy_score(val_labels,test_pred_3))
print("Accuracy Score: ", rfc_3.score(val_X,val_labels))
print("Precision: ", precision_score(val_labels,test_pred_3,average=None))
print("Recall: ", recall_score(val_labels,test_pred_3,average=None,zero_division=0))
print("F1: ", f1_score(val_labels,test_pred_3,average=None))
print("\n")

#Best Hyperparameters based on CV
print("\nModel 4: CV tuned with Random-Search ")
print("Parameters :", rf_random.best_params_)
print("Balanced Accuracy Score: ", balanced_accuracy_score(val_labels,random_pred))
print("Accuracy Score: ", rf_random.score(val_X,val_labels))
print("Precision: ", precision_score(val_labels,random_pred,average=None, zero_division=0))
print("Recall: ", recall_score(val_labels,random_pred,average=None,zero_division=0))
print("F1: ", f1_score(val_labels,random_pred,average=None,zero_division=0))
print("\n")



print("\nModel 5: CV tuned with Grid-Search ")
print("Parameters :", grid_search.best_params_)
print("Balanced Accuracy Score: ", balanced_accuracy_score(val_labels,grid_pred))
print("Accuracy Score: ", best_grid.score(val_X,val_labels))
print("Precision: ", precision_score(val_labels,grid_pred,average=None, zero_division=0))
print("Recall: ", recall_score(val_labels,grid_pred,average=None,zero_division=0))
print("F1: ", f1_score(val_labels,grid_pred,average=None,zero_division=0))
print("\n")

print("--- %s seconds ---" % (time.time() - start_time))

# In[ ]:


#CV Score of best estimator
#model 5
best_rf = RandomForestClassifier(bootstrap=False, max_depth=100, max_features= 'sqrt', min_samples_leaf= 2, min_samples_split=10, n_estimators= 50)
best_score = cross_val_score(best_rf,full_train_X,full_train_labels,scoring = 'balanced_accuracy',cv=3)
best_acc = cross_val_score(best_rf,full_train_X,full_train_labels,scoring='accuracy',cv=3)
print("Random Forest model - CV balanced accuracy: ", np.mean(best_score))
print("Random Forest model - CV accuracy: ", np.mean(best_acc))





# In[ ]:
#Results

# Basic decision tree accuracy:  0.7094736842105264
# Model 1: No parameters
# Balanced Accuracy Score:  0.339837362671898
# Accuracy Score:  0.7478947368421053
# Precision:  [0.78503046 0.92735043 0.80769231 1.         0.5        1.
#  0.39726027 1.         1.         0.59697733 0.6       ]
# Recall:  [0.92893924 0.97309417 0.41176471 0.11111111 0.1025641  0.10714286
#  0.17365269 0.04545455 0.04545455 0.74528302 0.09375   ]
# F1:  [0.8509434  0.94967177 0.54545455 0.2        0.17021277 0.19354839
#  0.24166667 0.08695652 0.08695652 0.66293706 0.16216216]


# Model 2: max_depth 20, n_est = 200
# Balanced Accuracy Score:  0.23331903893210654
# Accuracy Score:  0.7068421052631579
# Precision:  [0.70945427 0.93809524 0.         0.         0.         0.
#  0.37931034 0.         0.         0.58888889 0.        ]
# Recall:  [0.95056643 0.88340807 0.         0.         0.         0.
#  0.06586826 0.         0.         0.66666667 0.        ]
# F1:  [0.8125     0.90993072 0.         0.         0.         0.
#  0.1122449  0.         0.         0.62536873 0.        ]


# Model 3: max_depth 50, n_est = 50
# Balanced Accuracy Score:  0.3031860416329582
# /Users/anjayfriedman1/opt/anaconda3/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# Accuracy Score:  0.7431578947368421
# Precision:  [0.76890399 0.92139738 0.875      1.         0.33333333 1.
#  0.45783133 1.         0.         0.61979167 0.4       ]
# Recall:  [0.93202884 0.94618834 0.2745098  0.03703704 0.02564103 0.03571429
#  0.22754491 0.04545455 0.         0.74842767 0.0625    ]
# F1:  [0.84264432 0.93362832 0.41791045 0.07142857 0.04761905 0.06896552
#  0.304      0.08695652 0.         0.67806268 0.10810811]



# Model 4: CV tuned with Random-Search 
# Parameters : {'n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 75, 'bootstrap': False}
# Balanced Accuracy Score:  0.32620818861592227
# /Users/anjayfriedman1/opt/anaconda3/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# Accuracy Score:  0.32620818861592227
# Precision:  [0.78227194 0.91845494 0.85       1.         0.6        1.
#  0.41176471 1.         0.5        0.61483254 0.75      ]
# Recall:  [0.9361483  0.95964126 0.33333333 0.03703704 0.07692308 0.03571429
#  0.1257485  0.13636364 0.04545455 0.8081761  0.09375   ]
# F1:  [0.85232068 0.93859649 0.47887324 0.07142857 0.13636364 0.06896552
#  0.19266055 0.24       0.08333333 0.69836957 0.16666667]



# Model 5: CV tuned with Grid-Search 
# Parameters : {'bootstrap': False, 'max_depth': 100, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50}
# Balanced Accuracy Score:  0.3693249963108251
# Accuracy Score:  0.7589473684210526
# Precision:  [0.79170344 0.91983122 0.80645161 1.         0.33333333 1.
#  0.5        1.         1.         0.62371134 0.6       ]
# Recall:  [0.92378991 0.97757848 0.49019608 0.11111111 0.07692308 0.10714286
#  0.24550898 0.13636364 0.04545455 0.76100629 0.1875    ]
# F1:  [0.8526616  0.94782609 0.6097561  0.2        0.125      0.19354839
#  0.32931727 0.24       0.08695652 0.68555241 0.28571429]


# Parameters : {'bootstrap': False, 'max_depth': 100, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50}
#Best Random Forest model - CV balanced accuracy:  0.35233208885801565
#Best Random Forest model - CV accuracy:  0.7377892462428565



