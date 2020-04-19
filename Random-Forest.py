#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

from sklearn.tree import DecisionTreeClassifier


# In[2]:


# In[2]:


start_time = time.time()
train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()

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


# In[3]:


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
rf_random.fit(train_X, train_labels)

print(rf_random.best_params_)

best_random = rf_random.best_estimator_
random_pred = best_random.predict(val_X)
best_random_acc = accuracy_score(random_pred,val_labels)
print("Accuracy of best random: ", best_random_acc)

print("--- %s seconds ---" % (time.time() - start_time))



#5-10 minutes on a slow computer


# In[5]:


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
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, scoring = "balanced_accuracy",
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_X,train_labels)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_pred = best_grid.predict(val_X)
best_grid_acc = accuracy_score(grid_pred,val_labels)
print("Accuracy of best grid: ", best_grid_acc)

print("--- %s seconds ---" % (time.time() - start_time))
#about 5 minutes 


# In[6]:


# In[5]:


#Results

start_time = time.time()

#Decision Tree
print("Basic decision tree accuracy: ", accuracy_score(pred_dt,val_labels))

#Model 1
print("Model 1: No parameters")
print("Balanced Accuracy Score: ", balanced_accuracy_score(test_pred_1,val_labels))
print("Accuracy Score: ", rfc_1.score(val_X,val_labels))
print("Precision: ", precision_score(test_pred_1,val_labels,average=None))
print("Recall: ", recall_score(test_pred_1,val_labels,average=None,zero_division=0)) 
print("F1: ", f1_score(test_pred_1,val_labels,average=None))
print("\n")


#Model 2
print("Model 2: max_depth 20, n_est = 200")
print("Balanced Accuracy Score: ", balanced_accuracy_score(test_pred_2,val_labels))
print("Accuracy Score: ", rfc_2.score(val_X,val_labels))
print("Precision: ", precision_score(test_pred_2,val_labels,average=None))
print("Recall: ", recall_score(test_pred_2,val_labels,average=None,zero_division=0))
print("F1: ", f1_score(test_pred_2,val_labels,average=None))
print("\n")


#Model 3
print("Model 3: max_depth 50, n_est = 50")
print("Balanced Accuracy Score: ", balanced_accuracy_score(test_pred_3,val_labels))
print("Accuracy Score: ", rfc_3.score(val_X,val_labels))
print("Precision: ", precision_score(test_pred_3,val_labels,average=None))
print("Recall: ", recall_score(test_pred_3,val_labels,average=None,zero_division=0))
print("F1: ", f1_score(test_pred_3,val_labels,average=None))
print("\n")

#Best Hyperparameters based on CV
print("\nModel 4: CV tuned with Random-Search ")
print("Parameters :", rf_random.best_params_)
print("Balanced Accuracy Score: ", balanced_accuracy_score(random_pred,val_labels))
print("Accuracy Score: ", rf_random.score(val_X,val_labels))
print("Precision: ", precision_score(random_pred,val_labels,average=None, zero_division=0))
print("Recall: ", recall_score(random_pred,val_labels,average=None,zero_division=0))
print("F1: ", f1_score(random_pred,val_labels,average=None,zero_division=0))
print("\n")



print("\nModel 5: CV tuned with Grid-Search ")
print("Parameters :", grid_search.best_params_)
print("Balanced Accuracy Score: ", balanced_accuracy_score(grid_pred,val_labels))
print("Accuracy Score: ", best_grid.score(val_X,val_labels))
print("Precision: ", precision_score(grid_pred,val_labels,average=None, zero_division=0))
print("Recall: ", recall_score(grid_pred,val_labels,average=None,zero_division=0))
print("F1: ", f1_score(grid_pred,val_labels,average=None,zero_division=0))
print("\n")

print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


#Results

# Basic decision tree accuracy:  0.6963157894736842
# Model 1: No parameters
# Balanced Accuracy Score:  0.7592188374906157
# Accuracy Score:  0.741578947368421
# Precision:  [0.02173913 0.25       0.2        0.24528302 0.03571429 0.03448276
#  0.19886364 0.9321663  0.98678414 0.25       0.75574713]
# Recall:  [0.25       0.57142857 1.         0.92857143 1.         1.
#  0.4375     0.76207513 0.90322581 0.85714286 0.64146341]
# F1:  [0.04       0.34782609 0.33333333 0.3880597  0.06896552 0.06666667
#  0.2734375  0.83858268 0.94315789 0.38709677 0.6939314 ]


# Model 2: max_depth 20, n_est = 200
# Balanced Accuracy Score:  0.710607763064131
# /Users/anjayfriedman1/opt/anaconda3/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1859: UserWarning: y_pred contains classes not in y_true
#   warnings.warn('y_pred contains classes not in y_true')
# Accuracy Score:  0.6947368421052632
# Precision:  [0.         0.         0.         0.         0.         0.
#  0.05681818 0.95842451 0.92070485 0.04166667 0.6408046 ]
# Recall:  [0.         0.         0.         0.         0.         0.
#  0.32258065 0.68384075 0.92888889 1.         0.61772853]
# F1:  [0.         0.         0.         0.         0.         0.
#  0.09661836 0.79817768 0.92477876 0.08       0.62905501]


# Model 3: max_depth 50, n_est = 50
# Balanced Accuracy Score:  0.7512454173122065
# /Users/anjayfriedman1/opt/anaconda3/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1859: UserWarning: y_pred contains classes not in y_true
#   warnings.warn('y_pred contains classes not in y_true')
# Accuracy Score:  0.7294736842105263
# Precision:  [0.04347826 0.125      0.         0.18867925 0.         0.06896552
#  0.18181818 0.94201313 0.969163   0.16666667 0.71551724]
# Recall:  [0.5        0.66666667 0.         0.90909091 0.         1.
#  0.3902439  0.74224138 0.91286307 1.         0.64010283]
# F1:  [0.08       0.21052632 0.         0.3125     0.         0.12903226
#  0.24806202 0.83027965 0.94017094 0.28571429 0.67571235]



# Model 4: CV tuned with Random-Search 
# Parameters : {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': False}
# Balanced Accuracy Score:  0.7870539278494335
# Accuracy Score:  0.7489473684210526
# Precision:  [0.08695652 0.25       0.2        0.28301887 0.07142857 0.10344828
#  0.21022727 0.93654267 0.98678414 0.3125     0.74712644]
# Recall:  [0.57142857 0.57142857 1.         0.88235294 1.         1.
#  0.42045455 0.76978417 0.90322581 0.88235294 0.65656566]
# F1:  [0.1509434  0.34782609 0.33333333 0.42857143 0.13333333 0.1875
#  0.28030303 0.84501481 0.94315789 0.46153846 0.69892473]



# Model 5: CV tuned with Grid-Search 
# Parameters : {'bootstrap': False, 'max_depth': 100, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50}
# Balanced Accuracy Score:  0.7935883115767389
# /Users/anjayfriedman1/opt/anaconda3/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1859: UserWarning: y_pred contains classes not in y_true
#   warnings.warn('y_pred contains classes not in y_true')
# Accuracy Score:  0.7442105263157894
# Precision:  [0.10869565 0.25       0.2        0.24528302 0.         0.10344828
#  0.20454545 0.93326039 0.99118943 0.375      0.72988506]
# Recall:  [0.83333333 0.57142857 1.         0.92857143 0.         1.
#  0.43902439 0.76296959 0.90361446 0.85714286 0.63979849]
# F1:  [0.19230769 0.34782609 0.33333333 0.3880597  0.         0.1875
#  0.27906977 0.83956693 0.94537815 0.52173913 0.68187919]





