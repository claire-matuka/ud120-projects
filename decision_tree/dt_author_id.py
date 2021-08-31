#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
### import the sklearn module for Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
### import sklearn module for accuracy score
from sklearn.metrics import accuracy_score

### create classifier
clf = DecisionTreeClassifier(min_samples_split = 40)

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)

### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)

### get the accuracy
accuracy = accuracy_score(labels_test, pred)

print("Accuracy: ", accuracy)


### get the number of features in the data
number_features = len(features_train[0])

print("The total number of features: ", number_features)

#########################################################


