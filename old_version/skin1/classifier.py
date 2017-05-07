#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 12.11.2016
#File Name: 	classifier.py
# 
# Method Name : classifier
# 
# Description:  There are some classifiers of machine learning, you can choose which one is better for you :)  
# 
# Argument :	traindata and labeldata 
#
# Output:  		  return trained model 

import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import toolbox
import matplotlib.pyplot as plt

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.neighbors import NearestNeighbors
    #nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(train_x)
    #distances, indices = nbrs.kneighbors(train_x)
    #print indices
    #print distances
    model = KNeighborsClassifier(n_neighbors = 5,algorithm = 'ball_tree')
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier    #线性非线性核需要读文档
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    #model = SVC(kernel='linear', probability=True)
    #model = SVC(kernel='sigmoid')
    model = SVC(kernel='poly', degree=3, C=100)
    #model = SVC(kernel='rbf',C=10, gamma=0.001)
    model.fit(train_x, train_y)
    return model

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def readData(fileLoc):
    # the last Column is label so we select train data is [:,:-1]
    # now we change the dataframe to np.array 
    featuresTable = pd.read_pickle(fileLoc+'/featuresTable.pkl')
    data_x = featuresTable.ix[:,:-1].values
    data_y = featuresTable.ix[:,-1].values
    return data_x, data_y

def normalization(trainData):
    trainData_normalized = preprocessing.normalize(trainData, norm='l2',axis = 0)
    #axis used to normalize the data along. If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
    return trainData_normalized


#http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
def kFoldValidation(data_x, data_y, K, classifier):
    kf = KFold(n_splits=K)
    confusion_mat = np.zeros((np.unique(data_y).shape[0],np.unique(data_y).shape[0]))
    accuracy = 0
    numOfk=1
    for indexOfTrain, indexOfTest in kf.split(data_x):
        start_time = time.time()
        #print("%s %s" % (indexOfTrain, indexOfTest))
        train_x = data_x[indexOfTrain]
        test_x = data_x[indexOfTest]
        train_y = data_y[indexOfTrain]
        test_y = data_y[indexOfTest]
        start_time = time.time()
        model = classifier(train_x, train_y)
        #print indexOfTrain
        print 'training took %fs!' % (time.time() - start_time)
        predict = model.predict(train_x)
        #print "train" + str(metrics.accuracy_score(train_y, predict))
        predict = model.predict(test_x)
        """
        if model_save_file != None:
            model_save[classifier] = model
        """
        confusion_mat = confusion_mat + confusion_matrix(test_y, predict) 
        print 'accuracy of kfold %d : %.2f%%' %(numOfk, (100 * metrics.accuracy_score(test_y, predict))) 
        numOfk = numOfk+1
        accuracy = accuracy + metrics.accuracy_score(test_y, predict)      
    print 'average accuracy: %.2f%%' % (100 * accuracy/K)
    return 100 * accuracy/K,confusion_mat
 

def LOOValidation(data):
    loo = LeaveOneOut()
    for indexOfTrain, indexOfTest in loo.split(X):
        print("%s %s" % (indexOfTrain, indexOfTest))
    #这些返回的是它的index
 

if __name__ == '__main__':

   model_save_file = None
   model_save = {}
   
   test_classifiers = [#'NB', 
                       'KNN', 
                       'LR', 
                       'RF', 
                       'DT', 
                       'SVM', 
                       #'SVMCV',
                       'GBDT']
   classifiers = {
                  #'NB':naive_bayes_classifier, 
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,
                  'SVM':svm_classifier,
                #'SVMCV':svm_cross_validation,
                 'GBDT':gradient_boosting_classifier
   }
   
   print 'reading training and testing data...'
   data_x,data_y = readData('./')
   num_train, num_feat = data_x.shape
   class_names = np.unique(data_y)
   
   #====================================================================================#

   #normalization is very important before you train the model!!!!!!!!!!
   print 'first you need to normalization the train set'
   #data_x = normalization(data_x)
  
   #=====================================================================================#
   print '******************** Data Info *********************'
   print '#training data: %d, dimension of feature: %d' % (num_train, num_feat)


   for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        
        model = classifiers[classifier]
        accuracy,confusion_mat = kFoldValidation(data_x, data_y, 4, model) 
        print type(confusion_mat)
        
        print confusion_mat
        
        plt.figure()
        toolbox.plot_confusion_matrix(confusion_mat, classes=class_names, normalize=True,title='confusion matrix of '+classifier+' ACC :' + str(accuracy))
        plt.show()
    
