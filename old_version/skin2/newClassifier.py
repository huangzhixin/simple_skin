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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import toolbox
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    train_x = normalization(train_x)
    model = KNeighborsClassifier()
    params = {'n_neighbors': [1,2,3,4,5,6,7,8,9], 'algorithm' :['auto', 'ball_tree', 'kd_tree', 'brute'], 'weights' :['uniform', 'distance']}
    grid = GridSearchCV(estimator=model, param_grid=params,cv=5)
    grid = grid.fit(train_x,train_y)
    report(grid.cv_results_)
    best_parameters = grid.best_estimator_.get_params()
    #for para, val in best_parameters.items():
    #    print para, val
    model = KNeighborsClassifier(n_neighbors = best_parameters['n_neighbors'],algorithm=best_parameters['algorithm'], weights=best_parameters['weights'])
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    #train_x = normalization(train_x)
    model = LogisticRegression(penalty='l2',class_weight = 'balanced')
    params = {'solver' : ['newton-cg', 'lbfgs', 'liblinear','sag']}
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid = grid.fit(train_x, train_y)
    report(grid.cv_results_)
    best_parameters = grid.best_estimator_.get_params()
    # for para, val in best_parameters.items():
    #    print para, val
    model = LogisticRegression(penalty='l2',class_weight = 'balanced',solver=best_parameters['solver'])
    model.fit(train_x, train_y)
    return model

# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    #train_x = normalization(train_x)
    model = RandomForestClassifier(class_weight='balanced')
    params = {'n_estimators': [18,20,22,24,26,28,30,32],'criterion':['gini','entropy'],'max_features':['auto','log2',None ]}
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid = grid.fit(train_x, train_y)
    report(grid.cv_results_)
    best_parameters = grid.best_estimator_.get_params()
    # for para, val in best_parameters.items():
    #    print para, val
    model = RandomForestClassifier(class_weight='balanced',n_estimators=best_parameters['n_estimators'],criterion=best_parameters['criterion'],max_features=best_parameters['max_features'])
    model.fit(train_x, train_y)
    return model

# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    #train_x = normalization(train_x)
    params = {'splitter': ['best','random'], 'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 'log2', None],'class_weight': ['balanced',None]}
    model = tree.DecisionTreeClassifier()
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid = grid.fit(train_x, train_y)
    report(grid.cv_results_)
    best_parameters = grid.best_estimator_.get_params()
    # for para, val in best_parameters.items():
    #    print para, val
    model = tree.DecisionTreeClassifier(splitter=best_parameters['splitter'],criterion=best_parameters['criterion'],max_features=best_parameters['max_features'],class_weight=best_parameters['class_weight'])
    model.fit(train_x, train_y)
    return model

# SVM Classifier    #线性非线性核需要读文档
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC()
    train_x = normalization(train_x)
    params = {'C': [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0], 'kernel': [ 'poly'],'degree': [2,3,4,5,6,7,8,9], 'gamma': [0.01,0.05,0.1,0,2,0.5,1],'class_weight':[None, 'balanced'],'decision_function_shape':['ovo', 'ovr', None] }
    #params = {'C': [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0], 'kernel': ['rbf', 'sigmoid'],'class_weight':[None, 'balanced']}
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid = grid.fit(train_x, train_y)
    report(grid.cv_results_)
    #model = SVC(kernel='rbf',C=10, gamma=0.001)
   # model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    params = {'loss':['deviance'], 'learning_rate': [0.05,0.1,0.2,0.3,0.4],'n_estimators':[100,200,300],'max_depth':[3,4,5,6,7,8]}
    #params = {'loss': ['deviance'], 'learning_rate': [0.05, 0.1 ],'n_estimators': [100,200], 'max_depth':[5,6]}
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid = grid.fit(train_x, train_y)
    report(grid.cv_results_)
    best_parameters = grid.best_estimator_.get_params()
    # for para, val in best_parameters.items():
    #    print para, val
    model = GradientBoostingClassifier(loss=best_parameters['loss'], learning_rate=best_parameters['learning_rate'],
                                       n_estimators=best_parameters['n_estimators'],
                                       max_depth=best_parameters['max_depth'])
    model.fit(train_x, train_y)
    return model

def readData(fileLoc):
    # the last Column is label so we select train data is [:,:-1]
    # now we change the dataframe to np.array
    #
    # 算小类
    # featuresTable = pd.read_pickle(fileLoc + '/primaryClassFeatures.pkl')
    # featuresTable1 = pd.read_pickle(fileLoc + '/small_segmentFeatures.pkl')
    # data_x = featuresTable.values
    # data_y = featuresTable1.ix[:,-1].values
    # 算全类
    #featuresTable = pd.read_pickle(fileLoc + '/allClassFeatures.pkl')
    #featuresTable1 = pd.read_pickle(fileLoc + '/segmentFeatures.pkl')
    #data_x = featuresTable.values
    #data_y = featuresTable1.ix[:, -1].values
    # 算重力分类
    #featuresTable = pd.read_pickle(fileLoc + '/weightClassFeatures.pkl')
    #data_x = featuresTable.values
    #data_y = np.load(fileLoc + '/weightLabel.npy')[1:-1]

    #算中等frequence类
    featuresTable = pd.read_pickle(fileLoc + '/frequenceClassFeatures.pkl')
    data_x = featuresTable.values
    data_y = np.load(fileLoc + '/frequenceLabel.npy')[1:-1]


    print data_y
    return data_x, data_y

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


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





print 'reading training and testing data...'
data_x, data_y = readData('./a')
test_x, test_y = readData('./f')
print data_x.shape, test_x.shape
num_train, num_feat = data_x.shape
class_names = np.unique(data_y)
for i in range(0, class_names.shape[0]):
    class_names[i] = str(0) + class_names[i]
# class_names = np.load('e/classNames.npy')

# ====================================================================================#

# normalization is very important before you train the model!!!!!!!!!!
print 'first you need to normalization the train set'
# data_x = normalization(data_x)

train_x = np.concatenate((data_x, test_x), axis=0)
train_y = np.concatenate((data_y, test_y), axis=0)

#knn_classifier(train_x, train_y)
#logistic_regression_classifier(train_x, train_y)
#random_forest_classifier(train_x, train_y)
#decision_tree_classifier(train_x, train_y)
#svm_classifier(train_x, train_y)
gradient_boosting_classifier(train_x, train_y)

'''
model_save_file = None
model_save = {}

test_classifiers = [  # 'NB',
    'KNN',
    'LR',
    'RF',
    'DT',
    #'SVM',
    # 'SVMCV',
    'GBDT']
classifiers = {
    # 'NB':naive_bayes_classifier,
    'KNN': knn_classifier,
    'LR': logistic_regression_classifier,
    'RF': random_forest_classifier,
    'DT': decision_tree_classifier,
    #'SVM': svm_classifier,
    # 'SVMCV':svm_cross_validation,
    'GBDT': gradient_boosting_classifier
}


# =====================================================================================#
print '******************** Data Info *********************'
print '#training data: %d, dimension of feature: %d' % (num_train, num_feat)

for classifier in test_classifiers:
    print '******************* %s ********************' % classifier

    model = classifiers[classifier]
    accuracy, confusion_mat = kFoldValidation(data_x, data_y, 2, model)
    # accuracy, confusion_mat = LOOValidation(data_x,data_y,test_x, test_y,model)
    print type(confusion_mat)

    print confusion_mat

    plt.figure()
    toolbox.plot_confusion_matrix(confusion_mat, classes=class_names, normalize=True,
                                  title='confusion matrix of ' + classifier + ' ACC :' + str(accuracy))
    plt.show()

'''