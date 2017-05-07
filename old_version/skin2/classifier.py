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
from sklearn.model_selection import cross_val_score
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
    #from sklearn.neighbors import NearestNeighbors
    #nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(train_x)
    #distances, indices = nbrs.kneighbors(train_x)
    #print indices
    #print distances
    #train_x = normalization(train_x)
    model = KNeighborsClassifier(n_neighbors = 6,algorithm = 'auto', weights='distance')
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
    model = RandomForestClassifier(n_estimators=30, criterion='gini', max_features='log2')
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier(max_features=None, splitter='best', criterion='gini',class_weight='balanced')
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=300,loss='deviance',learning_rate=0.4,max_depth= 3)
    model.fit(train_x, train_y)
    print model.feature_importances_
    return model


# SVM Classifier    #线性非线性核需要读文档
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    #model = SVC(kernel='linear', probability=True)
    #model = SVC(kernel='sigmoid')
    model = SVC(kernel='poly', degree=3, C=10,probability=True)
    #model = SVC(kernel='rbf',C=10, gamma=0.001)
    model.fit(train_x, train_y)
    return model


def adaboost_classifier(train_x, train_y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier()
    model.fit(train_x, train_y)
    return model

def mlp_classifier(train_x, train_y):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(alpha=1)
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

def voter_classifier(train_x, train_y):
    from sklearn.ensemble import VotingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=6, algorithm='auto', weights='distance')
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(penalty='l2')
    from sklearn.ensemble import RandomForestClassifier
    RF = RandomForestClassifier(n_estimators=30, criterion='gini', max_features='log2')
    from sklearn import tree
    DT = tree.DecisionTreeClassifier(max_features=None, splitter='best', criterion='gini', class_weight='balanced')
    from sklearn.ensemble import GradientBoostingClassifier
    GBDT = GradientBoostingClassifier(n_estimators=300, loss='deviance', learning_rate=0.4, max_depth=3)
    from sklearn.svm import SVC
    SVM = SVC(kernel='poly', degree=3, C=10,probability=True)
    model = VotingClassifier(estimators=[('RF', RF), ('LR', LR), ('RF', RF),('DT', DT),('GBDT',GBDT),('GBDT',GBDT)], voting='soft')
    model.fit(train_x, train_y)
    #print 'good!'
    #scores = cross_val_score(model, train_x, train_y, cv=5, scoring='accuracy')
    #print scores
    return model



def readData(fileLoc):
    # the last Column is label so we select train data is [:,:-1]
    # now we change the dataframe to np.array 
    #
    # 算小类
    '''
    featuresTable = pd.read_pickle(fileLoc + '/primaryClassFeatures.pkl')
    featuresTable1 = pd.read_pickle(fileLoc + '/small_segmentFeatures.pkl')
    data_x = featuresTable.values
    data_y = featuresTable1.ix[:,-1].values
    labels = featuresTable1.ix[:, -1].values
    '''
    # 算全类
    featuresTable = pd.read_pickle(fileLoc + '/allClassFeatures.pkl')
    featuresTable1 = pd.read_pickle(fileLoc + '/segmentFeatures.pkl')
    data_x = featuresTable.values
    data_y = featuresTable1.ix[:, -1].values
    labels = featuresTable1.ix[:, -1].values

    '''
    #算重力分类
    featuresTable = pd.read_pickle(fileLoc + '/weightClassFeatures.pkl')
    data_x = featuresTable.values
    data_y = np.load(fileLoc + '/weightLabel.npy')[1:-1]
    featuresTable1 = pd.read_pickle(fileLoc + '/segmentFeatures.pkl')
    labels = featuresTable1.ix[:, -1].values
    '''
    '''
    # 算频率分类
    featuresTable = pd.read_pickle(fileLoc + '/frequenceClassFeatures.pkl')
    data_x = featuresTable.values
    data_y = np.load(fileLoc + '/frequenceLabel.npy')[1:-1]
    featuresTable1 = pd.read_pickle(fileLoc + '/middle_segmentFeatures.pkl')
    labels = featuresTable1.ix[:, -1].values

    '''
    '''
    #算高频率分类
    featuresTable = pd.read_pickle(fileLoc + '/highFrequence_ClassFeatures.pkl')
    data_x = featuresTable.values
    data_y = np.load(fileLoc + '/en_highFrequence_Label.npy')[1:-1]
    featuresTable1 = pd.read_pickle(fileLoc + '/highFrequence_segmentFeatures.pkl')
    labels = featuresTable1.ix[:, -1].values
    '''

    print data_y
    return data_x, data_y,labels

def normalization(trainData):
    trainData_normalized = preprocessing.normalize(trainData, norm='l2',axis = 0)
    #axis used to normalize the data along. If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
    return trainData_normalized


#http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
def kFoldValidation(data_x, data_y, data_labels,K, classifier):
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
        train_labels = data_labels[indexOfTrain]
        test_labels = data_labels[indexOfTest]
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
        for i in range(0,test_y.shape[0]):
            print test_labels[i],u''+test_y[i], u''+predict[i]
        print 'accuracy of kfold %d : %.2f%%' %(numOfk, (100 * metrics.accuracy_score(test_y, predict))) 
        numOfk = numOfk+1
        accuracy = accuracy + metrics.accuracy_score(test_y, predict)      
    print 'average accuracy: %.2f%%' % (100 * accuracy/K)
    return 100 * accuracy/K,confusion_mat
 

def LOOValidation(train_x, train_y,test_x, test_y,classifier):

    model = classifier(train_x, train_y)
    # print indexOfTrain

    # print "train" + str(metrics.accuracy_score(train_y, predict))
    predict = model.predict(test_x)
    #这些返回的是它的index
    confusion_mat = confusion_matrix(test_y, predict)
    print 'accuracy of leave one out: %.2f%%' % (100 * metrics.accuracy_score(test_y, predict))
    return 100 * metrics.accuracy_score(test_y, predict) , confusion_mat
 

if __name__ == '__main__':

   model_save_file = None
   model_save = {}
   
   test_classifiers = [#'NB', 
                       'KNN',
                       'LR',
                       'RF',
                       'DT',
                       #'SVM',
                       #'SVMCV',
                       'GBDT',
                       #'Adaboost',
                       #'MLP',
                       'VOTER']
   classifiers = {
                  #'NB':naive_bayes_classifier, 
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,
                  #'SVM':svm_classifier,
                #'SVMCV':svm_cross_validation,
                 'GBDT':gradient_boosting_classifier,
                 #'Adaboost':adaboost_classifier,
                 #'MLP': mlp_classifier
                 'VOTER': voter_classifier
                 }
   
   print 'reading training and testing data...'
   data_x,data_y,data_labels = readData('./a')
   test_x, test_y,test_labels = readData('./e')
   test_x1, test_y1, test_labels1 = readData('./b')
   test_x2, test_y2, test_labels2 = readData('./c')
   test_x3, test_y3, test_labels3 = readData('./d')
   test_x4, test_y4, test_labels4 = readData('./g')
   test_x5, test_y5, test_labels5 = readData('./h')
   test_x6, test_y6, test_labels6 = readData('./i')
   test_x7, test_y7, test_labels7 = readData('./j')
   test_x8, test_y8, test_labels8 = readData('./k')
   test_x9, tets_y9, test_labels9 = readData('./f')

   #print data_x.shape,test_x.shape
   num_train, num_feat = data_x.shape
   class_names = np.unique(data_y)
   for i in range(0,class_names.shape[0]):
       class_names[i] = str(0)+ class_names[i]
   #class_names = np.load('e/classNames.npy')
   
   #====================================================================================#

   #normalization is very important before you train the model!!!!!!!!!!
   print 'first you need to normalization the train set'
   #data_x = normalization(data_x)

   data_x=np.concatenate((data_x,test_x1,test_x2,test_x3,test_x4,test_x5,test_x6,test_x7,test_x8,test_x),axis=0)
   data_y= np.concatenate((data_y,test_y1,test_y2,test_y3,test_y4,test_y5,test_y6,test_y7,test_y8,test_y),axis=0)
   data_labels = np.concatenate((data_labels, test_labels1,test_labels2,test_labels3,test_labels4,test_labels5,test_labels6,test_labels7,test_labels8,test_labels), axis=0)
   #noemData= normalization(noemData)
   #data_x = noemData[0:num_train]
   #test_x = noemData[num_train:]
   #data_x = normalization(data_x)
   #test_x = normalization(test_x)
   '''
   pca = PCA(n_components='mle',whiten=False)
   pca.fit(data_x)
   print pca.explained_variance_ratio_
   print pca.explained_variance_
   print pca.n_components_
   data_x = pca.transform(data_x)
   '''
   #=====================================================================================#
   print '******************** Data Info *********************'
   print '#training data: %d, dimension of feature: %d' % (num_train, num_feat)


   for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        
        model = classifiers[classifier]
        accuracy,confusion_mat = kFoldValidation(data_x, data_y,data_labels, 6, model)
        #accuracy, confusion_mat = LOOValidation(data_x,data_y,test_x, test_y,model)
        print type(confusion_mat)
        
        print confusion_mat
        
        plt.figure()
        toolbox.plot_confusion_matrix(confusion_mat, classes=class_names, normalize=True,title='confusion matrix of '+classifier+' ACC :' + str(accuracy))
        plt.show()
    
