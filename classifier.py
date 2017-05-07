# coding:utf-8

# version 3.0
# Author: Zhixin Huang
# Date: 04.05.2017
# File Name: 	classifier.py
#
# Class Name : classifier
#
# Description:  There are some classifiers of machine learning based on grid search,
# you can choose which one is better for you :)
#


import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import time
import utilities
import matplotlib.pyplot as plt


class Classifier():
    def __init__(self, setting_name='trainset', config_name='clsConfig'):
        self.setting_name = setting_name
        self.config_name = config_name
        self.settings = None
        self.utilities = utilities.Utilities()

    def load_config(self):
        print 'Reading config file '
        data_table = pd.read_csv(self.config_name + '.csv')

        self.settings = pd.Series(data_table.value.values, index=data_table.Name.values)

        # change the string to number in settings
        #for i in range(0, self.settings.shape[0]):
        #    if str.isdigit(self.settings.values[i]) == True:
        #        self.settings.values[i] = int(self.settings.values[i])

        if os.path.exists(self.setting_name) == False:
            os.popen('mkdir ' + self.setting_name).readlines()

    def read_data(self, feature_table_name='segment_features'):
        self.data_x = None
        self.data_y = None
        for i in range(0, self.settings.index.shape[0]):
            features_table = pd.read_pickle(self.settings.index[i] + '/' + feature_table_name + '.pkl')
            classes = features_table.pop('classes') #pop 方法把这一列从原数据弹出，原数据不保留该列
            if i == 0:
                self.data_x = features_table.values
                self.data_y = classes.values
            else:
                new_data_x = features_table.values
                new_data_y = classes.values
                self.data_x = np.concatenate((self.data_x, new_data_x), axis=0)
                self.data_y = np.concatenate((self.data_y, new_data_y), axis=0)

        #print features_table
        self.class_names = np.unique(self.data_y)

    def report(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def normalization(self, trainData):
        trainData_normalized = preprocessing.normalize(trainData, norm='l2', axis=0)
        # axis used to normalize the data along. If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
        return trainData_normalized

    # Multinomial Naive Bayes Classifier
    def naive_bayes_classifier(self, train_x, train_y, K_fold):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=0.01)
        model.fit(train_x, train_y)
        return model

    # KNN Classifier
    def knn_classifier(self, train_x, train_y, K_fold=6):
        from sklearn.neighbors import KNeighborsClassifier
        train_x = self.normalization(train_x)
        model = KNeighborsClassifier()
        params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'weights': ['uniform', 'distance']}
        grid = GridSearchCV(estimator=model, param_grid=params, cv=K_fold)
        grid = grid.fit(train_x, train_y)
        self.report(grid.cv_results_)
        best_parameters = grid.best_estimator_.get_params()
        # for para, val in best_parameters.items():
        #    print para, val
        model = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], algorithm=best_parameters['algorithm'],
                                     weights=best_parameters['weights'])
        # model.fit(train_x, train_y)
        accuracy, confusion_mat = self.k_fold_validation(train_x, train_y, K_fold, model)
        return model, accuracy, confusion_mat

    # Logistic Regression Classifier
    def logistic_regression_classifier(self, train_x, train_y, K_fold=6):
        from sklearn.linear_model import LogisticRegression
        # train_x = normalization(train_x)
        model = LogisticRegression(penalty='l2', class_weight='balanced')
        params = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']}
        grid = GridSearchCV(estimator=model, param_grid=params, cv=K_fold)
        grid = grid.fit(train_x, train_y)
        self.report(grid.cv_results_)
        best_parameters = grid.best_estimator_.get_params()
        # for para, val in best_parameters.items():
        #    print para, val
        model = LogisticRegression(penalty='l2', class_weight='balanced', solver=best_parameters['solver'])
        # model.fit(train_x, train_y)
        accuracy, confusion_mat = self.k_fold_validation(train_x, train_y, K_fold, model)
        return model, accuracy, confusion_mat

    # Random Forest Classifier
    def random_forest_classifier(self, train_x, train_y, K_fold=6):
        from sklearn.ensemble import RandomForestClassifier
        # train_x = normalization(train_x)
        model = RandomForestClassifier(class_weight='balanced')
        params = {'n_estimators': [18, 20, 22, 24, 26, 28, 30, 32], 'criterion': ['gini', 'entropy'],
                  'max_features': ['auto', 'log2', None]}
        grid = GridSearchCV(estimator=model, param_grid=params, cv=K_fold)
        grid = grid.fit(train_x, train_y)
        self.report(grid.cv_results_)
        best_parameters = grid.best_estimator_.get_params()
        # for para, val in best_parameters.items():
        #    print para, val
        model = RandomForestClassifier(class_weight='balanced', n_estimators=best_parameters['n_estimators'],
                                       criterion=best_parameters['criterion'],
                                       max_features=best_parameters['max_features'])
        # model.fit(train_x, train_y)
        accuracy, confusion_mat = self.k_fold_validation(train_x, train_y, K_fold, model)
        return model, accuracy, confusion_mat

    # Decision Tree Classifier
    def decision_tree_classifier(self, train_x, train_y, K_fold=6):
        from sklearn import tree
        # train_x = normalization(train_x)
        params = {'splitter': ['best', 'random'], 'criterion': ['gini', 'entropy'],
                  'max_features': ['auto', 'log2', None], 'class_weight': ['balanced', None]}
        model = tree.DecisionTreeClassifier()
        grid = GridSearchCV(estimator=model, param_grid=params, cv=K_fold)
        grid = grid.fit(train_x, train_y)
        self.report(grid.cv_results_)
        best_parameters = grid.best_estimator_.get_params()
        # for para, val in best_parameters.items():
        #    print para, val
        model = tree.DecisionTreeClassifier(splitter=best_parameters['splitter'],
                                            criterion=best_parameters['criterion'],
                                            max_features=best_parameters['max_features'],
                                            class_weight=best_parameters['class_weight'])
        # model.fit(train_x, train_y)
        accuracy, confusion_mat = self.k_fold_validation(train_x, train_y, K_fold, model)
        return model, accuracy, confusion_mat

    # SVM Classifier    #线性非线性核需要读文档
    def svm_classifier(self, train_x, train_y, K_fold=6):
        from sklearn.svm import SVC
        model = SVC()
        train_x = self.normalization(train_x)
        params = {'C': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 'kernel': ['poly'],
                  'degree': [2, 3, 4, 5, 6, 7, 8, 9], 'gamma': [0.01, 0.05, 0.1, 0, 2, 0.5, 1],
                  'class_weight': [None, 'balanced'], 'decision_function_shape': ['ovo', 'ovr', None]}
        # params = {'C': [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0], 'kernel': ['rbf', 'sigmoid'],'class_weight':[None, 'balanced']}
        grid = GridSearchCV(estimator=model, param_grid=params, cv=K_fold)
        grid = grid.fit(train_x, train_y)
        self.report(grid.cv_results_)
        # model = SVC(kernel='rbf',C=10, gamma=0.001)
        # model.fit(train_x, train_y)
        accuracy, confusion_mat = self.k_fold_validation(train_x, train_y, K_fold, model)
        return model, accuracy, confusion_mat

    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self, train_x, train_y, K_fold=6):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        params = {'loss': ['deviance'], 'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4], 'n_estimators': [100, 200, 300],
                  'max_depth': [3, 4, 5, 6, 7, 8]}
        # params = {'loss': ['deviance'], 'learning_rate': [0.05, 0.1 ],'n_estimators': [100,200], 'max_depth':[5,6]}
        grid = GridSearchCV(estimator=model, param_grid=params, cv=K_fold)
        grid = grid.fit(train_x, train_y)
        self.report(grid.cv_results_)
        best_parameters = grid.best_estimator_.get_params()
        # for para, val in best_parameters.items():
        #    print para, val
        model = GradientBoostingClassifier(loss=best_parameters['loss'], learning_rate=best_parameters['learning_rate'],
                                           n_estimators=best_parameters['n_estimators'],
                                           max_depth=best_parameters['max_depth'])
        # model.fit(train_x, train_y)
        accuracy, confusion_mat = self.k_fold_validation(train_x, train_y, K_fold, model)
        return model, accuracy, confusion_mat

    def voter_classifier(self, train_x, train_y, K_fold=6, models=[]):
        from sklearn.ensemble import VotingClassifier
        model = VotingClassifier(
            estimators=models, voting='soft')
        # model.fit(train_x, train_y)
        accuracy, confusion_mat = self.k_fold_validation(train_x, train_y, K_fold, model)
        return model, accuracy, confusion_mat

    # http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
    def k_fold_validation(self, data_x, data_y, K, model):
        kf = KFold(n_splits=K)
        confusion_mat = np.zeros((np.unique(data_y).shape[0], np.unique(data_y).shape[0]))
        accuracy = 0
        numOfk = 1
        for indexOfTrain, indexOfTest in kf.split(data_x):
            start_time = time.time()
            # print("%s %s" % (indexOfTrain, indexOfTest))
            train_x = data_x[indexOfTrain]
            test_x = data_x[indexOfTest]
            train_y = data_y[indexOfTrain]
            test_y = data_y[indexOfTest]
            start_time = time.time()
            model.fit(train_x, train_y)

            # print indexOfTrain
            print 'training took %fs!' % (time.time() - start_time)
            # predict = model.predict(train_x)
            # print "train" + str(metrics.accuracy_score(train_y, predict))
            predict = model.predict(test_x)
            """
            if model_save_file != None:
                model_save[classifier] = model
            """
            confusion_mat = confusion_mat + confusion_matrix(test_y, predict)
            print 'accuracy of kfold %d : %.2f%%' % (numOfk, (100 * metrics.accuracy_score(test_y, predict)))
            numOfk = numOfk + 1
            accuracy = accuracy + metrics.accuracy_score(test_y, predict)
        print 'average accuracy: %.2f%%' % (100 * accuracy / K)
        return 100 * accuracy / K, confusion_mat

    def auto_validation(self):
        test_classifiers = [  # 'NB',
            'KNN',
            'LR',
            'RF',
            'DT',
            # 'SVM',
            # 'SVMCV',
            'GBDT',
            # 'Adaboost',
            # 'MLP',
            'VOTER']

        classifiers = {
            # 'NB':naive_bayes_classifier,
            'KNN': self.knn_classifier,
            'LR': self.logistic_regression_classifier,
            'RF': self.random_forest_classifier,
            'DT': self.decision_tree_classifier,
            # 'SVM':self.svm_classifier,
            # 'SVMCV':self.svm_cross_validation,
            'GBDT': self.gradient_boosting_classifier,

            'VOTER': self.voter_classifier
        }

        self.load_config()
        self.read_data()
        num_train, num_feat = self.data_x.shape

        print '******************** Data Info *********************'
        print '#training data: %d, dimension of feature: %d' % (num_train, num_feat)

        models = []     #good model for voter classifier

        for classifier in test_classifiers:
            print '******************* %s ********************' % classifier

            model = classifiers[classifier]

            model, accuracy, confusion_mat = model(self.data_x, self.data_y, 2)

            #if classifier != 'VOTER':
            #    if accuracy > 65 :
            #        models.append(classifiers[classifier])


            print confusion_mat

            plt.figure()
            self.utilities.plot_confusion_matrix(confusion_mat, classes=self.class_names, normalize=True,
                                                 title='confusion matrix of ' + classifier + ' ACC :' + str(accuracy))
            plt.show()


if __name__ == "__main__":
    classifier = Classifier()
    classifier.auto_validation()