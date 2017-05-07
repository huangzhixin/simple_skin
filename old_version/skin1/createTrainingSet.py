#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 12.11.2016
#File Name: 	createTrainingSet.py
# 
# Method Name : createTrainingSet
# 
# Description:  This method is combine all the featuresTable from the
#               various runs and create a training set
# 
# Argument :	none 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

clsConfig = pd.read_pickle('clsconfig.pkl')

number_of_runs = clsConfig.shape[0]

trainingSet = pd.DataFrame()

#the first line is kfold, so we just begin at 1
for i in range(1,number_of_runs):

  run_name = clsConfig.index[i]

  print run_name

  fileLoc = run_name+'/featuresTable.pkl'

  df = pd.read_pickle(fileLoc)
  #every operation on DataFrame, the DF was not changed self, the change will be return to new value
  trainingSet = trainingSet.append(df)

print '* saving trainingSet *'
trainingSet.to_pickle('featuresTable.pkl')

print '* training set created. can be used in classification learner *'
  
