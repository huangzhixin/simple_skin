#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 31.10.2016
# File Name: thresholdClean.py
# 
# Method Name : thresholdClean
# 
# Description:  This method is used to apply a threhsold to reduce noice
#               and replace values below threshold with 1 i.e(the lowest value in the mat)
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  n X n data read from  text file

#load config and data_stream

import pandas as pd
import pickle
import os
import numpy as np


settings = pd.read_pickle('settings.pkl')

data_stream = np.load(settings['Name_of_run']+'/data_stream.npy')

print'* applying threshold clean*'
print'* threshold value *'
print settings["clean_threshold"]
print data_stream.shape
meanValue = np.mean(data_stream,axis=2)
print meanValue
clean_threshold = 0
data_stream1 = data_stream.copy() 
if settings['clean_threshold'] == 'mean':
  print'* clean the data with mean*'
  clean_threshold = data_stream[:,:,100]
  for i in range(0,settings['minTime']):
    data_stream1[:,:,i] = data_stream[:,:,i] - clean_threshold
  data_stream1[data_stream1 < 1] = 1
  
else :
  clean_threshold = settings['clean_threshold']
  #clean_threshold = 20
  data_stream[data_stream <clean_threshold] = 1



print '* saving threshold clean data * '

np.save(settings['Name_of_run']+'/thresholdCleanData',data_stream1)
