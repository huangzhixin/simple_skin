#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 12.11.2016
#File Name: 	loadClassificationConfig.py
# 
# Method Name : loadClassificationConfig
# 
# Description:  This method is  specify the name of runs for classifcation 
# 
# Argument :	Name of the csv file with extenstion 
import pandas as pd
import pickle
import os  

# reading file name of the config file
configFileName = raw_input("Enter the name of config file,(clsConfig.csv) ")

#default file name case
if configFileName == '':
   configFileName = 'clsConfig.csv'

# importing pre structured config file
print 'Reading config file '
data_table = pd.read_csv(configFileName)

settings = pd.Series(data_table["value"].values, index = data_table["Name"].values)

"""
# change the string to number in settings 
for i in range(0,settings.shape[0]):
  if str.isdigit(settings.values[i]) == True:
    settings.values[i]=int(settings.values[i])
"""
settings.to_pickle('clsconfig.pkl')


print '* saving class config *'
print settings
