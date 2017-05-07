#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Create Date: 31.10.2016
# Latest Data: 28.01.2017
#File Name: 	loadConfig.py
# 
# Method Name : loadConfig
# 
# Description:  This method is used to load all the global settings, We are reading values from a csv file
# 
# Argument :	Name of the csv file with extenstion 

import pandas as pd
import pickle
import os  

def loadConfig(configFileName, settingName):
  # reading file name of the config file
  #configFileName = raw_input("Enter the name of config file,(config.csv) ")

  #default file name case
  #if configFileName == '':
  #   configFileName = 'config.csv'

  # importing pre structured config file
  print 'Reading config file '
  data_table = pd.read_csv(configFileName)

  settings = pd.Series(data_table.value.values, index = data_table.Name.values)

  # change the string to number in settings
  for i in range(0,settings.shape[0]):
    if str.isdigit(settings.values[i]) == True:
      settings.values[i]=int(settings.values[i])

  settings.to_pickle(settingName+'.pkl')


  print '* saving settings *'
  print settings
  #make direction
  # making a directory with name of run

  if os.path.exists(settings['Name_of_run']) == False:
     os.popen('mkdir '+ settings['Name_of_run']).readlines()


