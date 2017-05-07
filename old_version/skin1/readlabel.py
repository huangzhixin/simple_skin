#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 01.11.2016
# File Name: readlabel.py
# 
# Method Name : readlabel
# 
# Description:  This method  is used read label from the text file
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  beginTime, endTime, class


# reading text file 

# Checking if file exists in given location 

import pandas as pd
import os

settings = pd.read_pickle('settings.pkl')


 
fileLoc = settings['File_location']+settings['lable_file']

if os.path.isfile(fileLoc) == False:
  print 'fopen '+ fileLoc +' failed!!'


print '* reading lablefile  * ' 

labelData = pd.read_csv(fileLoc,names = ['beginTime', 'endTime', 'class'])

settings['total_events'] = labelData.shape[0]

settings['number_of_events'] = settings['total_events'] - 2; # clear not required variables

settings.to_pickle('settings.pkl')
print settings
print '* save lablefile  * ' 
labelData.to_pickle(settings['Name_of_run']+'/labelData.pkl')

