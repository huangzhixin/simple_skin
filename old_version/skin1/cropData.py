#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 29.10.2016
# File Name: cropData.m
# 
# Method Name : cropData
# 
# Description:  This method crops the read data stream to smaller size as
#               specified in config csv
# 

# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  n X n data read from  text file

import pandas as pd
import numpy as np

settings = pd.read_pickle('settings.pkl')

data_stream = np.load(settings['Name_of_run']+'/data_stream.npy')

print '* Cropping read data * '

#data_stream_new = np.zeros((settings['Crop_row end']-settings['Crop_row begin']+1,settings['Crop_column_end']-settings['Crop_column_begin']+1,settings['minTime']))

mat_size = settings['mat_size_x'] * settings['mat_size_y']

'''
for i in range(0,settings['minTime']-1):

  frameTemp = data_stream[:,:,i]
  frame_cropped = frameTemp[settings['Crop_row begin']-1:settings['Crop_row end'],settings['Crop_column_begin']-1:settings['Crop_column_end']]
  data_stream_new[:,:,i]=frame_cropped
'''

data_stream_new = data_stream[settings['Crop_row begin']-1:settings['Crop_row end'],settings['Crop_column_begin']-1:settings['Crop_column_end'],:]

#In python the first index is 0, so you need settings['Crop_row begin']-1

data_stream =  data_stream_new

print '* saving crop data * '

np.save(settings['Name_of_run']+'/data_stream',data_stream)


