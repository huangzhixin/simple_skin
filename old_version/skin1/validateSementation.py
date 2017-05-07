#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 01.11.2016
#File Name: 	validateSementation.py
# 
# Method Name : validateSementation
# 
# Description:  Generating images to validate segmentation against given
#               class
# 
# Argument :	none data is read from segmented_frame
#
# Output:  		  one image per activity

import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


settings = pd.read_pickle('settings.pkl')
segmented_frame = np.load(settings['Name_of_run']+'/segmented_frame.npy')
labelData = pd.read_pickle(settings['Name_of_run']+'/labelData.pkl')

print segmented_frame[:,:,58]
for i in range(1,settings["total_events"]-1):
  #fig 0 and fig 101 is synchronizframe
  fig = plt.matshow(segmented_frame[:,:,i],fignum=0,cmap=plt.cm.viridis)
  filepath = settings['Name_of_run']+"/_"+str(i)+"_"+labelData["class"][i]+".png"
  print filepath
  fig.write_png(filepath)

