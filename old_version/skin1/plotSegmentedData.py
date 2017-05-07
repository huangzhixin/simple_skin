#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 02.11.2016
# File Name: plotSegmentedData.py
# 
# Method Name : plotSegmentedData
# 
# Description:  This file plots segmented graphs for pressure , area and
# weight feature, marked with the classes to validate segmentation and also
# visualize  the features for each class
# 
# Argument : 	  none  
# 			
# Output:  		   figure for  pressure , area and weight

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toolbox

settings = pd.read_pickle('settings.pkl')
labelData = pd.read_pickle(settings['Name_of_run']+'/labelData.pkl')
data_stream = np.load(settings['Name_of_run']+'/thresholdCleanData.npy')
activity_start_frame = np.load(settings['Name_of_run']+'/activity_start_frame.npy')
activity_end_frame = np.load(settings['Name_of_run']+'/activity_end_frame.npy')

data = np.sum(data_stream,axis=(0,1))

label = labelData["class"]

toolbox.plotSegmentedData(data,label,activity_start_frame,activity_end_frame)

"""
unique_classes = labelData["class"].unique()
num_of_classes  = unique_classes.shape[0]
lines = []
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1.0, num_of_classes))

fig = plt.figure()
fig.hold()
fig.suptitle('Segmented sum', fontsize=12)


for i in range(0,settings["total_events"]):
  index = 1
  currentClass = labelData["class"][i]
  for j in range(0,num_of_classes):
    if unique_classes[j]==currentClass:
      index = j
      c = colors[index,:]
      data = np.sum(data_stream[:,:,activity_start_frame[i]:activity_end_frame[i]],axis=(0,1))
      plt.plot(range(int(activity_start_frame[i]),int(activity_end_frame[i])),data,color=c,linewidth=1)

plt.xlabel('frame number')

plt.ylabel('Area value')


#only for test the color map
for k in range(0,num_of_classes):
  c = colors[k,:]
  line, = plt.plot(range(1,10),color=c,linewidth=4)
  lines.append(line)

fig.legend(lines,unique_classes,'upper right')

plt.show()

"""
    
