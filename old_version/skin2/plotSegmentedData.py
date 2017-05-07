#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 02.11.2016
# Latest Date: 28.01.2017
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
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
plt.rcParams['font.sans-serif']=['simhei'] #用来正常显示中文标签

def plotSegmentedData(seetingName, nameOfFrameFeature, nameOfFeature,nameOfStart, NameOfEnd,NameOfLabel,isSort=False):
  settings = pd.read_pickle(seetingName + '.pkl')
  #labelData = pd.read_pickle(settings['Name_of_run']+'/'+nameOfLabel+'.pkl')
  frameFeatures = pd.read_pickle(settings['Name_of_run'] + '/'+nameOfFrameFeature+'.pkl')
  data = frameFeatures[nameOfFeature]
  #data_stream = np.load(settings['Name_of_run'] + '/combineData.npy')
  activity_start_frame = np.load(settings['Name_of_run']+'/'+nameOfStart+'.npy')
  activity_end_frame = np.load(settings['Name_of_run']+'/'+NameOfEnd+'.npy')

  #data = np.sum(data_stream,axis=(0,1))

  #label = labelData["class"]
  label = np.load(settings['Name_of_run'] + '/'+NameOfLabel+'.npy')
  toolbox.plotSegmentedDatas(data,label,activity_start_frame,activity_end_frame,isSort=isSort)

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
    
