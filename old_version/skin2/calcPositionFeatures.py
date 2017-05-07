#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 05.11.2016
# Latest Date: 29.01.2017
# File Name: calcpositionFeatures.py
# 
# Method Name : calcpositionFeatures
# 
# Description:  This file is used to calculate start , middle and end
# location  of singal on matrix
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  	  startPos_x 
#                 middlePos_x
#                 endPos_x
#                 startPos_y
#                 middlePos_y
#                 endPos_y
#                 dir_x
#                 dir_y    

# calculate size

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calcPositionFeatures(settingName):
  settings = pd.read_pickle(settingName + '.pkl')

  data_stream = np.load(settings['Name_of_run']+'/thresholdCleanData.npy')
  activity_start_frame = np.load(settings['Name_of_run']+'/activity_start_frame.npy')
  activity_end_frame = np.load(settings['Name_of_run']+'/activity_end_frame.npy')
  number_of_frames = data_stream.shape[2]

  rows = data_stream.shape[0]
  cols = data_stream.shape[1]

  #define the all array value what we use
  maxValue = np.zeros((number_of_frames))
  index = np.zeros((number_of_frames))
  peak_x = np.zeros((number_of_frames))
  peak_y = np.zeros((number_of_frames))
  direction_start_frame = np.zeros((settings["number_of_events"]))
  direction_end_frame =   np.zeros((settings["number_of_events"]))
  dir_middle_x =          np.zeros((settings["number_of_events"]))
  dir_middle_y =          np.zeros((settings["number_of_events"]))
  dir_end_x =             np.zeros((settings["number_of_events"]))
  dir_end_y =             np.zeros((settings["number_of_events"]))
  dir_x =                 np.zeros((settings["number_of_events"]))
  dir_y =                 np.zeros((settings["number_of_events"]))

  print'* finding the position of peak in every frame  *'
  for i in range(0, number_of_frames):            #这部分很可能出问题，算出的x和y轴和matlab是反的
                                                #我做的程序和matlab的数据x和y是颠倒的！！！！！！去readFile查原因
    frame = data_stream[:,:,i]
    maxValue[i] = np.max(frame)
    index[i] = np.argmax(frame)+1
    peak_y[i] = np.mod(index[i],rows)-1
    if peak_y[i] == -1:                           #judge edge
      peak_y[i] = cols
    peak_x[i] = int(np.round(index[i] / float(cols))-1)


  print'* finding the direction of active in every segment  *'
  #constructing direction vector for each segment，
  for i in range(0,settings["number_of_events"]):
    x_loc = peak_x[int(activity_start_frame[i+1]):int(activity_end_frame[i+1])]     #+1 means remove the first and last Synchronize frame
    y_loc = peak_y[int(activity_start_frame[i+1]):int(activity_end_frame[i+1])]
    max_val = maxValue[int(activity_start_frame[i+1]):int(activity_end_frame[i+1])]
    dictTable = { 'x_loc':x_loc,
                'y_loc':y_loc,
                'max_val':max_val,
                'frame_number':range(int(activity_start_frame[i+1]),int(activity_end_frame[i+1]))}
    tables = pd.DataFrame(dictTable)
    selectedDirectionTable = tables.sort(columns='max_val',ascending=False)
    #sort with descending, default is True = ascend
    #这里一定要注意当你使用sort，tables自己并没有改变，sort后的新排序的数据在返回值里！！！
    selectedDirectionTable = selectedDirectionTable[0:60]
    #if i == 0:
    #  print selectedDirectionTable
    selectedDirectionTable = selectedDirectionTable.sort(columns='frame_number')     #sort with ascend
    #if i == 0:
    #  print selectedDirectionTable
    direction_start_frame[i] = int(np.min(selectedDirectionTable["frame_number"]))
    direction_end_frame[i] = int(np.max(selectedDirectionTable["frame_number"]))
    """"
    #flowing code is from matlab
    #but in python too many array need to define, so I just write them sample and integrated
    x(i,:) = selectedDirectionTable.x_loc ;
    y(i,:) = selectedDirectionTable.y_loc ;

    startPos_x(i,:) = mean( x(i,1:10));
    middlePos_x(i,:) = mean( x(i,25:35));
    endPos_x(i,:) = mean( x(i,50:60));
    startPos_y(i,:) = mean( y(i,1:10));
    middlePos_y(i,:) = mean( y(i,25:35));
    endPos_y(i,:) = mean( y(i,50:60));

    dir_middle_x(i,:) = middlePos_x(i,:) - startPos_x(i,:);
    dir_middle_y(i,:) = middlePos_y(i,:) - startPos_y(i,:);
    dir_end_x(i,:) = endPos_x(i,:) - middlePos_x(i,:);
    dir_end_y(i,:) = endPos_y(i,:) - middlePos_y(i,:);
    dir_x(i,:) = endPos_x(i,:) - startPos_x(i,:);
    dir_y(i,:) = endPos_y(i,:) - startPos_y(i,:);
    """
    x = selectedDirectionTable.x_loc ;
    y = selectedDirectionTable.y_loc ;

    startPos_x = np.mean( x[0:10])
    middlePos_x = np.mean( x[25:35])
    endPos_x = np.mean( x[50:60])
    startPos_y = np.mean( y[0:10])
    middlePos_y = np.mean( y[25:35])
    endPos_y = np.mean( y[50:60])

    dir_middle_x[i] = middlePos_x - startPos_x
    dir_middle_y[i] = middlePos_y - startPos_y
    dir_end_x[i] = endPos_x - middlePos_x
    dir_end_y[i] = endPos_y - middlePos_y
    dir_x[i] = endPos_x - startPos_x
    dir_y[i] = endPos_y - startPos_y
    #end for i

  print'* saving direction features *'

  featureTable = {'direction_start_frame':direction_start_frame,
                'direction_end_frame':direction_end_frame,
                'dir_middle_x':dir_middle_x,
                'dir_middle_y':dir_middle_y,
                'dir_end_x':dir_end_x,
                'dir_end_y':dir_end_y,
                'dir_x':dir_x,
                'dir_y':dir_y}
  featureTable = pd.DataFrame(featureTable)
  featureTable.to_pickle(settings['Name_of_run']+'/directionFeatures.pkl')
