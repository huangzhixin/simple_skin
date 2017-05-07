#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 08.11.2016
# Latest Date: 29.01.2017
#
# Method Name : calcFrameHuFeatures
# 
# Description:  This file is used to calculate 7 hu  features  for the
#               segmented frames
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  7 hu features calculated 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def calcFrameHuFeature(settingName):
  settings = pd.read_pickle(settingName+'.pkl')

  number_of_frames = settings['number_of_events']

  segmented_frame = np.load(settings['Name_of_run']+'/segmented_frame.npy')

  frameHuFeatures = np.zeros((number_of_frames,7))

  print '* Calculating hu features  * '
  # in segmented_frame have 102 frames include 2 Synchronize frame
  for i in range(0,number_of_frames):

    frame = segmented_frame[:,:,i+1]
    frameHuFeatures[i] = cv2.HuMoments(cv2.moments(frame)).flatten() #flatten: from array[[]] to array[]
  
  #end for

  #print frameHuFeatures.shape
  print '* Saving hu features  * '

  features = {'frameHuFeatures1': frameHuFeatures[:,0],
            'frameHuFeatures2': frameHuFeatures[:,1],
            'frameHuFeatures3': frameHuFeatures[:,2],
            'frameHuFeatures4': frameHuFeatures[:,3],
            'frameHuFeatures5': frameHuFeatures[:,4],
            'frameHuFeatures6': frameHuFeatures[:,5],
            'frameHuFeatures7': frameHuFeatures[:,6]
           }

  featureTable = pd.DataFrame(features)

  featureTable.to_pickle(settings['Name_of_run']+'/frameHuFeatures.pkl')
    
