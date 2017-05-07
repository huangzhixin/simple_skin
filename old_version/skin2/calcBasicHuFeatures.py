#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 08.11.2016
#
# File Name: calcBasicHuFeatures.py
# 
# Method Name : calcBasicHuFeatures
# 
# Description:  This file is used to calculate  hu moments for basic
# features is  pressure, area and centre of weight x and y
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		   hu moment features for  pressure, area, weight and
#                  Centre of weight 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def calcBasicHuFeatures(settingName):
  settings = pd.read_pickle(settingName+'.pkl')

  number_of_frames = settings['number_of_events']

  #segmented_frame = np.load(settings['Name_of_run']+'/segmented_frame.npy')

  frameFeatures = pd.read_pickle(settings['Name_of_run']+'/frameFeatures.pkl')

  positionFeatures = pd.read_pickle(settings['Name_of_run']+'/directionFeatures.pkl')

  direction_start_frame = positionFeatures['direction_start_frame']
  direction_end_frame = positionFeatures['direction_end_frame']

  humoments_area_feature = np.zeros((number_of_frames,7))
  humoments_pressure_feature = np.zeros((number_of_frames,7))
  humoments_weight_feature = np.zeros((number_of_frames,7))
  humoments_centreOfWeightXFeature = np.zeros((number_of_frames,7))
  humoments_centreOfWeightYFeature = np.zeros((number_of_frames,7))

  print '* Calculating basic hu features  * '

  for i in range(0,settings['number_of_events']):

    # type(areaFeature) =>pandas.core.series.Series
    # type(areaFeature.values) => numpy.ndarray
    areaFeature = frameFeatures["areaFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])]

    humoments_area_feature[i] =  cv2.HuMoments(cv2.moments(areaFeature.values)).flatten()

    pressureFeature = frameFeatures["pressureFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])]
    humoments_pressure_feature[i] = cv2.HuMoments(cv2.moments(pressureFeature.values)).flatten()
  
    weightFeature = frameFeatures["weightFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])]
    humoments_weight_feature[i] = cv2.HuMoments(cv2.moments(weightFeature.values)).flatten()

    centreOfWeightXFeature = frameFeatures["centreOfWeightXFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])]
    humoments_centreOfWeightXFeature[i] = cv2.HuMoments(cv2.moments(centreOfWeightXFeature.values)).flatten()

    centreOfWeightYFeature = frameFeatures["centreOfWeightYFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])]
    humoments_centreOfWeightYFeature[i] = cv2.HuMoments(cv2.moments(centreOfWeightYFeature.values)).flatten()
  # end for

  print '* saving basic HuFeatures *'

