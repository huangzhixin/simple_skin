#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 07.11.2016
# File Name: calcStatisticalFeatures.py
# 
# Method Name : calcStatisticalFeatures
# 
# Description:  This file is used to calculate  statistical features(i.e mean, variance, min , max) for basic
# features ie  pressure, area and centre of weight x and y
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		   statistical features mean , standard deviation and min
#                  max values for basic features area, pressure, weight

# checking dimensions of data stream

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

settings = pd.read_pickle('settings.pkl')

frameFeatures = pd.read_pickle(settings['Name_of_run']+'/frameFeatures.pkl')

positionFeatures = pd.read_pickle(settings['Name_of_run']+'/directionFeatures.pkl') 

direction_start_frame = positionFeatures['direction_start_frame']
direction_end_frame = positionFeatures['direction_end_frame']

#first we need define all array of the features in python, in Matlab we do not need 
mean_area_feature = np.zeros((settings["number_of_events"]))
mean_pressure_feature = np.zeros((settings["number_of_events"]))
mean_weight_feature = np.zeros((settings["number_of_events"]))
min_area_feature = np.zeros((settings["number_of_events"]))
max_area_feature = np.zeros((settings["number_of_events"]))
min_pressure_feature = np.zeros((settings["number_of_events"]))
max_pressure_feature = np.zeros((settings["number_of_events"]))
min_weight_feature = np.zeros((settings["number_of_events"]))
max_weight_feature = np.zeros((settings["number_of_events"]))
std_area_feature = np.zeros((settings["number_of_events"]))
std_pressure_feature = np.zeros((settings["number_of_events"]))
std_weight_feature = np.zeros((settings["number_of_events"]))

for i in range(0,settings['number_of_events']):
  # type(areaFeature) =>pandas.core.series.Series
  # type(areaFeature.values) => numpy.ndarray
  # but np.mean can compatible with pandas.core.series.Series
  mean_area_feature[i] = np.mean(frameFeatures["areaFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
  mean_pressure_feature[i] = np.mean(frameFeatures["pressureFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
  mean_weight_feature[i] = np.mean(frameFeatures["weightFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])

  min_area_feature[i] = np.min(frameFeatures["areaFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
  max_area_feature[i] = np.max(frameFeatures["areaFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])

  min_pressure_feature[i] = np.min(frameFeatures["pressureFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
  max_pressure_feature[i] = np.max(frameFeatures["pressureFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])

  min_weight_feature[i] = np.min(frameFeatures["weightFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
  max_weight_feature[i] = np.max(frameFeatures["weightFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])

  std_area_feature[i] = np.std(frameFeatures["areaFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
  std_pressure_feature[i] = np.std(frameFeatures["pressureFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
  std_weight_feature[i] = np.std(frameFeatures["weightFeature"][int(direction_start_frame[i]):int(direction_end_frame[i])])
# end for

print'* saving Statistical Features *'

featureTable = {"mean_area_feature" : mean_area_feature,
                "mean_pressure_feature" : mean_pressure_feature,
                "mean_weight_feature" : mean_weight_feature,
                "min_area_feature" : min_area_feature,
                "max_area_feature" : max_area_feature,
                "min_pressure_feature" : min_pressure_feature,
                "max_pressure_feature" : max_pressure_feature,
                "min_weight_feature" : min_weight_feature,
                "max_weight_feature" : max_weight_feature,
                "std_area_feature" : std_area_feature,
                "std_pressure_feature" : std_pressure_feature,
                "std_weight_feature" : std_weight_feature
               }

featureTable = pd.DataFrame(featureTable)
featureTable.to_pickle(settings['Name_of_run']+'/statisticalFeatures.pkl')
