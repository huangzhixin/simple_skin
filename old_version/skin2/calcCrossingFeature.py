#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 07.11.2016
# Latest Date: 29.01.2017
# File Name: calcCrossingFeatures.py
# 
# Method Name : calcCrossingFeatures
# 
# Description:  This file is used to calculate the number of Zero crossings for pressure, area , 
#               and weight for each classread more about zero crossing  here .  
#               We calculate mean for each segment and us it as a threshold for calculate the number of crossing
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  	  crossingArea  [nX1]
#                 crossingPressure  [nX1]
#                 crossingWeight  [nX1]
#                 where n is the number of classes	   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolbox

def calcCrossingFeatures(settingName):
  settings = pd.read_pickle(settingName+'.pkl')

  activity_start_frame = np.load(settings['Name_of_run']+'/activity_start_frame.npy')
  activity_end_frame = np.load(settings['Name_of_run']+'/activity_end_frame.npy')

  frameFeatures = pd.read_pickle(settings['Name_of_run']+'/frameFeatures.pkl')

  #first we need define all array of the features in python, in Matlab we do not need
  crossingWeight = np.zeros((settings["number_of_events"]))
  crossingArea = np.zeros((settings["number_of_events"]))
  crossingPressure = np.zeros((settings["number_of_events"]))

  for i in range(0,settings['number_of_events']):
    # type(areaFeature) =>pandas.core.series.Series
    # type(areaFeature.values) => numpy.ndarray
    # but np.mean can compatible with pandas.core.series.Series
    meanWt = np.max(frameFeatures["weightFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])])/float(3)
    crossingWeight[i] = np.size(toolbox.crossing(frameFeatures["weightFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])],meanWt))

    meanArea = np.max(frameFeatures["areaFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])])/float(2)
    crossingArea[i] = np.size(toolbox.crossing(frameFeatures["areaFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])],meanArea))

    meanPressure = np.max(frameFeatures["pressureFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])])/float(3)
    crossingPressure[i] = np.size(toolbox.crossing(frameFeatures["pressureFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])],meanPressure))
  #end for

  print'* saving crossing features *'

  featureTable = {"crossingWeight" : crossingWeight,
                "crossingArea" : crossingArea,
                "crossingPressure" : crossingPressure
               }

  featureTable = pd.DataFrame(featureTable)
  featureTable.to_pickle(settings['Name_of_run']+'/crossingFeatures.pkl')

