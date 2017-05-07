#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 07.11.2016
#
#File Name: 	calcPeakCount.py
# 
# Method Name : calcPeakCount
# 
# Description:  Count number of peaks for each class for area, weight
#               and pressure
# 
#Argument :	none 
#
# Output:       peakCountPressure
#               peakCountArea
#               peakCountWeight

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolbox
from peakdetect import peakdetect


settings = pd.read_pickle('settings.pkl')

activity_start_frame = np.load(settings['Name_of_run']+'/activity_start_frame.npy')
activity_end_frame = np.load(settings['Name_of_run']+'/activity_end_frame.npy')

frameFeatures = pd.read_pickle(settings['Name_of_run']+'/frameFeatures.pkl')

peakCountWeight = np.zeros((settings["number_of_events"]))
peakCountArea = np.zeros((settings["number_of_events"]))
peakCountPressure = np.zeros((settings["number_of_events"]))

print '* caculating peakFeatures *'

for i in range(0,settings['number_of_events']):

  # type(areaFeature) =>pandas.core.series.Series
  # type(areaFeature.values) => numpy.ndarray
  # but np.mean can compatible with pandas.core.series.Series


  peaks = peakdetect(frameFeatures["weightFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])], lookahead=10)
  #peaks have two sublist one is peak another is valley，we just need peaks
  peakArray = np.array(peaks[0])
  peakCountWeight[i] = peakArray.shape[0]
  #print peakCountWeight[i]
  

  peaks = peakdetect(frameFeatures["areaFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])], lookahead=10)
  #peaks have two sublist one is peak another is valley，we just need peaks
  peakArray = np.array(peaks[0])
  peakCountArea[i] = peakArray.shape[0]
  print peakCountArea[i]
  
  peaks = peakdetect(frameFeatures["pressureFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])], lookahead=10)
  #peaks have two sublist one is peak another is valley，we just need peaks
  peakArray = np.array(peaks[0])
  peakCountPressure[i] = peakArray.shape[0]
  #print peakCountPressure[i]

  #meanWt = np.mean(frameFeatures["weightFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])])/float(2)
  #weightPeaks = toolbox.findPeaks(frameFeatures["weightFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])], int(meanWt))
  #peakCountWeight[i] = weightPeaks.shape[0]
  

  #meanArea = np.mean(frameFeatures["areaFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])])/float(2)
  #areaPeaks = toolbox.findPeaks(frameFeatures["areaFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])], meanArea)
  #peakCountArea[i] = areaPeaks.shape[0]


  #meanPressure = np.mean(frameFeatures["pressureFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])])/float(2)
  #pressurePeaks = toolbox.findPeaks(frameFeatures["pressureFeatureSmooth"][int(activity_start_frame[i+1]):int(activity_end_frame[i+1])], meanPressure)
  #peakCountPressure[i] = pressurePeaks.shape[0]

#end for

print '* saving peakFeatures *'

features = {"peakCountWeight":peakCountWeight,
            "peakCountArea" : peakCountArea,
            "peakCountPressure" : peakCountPressure
           }
featureTable = pd.DataFrame(features)

featureTable.to_pickle(settings['Name_of_run']+'/peakFeatures.pkl')

