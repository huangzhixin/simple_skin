#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 09.11.2016
# Latest Date: 29.01.2017
# File Name: saveFeaturesTable.py
# 
# Method Name : saveFeaturesTable
# 
# Description:  This file is save all the calculated features to make a
# table that can be used for training
# 
# Argument : 	  none  
# 			
# Output:  		   feature table with features and class

import numpy as np
import pandas as pd

def saveFeaturesTable(settingName):
  settings = pd.read_pickle(settingName +'.pkl')

  #frameFeatures was used to only calculte another following segmented features

  #frameFeatures = pd.read_pickle(settings['Name_of_run']+'/frameFeatures.pkl')


  frameFeatures_segmented = pd.read_pickle(settings['Name_of_run']+'/frameFeatures_segmented.pkl')

  statisticalFeatures = pd.read_pickle(settings['Name_of_run']+'/statisticalFeatures.pkl')

  directionFeatures = pd.read_pickle(settings['Name_of_run']+'/directionFeatures.pkl')

  peakFeatures = pd.read_pickle(settings['Name_of_run']+'/peakFeatures.pkl')

  frameHuFeatures = pd.read_pickle(settings['Name_of_run']+'/frameHuFeatures.pkl')

  crossingFeatures = pd.read_pickle(settings['Name_of_run']+'/crossingFeatures.pkl')

  labeles = pd.read_pickle(settings['Name_of_run']+'/labelData.pkl')['class']

  # in segmented_frame have 102 frames include 2 Synchronize frame, so we need to remove that
  labeles = pd.DataFrame(labeles.values[1:settings.number_of_events+1],columns=['class'])
  frameFeatures_segmented = pd.DataFrame(frameFeatures_segmented.values[1:settings.number_of_events+1],columns=['areaFeature','weightFeature','pressureFeature','centreOfWeightXFeature','centreOfWeightYFeature'])
  """
  #drop 方法有问题，他好像并没有删掉那一行，而是把这一行设为空，这样的方法作出来，labeles第一个值为nan
  frameFeatures_segmented = frameFeatures_segmented.drop(0)
  frameFeatures_segmented = frameFeatures_segmented.drop(101)
  frameFeatures_segmented = frameFeatures_segmented.reindex(index=statisticalFeatures.index)
  labeles = labeles.drop(0)
  labeles = labeles.drop(101)
  labeles = labeles.reindex(index=statisticalFeatures.index)
  """
  featuresTable = frameFeatures_segmented

  print "* saving and joining all Features in one FeaturesTable*"
  #featuresTable can not change yourself, the new joined table is in return value
  #newFeaturesTable = featuresTable.join([statisticalFeatures,directionFeatures,peakFeatures,frameHuFeatures,crossingFeatures,labeles])
  newFeaturesTable = featuresTable.join([statisticalFeatures,peakFeatures,frameHuFeatures,labeles])
  newFeaturesTable.to_pickle(settings['Name_of_run']+'/featuresTable.pkl')
