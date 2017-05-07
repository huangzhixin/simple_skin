#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 05.11.2016
# Latest Date: 29.01.2017
# File Name: calcSegmentedFrameFeatures.py
# 
# Method Name : calcSegmentedFrameFeatures
# 
# Description:  This file is used to calculate basic features  weight, area
# , and cetre of weight for the segmented frames
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  basic features(area, weight, pressure) 

# checking dimensions of data stream

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calcSegmentedFrameFeatures(settingName):
  print '* calculating Segmented frame features * '

  settings = pd.read_pickle(settingName + '.pkl')

  segmented_frame = np.load(settings['Name_of_run']+'/segmented_frame.npy')
  data_stream = np.load(settings['Name_of_run']+'/thresholdCleanData.npy')

  number_of_segments = segmented_frame.shape[2]

  rows = segmented_frame.shape[0]
  cols = segmented_frame.shape[1]

  # prelocating
  areaFeature_segmented = np.zeros((number_of_segments,1))
  pressureFeature_segmented = np.zeros((number_of_segments,1))
  weightFeature_segmented = np.zeros((number_of_segments,1))
  centreOfWeightXFeature_segmented = np.zeros((number_of_segments,1))
  centreOfWeightYFeature_segmented = np.zeros((number_of_segments,1))

  threshold = np.mean(segmented_frame,axis=2)

  # calculating mask frames having more value than threshold

  for i in range(0,number_of_segments):
    #包括前后两个同步帧
    frame = segmented_frame[:,:,i]
    bwMask = np.zeros(frame.shape)
    #bwMask = frame > threshold
    threshold = data_stream[:,:,5]
    bwMask = frame > threshold
    #当a和b为array时， a * b 计算了a和b的数量积（对应Matlab的 a .* b ），
    # dot(a, b) 计算了a和b的矢量积（对应Matlab的 a * b ）
    #in python sum(mat) == in matlab sum(sum(mat))

    weight = np.sum(frame*bwMask)
    area = np.sum(bwMask)
    if area > 10 :
      pressure = np.sum(bwMask*frame)/float(area)
    else:
      pressure = 0

    #第一次听说这么算重心的方法，牛！
    X,Y = np.meshgrid(range(0,frame.shape[1]),range(0,frame.shape[0]))

    if area > 10 :
      centerOfWeightX = np.sum(frame*X)/float(np.sum(frame))
      centerOfWeightY = np.sum(frame*Y)/float(np.sum(frame))
    else:
      centerOfWeightX = frame.shape[0]/2
      centerOfWeightY = frame.shape[1]/2

    areaFeature_segmented[i] = area
    weightFeature_segmented[i] = weight
    pressureFeature_segmented[i]= pressure
    centreOfWeightXFeature_segmented[i] = centerOfWeightX
    centreOfWeightYFeature_segmented[i] = centerOfWeightY

  print '* saving frame features * '
  #format : from array n*1 [[,,,]]to array n [,,,,]

  features = {'areaFeature':areaFeature_segmented[:,0],
            'weightFeature':weightFeature_segmented[:,0],
            'pressureFeature':pressureFeature_segmented[:,0],
            'centreOfWeightXFeature':centreOfWeightXFeature_segmented[:,0],
            'centreOfWeightYFeature':centreOfWeightYFeature_segmented[:,0]}

  featureTable = pd.DataFrame(features)

  featureTable.to_pickle(settings['Name_of_run']+'/frameFeatures_segmented.pkl')
