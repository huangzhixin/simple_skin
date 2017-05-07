# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 04.11.2016
# Latest Date: 03.04.2017
# File Name: calcFrameFeatures.py
# 
# Method Name : calcFrameFeatures
# 
# Description:  This file is used to calculate basic features  weight, area , and cetre of weight for the data stream
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  basic features(area, weight, pressure) 

# checking dimensions of data stream

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import toolbox


def calcFrameFeature(settingName, nameOfData, saveName):
    print '* calculating frame features * '

    settings = pd.read_pickle(settingName + '.pkl')

    data_stream = np.load(settings['Name_of_run'] + '/' + nameOfData + '.npy')
    # data_stream = np.load(settings['Name_of_run'] + '/combineData.npy')
    number_of_frames = data_stream.shape[2]

    rows = data_stream.shape[0]
    cols = data_stream.shape[1]

    # prelocating
    areaFeature = np.zeros((number_of_frames))
    pressureFeature = np.zeros((number_of_frames))
    weightFeature = np.zeros((number_of_frames))
    centreOfWeightXFeature = np.zeros((number_of_frames))
    centreOfWeightYFeature = np.zeros((number_of_frames))
    pressureFeatureSmooth = np.zeros((number_of_frames))
    areaFeatureSmooth = np.zeros((number_of_frames))
    weightFeatureSmooth = np.zeros((number_of_frames))

    threshold = np.median(data_stream, axis=2)
    # 这里的threshold可能和matlab算得不太一样，导致后面的bwMask算得可能也不一样，导致这里的所有feature算得都和matlab不一样
    # octave 无法用mean2，后面在实验用matlab试试


    # calculating mask frames having more value than threshold

    for i in range(0, number_of_frames):

        frame = data_stream[:, :, i]
        bwMask = np.zeros(frame.shape)
        bwMask = frame > threshold

        # 当a和b为array时， a * b 计算了a和b的数量积（对应Matlab的 a .* b ），
        # dot(a, b) 计算了a和b的矢量积（对应Matlab的 a * b ）
        # in python sum(mat) == in matlab sum(sum(mat))

        weight = np.sum(frame * bwMask)
        area = np.sum(bwMask)
        if area > 10:
            pressure = np.sum(bwMask * frame) / float(area)
        else:
            pressure = 0

        # 第一次听说这么算重心的方法，牛！
        X, Y = np.meshgrid(range(0, frame.shape[1]), range(0, frame.shape[0]))

        if area > 10:
            centerOfWeightX = np.sum(frame * X) / float(np.sum(frame))
            centerOfWeightY = np.sum(frame * Y) / float(np.sum(frame))
        else:
            centerOfWeightX = frame.shape[0] / 2
            centerOfWeightY = frame.shape[1] / 2

        areaFeature[i] = area
        weightFeature[i] = weight
        pressureFeature[i] = pressure
        centreOfWeightXFeature[i] = centerOfWeightX
        centreOfWeightYFeature[i] = centerOfWeightY
    # end for



    pressureFeatureSmooth = toolbox.dofilter(pressureFeature)
    areaFeatureSmooth = toolbox.dofilter(areaFeature)
    weightFeatureSmooth = toolbox.dofilter(weightFeature)

    # pressureFeatureSmooth[1:60] = 0
    # areaFeatureSmooth[1:60] = 0
    # weightFeatureSmooth[1:60] = 0

    """
  
    https://docs.scipy.org/doc/scipy-0.18.1/reference/signal.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html
  
    #applying a lowpass filter
    pressureFeatureSmooth = dofilter(pressureFeature)
    areaFeatureSmooth =  dofilter(areaFeature);
    weightFeatureSmooth =  dofilter(weightFeature)
    pressureFeatureSmooth(1:60) = 0
    areaFeatureSmooth(1:60) = 0
    weightFeatureSmooth(1:60) = 0
    """

    print '* saving frame features * '
    # format : from array n*1 [[,,,]]to array n [,,,,]

    features = {'areaFeature': areaFeature,
                'weightFeature': weightFeature,
                'pressureFeature': pressureFeature,
                'centreOfWeightXFeature': centreOfWeightXFeature,
                'centreOfWeightYFeature': centreOfWeightYFeature,
                'pressureFeatureSmooth': pressureFeatureSmooth,
                'areaFeatureSmooth': areaFeatureSmooth,
                'weightFeatureSmooth': weightFeatureSmooth
                }

    featureTable = pd.DataFrame(features)

    # featureTable.to_pickle(settings['Name_of_run']+'/frameFeatures.pkl')
    featureTable.to_pickle(settings['Name_of_run'] + '/' + saveName + '.pkl')


def calcDiffFrameFeature(settingName):
    settings = pd.read_pickle(settingName + '.pkl')
    frontFrameFeatures = pd.read_pickle(settings['Name_of_run'] + '/' + 'frontFrameFeature' + '.pkl')
    backFrameFeatures = pd.read_pickle(settings['Name_of_run'] + '/' + 'backFrameFeature' + '.pkl')
    diffFrameFeature = frontFrameFeatures - backFrameFeatures
    diffFrameFeature.to_pickle(settings['Name_of_run'] + '/' + 'diffFrameFeature' + '.pkl')