# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 11.03.2017
# Latest Date: 11.03.2017
# File Name: calcAreaFeatures.py
# 
# Method Name : calcAreaFeatures
# 
# Description:  This file is used to calculate area features in form Mask for the segment data stream
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  Masks features

# checking dimensions of data stream

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
from toolbox import *


def calcAreaFeature(settingName, nameOfData, saveName):
    print '* calculating frame features * '

    settings = pd.read_pickle(settingName + '.pkl')

    data_stream = np.load(settings['Name_of_run'] + '/' + nameOfData + '.npy')
    activity_start_frame = np.load(settings['Name_of_run'] + '/activity_start_frame.npy')
    activity_end_frame = np.load(settings['Name_of_run'] + '/activity_end_frame.npy')
    labelData = pd.read_pickle(settings['Name_of_run'] + '/labelData.pkl')

    # data_stream = np.load(settings['Name_of_run'] + '/combineData.npy')
    number_of_frames = data_stream.shape[2]

    number_of_events = settings['number_of_events']
    rows = data_stream.shape[0]
    cols = data_stream.shape[1]
    bwMasks = np.zeros((rows, cols, number_of_events))  #存的每一段数据的面积分布，用1和0表示
    Ydata = np.zeros(number_of_events+2)      #存的是每一段数据合成一片后的面积大小用于显示
    label = labelData["class"]

    for i in range(1, number_of_events+1):
        segmentedData = data_stream[:, :, activity_start_frame[i]:activity_end_frame[i]]
        maxs = np.max(segmentedData, axis=2)
        mins = np.min(segmentedData, axis=2)
        means = np.mean(segmentedData, axis=2)
        #means = (maxs - mins)/2
        bwMasks[:, :, i-1] = (maxs-mins) > means
        Ydata[i] = np.sum(bwMasks[:, :, i-1])

    plotScatterDat(range(0,number_of_events+2), Ydata, label)
    np.save(settings['Name_of_run'] + '/'+saveName, bwMasks)