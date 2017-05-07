#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 02.11.2016
# Latest Date: 28.01.2017
# File Name: synchronizCurve.py
# 
# Method Name : synchronizCurve
# 
# Description:  This method is used to find the synchroniz frame for segmentData,you can manualy choose the number of synchroniz frame by figur, or you can get the number of synchroniz automaticly, but maybe make some error, you can choice witch ways is good. 

# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  (begin number of synchroniz,end number of synchroniz)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox import *
from scipy.signal import find_peaks_cwt
import scipy.signal as signal

def synchronizCurve(settingName1, settingName2, isDisplay=True):
  settings1 = pd.read_pickle(settingName1 + '.pkl')
  settings2 = pd.read_pickle(settingName2 + '.pkl')

  data_stream1 = np.load(settings1['Name_of_run']+'/thresholdCleanData.npy')
  data_stream2 = np.load(settings2['Name_of_run']+'/thresholdCleanData.npy')
  #print sum(sum(data_stream,0),0)[700]
  if isDisplay ==True:
    fig = plt.figure()
    #fig.hold()
    #plt.plot(sum(sum(data_stream1[:,:,1068:], 0), 0))
    #plt.plot(sum(sum(data_stream2[:,:,2262:], 0), 0), 'r-')
    plt.plot(sum(sum(data_stream1, 0), 0))
    fig = plt.figure()
    plt.plot(sum(sum(data_stream2, 0), 0), 'r-')
    plt.show()
  else:
    synchronisationStartFrame1 = 500-200
    synchronisationStartFrame2 = 900-200
    synchronisationEndFrame1 = 63200+300


    synchronisationEndFrame2 = 63600+300

    #synchronisationStartFrame = max(synchronisationStartFrame1,synchronisationStartFrame2)
    #synchronisationEndFrame = max(synchronisationEndFrame1, synchronisationEndFrame2)
    #print synchronisationEndFrame
    data_streamnew1 = np.zeros((data_stream1.shape[0],data_stream1.shape[1],synchronisationEndFrame1-synchronisationStartFrame1))
    data_streamnew2 = np.zeros((data_stream2.shape[0],data_stream2.shape[1],synchronisationEndFrame2-synchronisationStartFrame2))
    data_streamnew1 = data_stream1[:,:,synchronisationStartFrame1:synchronisationEndFrame1]
    data_streamnew2 = data_stream2[:,:,synchronisationStartFrame2:synchronisationEndFrame2]
    data_stream = np.concatenate((data_streamnew1, data_streamnew2), axis=1)
    print data_stream.shape
    fig = plt.figure()
    fig.hold()
    #plt.plot(sum(sum(data_stream, 0), 0))    #画出合成的图像
    plt.plot(sum(sum(data_streamnew1, 0), 0))
    plt.plot(sum(sum(data_streamnew2, 0), 0),'r-')
    plt.show()
    print '* saving combined of two stream data * '

    np.save(settings1['Name_of_run']+'/thresholdCleanData',data_stream)

    settings1['minTime'] = data_stream.shape[2]

    settings1.to_pickle(settingName1 + '.pkl')

def plotCurve(settingName, nameOfDatastream):
    settings = pd.read_pickle(settingName + '.pkl')
    data_stream = np.load(settings['Name_of_run'] + '/' + nameOfDatastream + '.npy')
    fig = plt.figure()

    plt.plot(sum(sum(data_stream, 0), 0))
    #print data_stream[:, :, 1000]
    #print data_stream[:, :, 2000]
    plt.show()
    plotOneFrame(data_stream[:, :, 1000])