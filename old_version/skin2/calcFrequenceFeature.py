# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 11.03.2017
# Latest Date: 11.03.2017
# File Name: calcFrequenceFeatures.py
#
# Method Name : calcFrequenceFeatures
#
# Description:  This file is used to calculate Frequence  for the segment data stream
#
# Argument : 	  none  (values are read from settings loaded in load settings )
#
# Output:  		  Frequence features

# checking dimensions of data stream

import numpy as np
import pylab as pl
from toolbox import *


def calcFrequenceFeature(settingName, nameOfFrameFeature,nameOfFeature, saveName):
    settings = pd.read_pickle(settingName + '.pkl')
    frameFeatures = pd.read_pickle(settings['Name_of_run'] + '/' + nameOfFrameFeature + '.pkl')
    data_stream = frameFeatures[nameOfFeature]
    #data_stream = np.load(settings['Name_of_run'] + '/' + nameOfData + '.npy')
    activity_start_frame = np.load(settings['Name_of_run'] + '/sorted_activity_start_frame.npy')
    activity_end_frame = np.load(settings['Name_of_run'] + '/sorted_activity_end_frame.npy')
    labelData =  np.load(settings['Name_of_run'] + '/'+'sorted_label'+'.npy')
    number_of_events = labelData.shape[0]
    #rows = data_stream.shape[0]
    #cols = data_stream.shape[1]

    #label = labelData["class"]



    for i in range(1, number_of_events + 1):
        segmentedData = data_stream[activity_start_frame[i]:activity_end_frame[i]]
        #inputSignal = sum(sum(segmentedData, 0), 0)
        inputSignal = segmentedData
        calcMainFrequence(inputSignal, isDisply=True)

    #plotScatterDat(range(0, number_of_events + 2), Ydata, label)
    #np.save(settings['Name_of_run'] + '/' + saveName, bwMasks)


def calcMainFrequence(inputSignal, isDisply=False):
    sampling_rate = 80
    fft_size = inputSignal.shape[0]
    t = np.arange(0, int(399/41), 1.0 / sampling_rate)

    x = inputSignal
    xs = x[:fft_size]
    xf = np.fft.rfft(xs) / fft_size
    freqs = np.linspace(0, sampling_rate / 2, fft_size / 2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    if isDisply == True:
        pl.figure(figsize=(8, 4))
        pl.subplot(211)
        pl.plot(t[:fft_size], xs)
        pl.xlabel(u"time(s)")
        pl.title(u"time domain and frequence domain")
        pl.subplot(212)
        pl.plot(freqs, xfp)
        pl.xlabel(u"frequence(Hz)")
        pl.subplots_adjust(hspace=0.4)
        pl.show()
