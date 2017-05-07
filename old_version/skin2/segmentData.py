# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 02.11.2016
# Latest Date: 28.01.2017
# File Name: 	segmentData.py
# 
# Method Name : segmentData
# 
# Description:  We are here we are importing the labels for the activities
#               and mark them in our data stream, we also generate start and end frames of  
# 
# Argument :	Name of the csv file with extenstion 
#
# Output:  		  basic features(area, weight, pressure) 

# startFrame  -frame of synchronisation end
# lastFrame - frame of synchronisation begin

import pandas as pd
import numpy as np
import toolbox


def segmentData(settingName, isAuto=True, synStartFrame=857, synEndFrame=61800):
    settings = pd.read_pickle(settingName + '.pkl')
    labelData = pd.read_pickle(settings['Name_of_run'] + '/labelData.pkl')
    data_stream = np.load(settings['Name_of_run'] + '/thresholdCleanData.npy')
    # data_stream = np.load(settings['Name_of_run'] + '/combineData.npy')
    frameFeatures = pd.read_pickle(settings['Name_of_run'] + '/frameFeatures.pkl')

    if isAuto == True:
        crossingPoints = toolbox.crossing(frameFeatures['areaFeatureSmooth'].values, 60);
        synchronisationStartFrame = crossingPoints[1]  # second
        synchronisationEndFrame = crossingPoints[2]  # third
    else:
        synchronisationStartFrame = synStartFrame
        synchronisationEndFrame = synEndFrame

    print "synchronisationStartFrame is " + str(synchronisationStartFrame)
    print "synchronisationEndFrame is " + str(synchronisationEndFrame)
    # to be modified by checking with areaFeature / plotting imagesc(areaFeature)
    # synchronisationStartFrame = 857
    # synchronisationEndFrame = 26563

    # synchronisationStartFrame = settings["synchronisationStartFrame"]
    # synchronisationEndFrame = settings["synchronisationEndFrame"]

    print '* executing segmentData*'
    frame_rate = (synchronisationEndFrame - synchronisationStartFrame - 100) / float(
        (labelData["beginTime"][settings["total_events"] - 1] - labelData["endTime"][0]))

    # frame_rate = 0.0595
    print '* calculated frame rate in Hz*'
    print frame_rate * 1000

    activity_start_frame = np.fix(labelData["beginTime"] * frame_rate)
    activity_end_frame = np.fix(labelData["endTime"] * frame_rate)
    frameOffset = synchronisationStartFrame - activity_end_frame[0]

    activity_start_frame = activity_start_frame + frameOffset
    activity_end_frame = activity_end_frame + frameOffset

    segmented_frame = np.zeros((data_stream.shape[0], data_stream.shape[1], settings["total_events"]))
    activity_start_frame[activity_start_frame > settings["minTime"]] = settings["minTime"]
    activity_end_frame[activity_end_frame > settings["minTime"]] = settings["minTime"]
    activity_start_frame[activity_start_frame < 0] = 1
    activity_end_frame[activity_end_frame < 0] = 1

    for i in range(0, settings["total_events"] - 1):
        segmented_frame[:, :, i] = np.max(data_stream[:, :, int(activity_start_frame[i]):int(activity_end_frame[i])], 2)

    print'* saving segementedData *'
    np.save(settings['Name_of_run'] + '/segmented_frame', segmented_frame)
    np.save(settings['Name_of_run'] + '/activity_start_frame', activity_start_frame)
    np.save(settings['Name_of_run'] + '/activity_end_frame', activity_end_frame)
