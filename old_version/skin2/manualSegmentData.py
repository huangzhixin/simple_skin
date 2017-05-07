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


def manualSegmentData(settingName):
    settings = pd.read_pickle(settingName + '.pkl')

    data_stream = np.load(settings['Name_of_run'] + '/thresholdCleanData.npy')
    frontside_CleanData = np.load(settings['Name_of_run'] + '/frontside_CleanData.npy')
    backside_CleanData = np.load(settings['Name_of_run'] + '/backside_CleanData.npy')



    activity_start_frame =  np.array([0,901,1400,1901,2401, 3150,3900,4800,5450,6200,6700,7200,7760,8200,8800,9600,10150,10600, 11200, 12000,12600,13300, 14200, 14700])
    activity_end_frame = np.array([600,1300,1800,2300,2900, 3500,4300,5200,5800,6500,7000,7400,7800,8450,9200,9800,10400,10850, 11400, 12500,12900,13600, 14500, 15100])
    #print activity_start_frame.shape
    segmented_frame = np.zeros((data_stream.shape[0], data_stream.shape[1], settings["total_events"]))
    front_Segmented_frame = np.zeros((frontside_CleanData.shape[0], frontside_CleanData.shape[1], settings["total_events"]))
    back_Segmented_frame = np.zeros((backside_CleanData.shape[0], backside_CleanData.shape[1], settings["total_events"]))

    for i in range(0, settings["total_events"] - 1):
        segmented_frame[:, :, i] = np.max(data_stream[:, :, int(activity_start_frame[i]):int(activity_end_frame[i])], 2)
        front_Segmented_frame[:, :, i] = np.max(frontside_CleanData[:, :, int(activity_start_frame[i]):int(activity_end_frame[i])], 2)
        back_Segmented_frame[:, :, i] = np.max(backside_CleanData[:, :, int(activity_start_frame[i]):int(activity_end_frame[i])], 2)
    print'* saving segementedData *'
    np.save(settings['Name_of_run'] + '/segmented_frame', segmented_frame)
    np.save(settings['Name_of_run'] + '/front_Segmented_frame', front_Segmented_frame)
    np.save(settings['Name_of_run'] + '/back_Segmented_frame', back_Segmented_frame)
    np.save(settings['Name_of_run'] + '/activity_start_frame', activity_start_frame)
    np.save(settings['Name_of_run'] + '/activity_end_frame', activity_end_frame)
