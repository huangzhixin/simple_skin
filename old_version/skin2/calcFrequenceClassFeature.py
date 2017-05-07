# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 01.04.2017
# Latest Date: 01.04.2017
# File Name: calcPrimaryClassFeatures.py
#
# Method Name : calcPrimaryClassFeatures
#
# Description:  This file is used to calculate basic features for the first classification eg. sleeping, starting, ending, recording
#
# Argument : 	  none  (values are read from settings loaded in load settings )
#
# Output:  		  basic features(area, weight, pressure)

# checking dimensions of data stream

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from toolbox import *
from peakdetect import *
import matplotlib.pyplot as plt
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

def calcFrequenceClassFeatures(settingName):
    print '* calculating primary classification features * '

    settings = pd.read_pickle(settingName + '.pkl')

    activity_start_frame = np.load(settings['Name_of_run'] + '/middle_activity_start_frame.npy')
    activity_end_frame = np.load(settings['Name_of_run'] + '/middle_activity_end_frame.npy')

    sorted_label = np.load(settings['Name_of_run'] + '/middle_label.npy')

    frameFeatures = pd.read_pickle(settings['Name_of_run'] + '/frameFeatures.pkl')
    segmentFeatures = pd.read_pickle(settings['Name_of_run'] + '/middle_segmentFeatures.pkl')

    #frontsideCleanData = np.load(settings['Name_of_run'] + '/' + 'frontside_CleanData' + '.npy')
    #backsideCleanData = np.load(settings['Name_of_run'] + '/' + 'backside_CleanData' + '.npy')

    number_of_segments = activity_start_frame.__len__()-2
    #print number_of_segments
    # prelocating

    mean_weightFeature_segmented = np.zeros((number_of_segments))
    change_weightFeature_segmented = np.zeros((number_of_segments))
    max_weightFeature_segmented = np.zeros((number_of_segments))
    min_weightFeature_segmented = np.zeros((number_of_segments))
    median_weightFeature_segmented = np.zeros((number_of_segments))
    mean_pressureFeature_segmented = np.zeros((number_of_segments))
    change_pressureFeature_segmented = np.zeros((number_of_segments))
    max_pressureFeature_segmented = np.zeros((number_of_segments))
    min_pressureFeature_segmented = np.zeros((number_of_segments))
    median_pressureFeature_segmented = np.zeros((number_of_segments))
    peakcount = np.zeros((number_of_segments))
    vallycount = np.zeros((number_of_segments))

    #frontImageFeature = np.zeros((number_of_segments,5))
    #backImageFeature = np.zeros((number_of_segments,5))


    for i in range (0, number_of_segments):
        segment= frameFeatures["weightFeature"][activity_start_frame[i+1]:activity_end_frame[i+1]]
        mean_weightFeature_segmented[i] = np.mean(segment)
        change_weightFeature_segmented[i]= np.max(segment)-np.median(segment)
        max_weightFeature_segmented[i] = np.max(segment)
        min_weightFeature_segmented[i] = np.min(segment)
        median_weightFeature_segmented[i] = np.median(segment)
        segment = frameFeatures["areaFeatureSmooth"][activity_start_frame[i + 1]:activity_end_frame[i + 1]]
        mean_pressureFeature_segmented[i] = np.mean(segment)
        change_pressureFeature_segmented[i] = np.max(segment) - np.median(segment)
        max_pressureFeature_segmented[i] = np.max(segment)
        min_pressureFeature_segmented[i] = np.min(segment)
        median_pressureFeature_segmented[i] = np.median(segment)





        peaks = peakdetect(
            frameFeatures["weightFeatureSmooth"][int(activity_start_frame[i + 1]):int(activity_end_frame[i + 1])],
            lookahead=10)
        plt.plot(frameFeatures["weightFeatureSmooth"][int(activity_start_frame[i + 1]):int(activity_end_frame[i + 1])])
        #print peaks[0].__len__()
        #print peaks[1].__len__()
        #print u''+sorted_label[i]
        peakcount[i]=peaks[0].__len__()
        vallycount[i]=peaks[1].__len__()

       # plt.hold(True)
        #plt.plot(peaks[0][:,],frameFeatures["weightFeatureSmooth"][peaks[0]],'ro')
       # plt.plot(peaks[1], frameFeatures["weightFeatureSmooth"][peaks[1]],'b*')
       # plt.hold(False)
        #plt.show()
        #print peaks[0].__len__()
        #print peaks[1].__len__()

        #change_weightFeature_segmented[i] = frameFeatures["weightFeatureSmooth"][activity_start_frame[i+1]] - frameFeatures["weightFeatureSmooth"][activity_end_frame[i+1]]
    #print max_weightFeature_segmented
    features = {'mean_weightFeature_segmented': mean_weightFeature_segmented,
                #'change_weightFeature_segmented': change_weightFeature_segmented,
                'max_weightFeature_segmented': max_weightFeature_segmented,
                'min_weightFeature_segmented': min_weightFeature_segmented,
                'meadian_weightFeature_segmented': median_weightFeature_segmented,

                'mean_pressureFeature_segmented': mean_pressureFeature_segmented,
                #'change_pressureFeature_segmented': change_pressureFeature_segmented,
                'max_pressureFeature_segmented': max_pressureFeature_segmented,
                'min_pressureFeature_segmented': min_pressureFeature_segmented,
                'meadian_pressureFeature_segmented': median_pressureFeature_segmented,
                'peakcount':peakcount,
                'vallycount': vallycount
                }


    featureTable = pd.DataFrame(features)

    featureTable.to_pickle(settings['Name_of_run'] + '/frequenceClassFeatures.pkl')


def labelFrequenceClass(settingName,nameOfLabel,saveName):
    settings = pd.read_pickle(settingName + '.pkl')

    label = np.load(settings['Name_of_run'] + '/'+nameOfLabel+'.npy')

    highFrequences = ['戳','挠','划','摸','甩腿','抠','扔接','扔不接（乱扔）']
    lowFrequences = ['拍','捶']
    otherFrequences = ['挡','藏','递','长时间拿']
    frequenceLabel = label.copy()
    for i in range(0,frequenceLabel.shape[0]):
        for highFrequence in highFrequences:
            if label[i]==highFrequence:
                frequenceLabel[i]='highFrequence'
                break
        for lowFrequence in lowFrequences:
            if label[i]==lowFrequence:
                frequenceLabel[i]='lowFrequences'
                break
        '''
        for middlelow in middleLows:
            if label[i]==middlelow:
                primaryLabel[i]='middlelow'
                break
        '''
        for otherFrequence in otherFrequences:
            if label[i]==otherFrequence:
                frequenceLabel[i]='otherFrequences'
                break
    #for i in range(0,label.shape[0]):
    #    print u''+label[i]+'  '+u''+primaryLabel[i]
    #for i in range(0, weightLabel.shape[0]):
    #    print u''+weightLabel[i]
    np.save(settings['Name_of_run'] + '/'+saveName,frequenceLabel)

calcFrequenceClassFeatures('settings1')
labelFrequenceClass('settings1','middle_label','frequenceLabel')
calcFrequenceClassFeatures('settings2')
labelFrequenceClass('settings2','middle_label','frequenceLabel')
calcFrequenceClassFeatures('settings3')
labelFrequenceClass('settings3','middle_label','frequenceLabel')
calcFrequenceClassFeatures('settings6')
labelFrequenceClass('settings6','middle_label','frequenceLabel')

#labelPrimaryClass('settings1','sorted_label','sorted_primaryLabel')