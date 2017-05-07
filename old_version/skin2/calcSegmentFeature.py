#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 03.04.2017
# Latest Date: 03.04.2017
# File Name: calcSegmentFeatures.py
#
# Method Name : calcSegmentFeatures
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
import random
import cv2
import toolbox

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class segmentFeature():
    def __init__(self,nameOfSettings,nameOfData1,nameOfData2):
        self.settings = pd.read_pickle( nameOfSettings + '.pkl')
        self.frontsideCleanData = np.load(self.settings['Name_of_run'] + '/' + nameOfData1 + '.npy')
        self.backsideCleanData = np.load(self.settings['Name_of_run'] + '/' + nameOfData2 + '.npy')

        self.time = self.frontsideCleanData.shape[2]
        self.sizeOfFrontsideCleanData = self.frontsideCleanData.shape[0] * self.frontsideCleanData.shape[1]
        self.sizeOfBacksideCleanData = self.backsideCleanData.shape[0] * self.backsideCleanData.shape[1]
        self.serial_data_stream = np.zeros((self.time, self.sizeOfFrontsideCleanData + self.sizeOfBacksideCleanData + 1))
        '''
        self.activity_start_frame = np.load(self.settings['Name_of_run'] + '/small_activity_start_frame.npy')
        self.activity_end_frame = np.load(self.settings['Name_of_run'] + '/small_activity_end_frame.npy')
        self.label = np.load(self.settings['Name_of_run'] + '/small_label.npy')
        '''
        '''
        self.activity_start_frame = np.load(self.settings['Name_of_run'] + '/middle_activity_start_frame.npy')
        self.activity_end_frame = np.load(self.settings['Name_of_run'] + '/middle_activity_end_frame.npy')
        self.label = np.load( self.settings['Name_of_run'] + '/middle_label.npy')
        '''


        self.activity_start_frame = np.load(self.settings['Name_of_run'] + '/activity_start_frame.npy')
        self.activity_end_frame = np.load(self.settings['Name_of_run'] + '/activity_end_frame.npy')
        self.label = np.load(self.settings['Name_of_run'] + '/label.npy')

        '''
        self.activity_start_frame = np.load(self.settings['Name_of_run'] + '/highFrequence_activity_start_frame.npy')
        self.activity_end_frame = np.load(self.settings['Name_of_run'] + '/highFrequence_activity_end_frame.npy')
        self.label = np.load(self.settings['Name_of_run'] + '/highFrequence_label.npy')
        '''
        # 前后两个syn不要
        self.label = self.label[1:-1]
        self.activity_start_frame = self.activity_start_frame[1:-1]
        self.activity_end_frame = self.activity_end_frame[1:-1]
        self.unique_classes = np.unique(self.label)

        for i in range(0, self.unique_classes.shape[0]):
            print str(i), u'' + self.unique_classes[i]

        # 串行化  #以前的3维数据相当于横着切黄瓜， 现在改过来了，所以第一个参数是行,现在要弄成竖着的面包片
        for i in range(0, self.time):
            self.serial_data_stream[i, 0:self.sizeOfFrontsideCleanData] = self.frontsideCleanData[:, :, i].reshape(
                self.sizeOfFrontsideCleanData)
            self.serial_data_stream[i, self.sizeOfFrontsideCleanData:-1] = self.backsideCleanData[:, :, i].reshape(
                self.sizeOfBacksideCleanData)

        # 加标签 label虽然去除了第一个和最后一个syn，但是在activity_start_frame里[0]依然是syn
        for i in range(0, self.label.shape[0]):
            self.serial_data_stream[self.activity_start_frame[i]:self.activity_end_frame[i] + 1, -1] = \
            np.argwhere(self.unique_classes == self.label[i])[0][0]

    def getSegmentedFrame(self, windowssize, radioOfTwoWindows):
        segmentedData = []
        for i in range(0, self.label.shape[0]):
            sizeOfWindows = (self.activity_end_frame[i] - self.activity_start_frame[i]) / windowssize
            #print sizeOfWindows
            if sizeOfWindows > 0 :
                step_len = sizeOfWindows - int(radioOfTwoWindows * sizeOfWindows)
            else:
                sizeOfWindows=1
                step_len=1

            print i,sizeOfWindows,step_len,self.activity_end_frame[i] - self.activity_start_frame[i]
            # step_len = sizeOfWindows
            for j in range(self.activity_start_frame[i], self.activity_end_frame[i] - sizeOfWindows, step_len):
                segmentedData.append(self.serial_data_stream[j:j + sizeOfWindows, :])


        #print segmentedData[5]
        # random 打乱
        random.shuffle(segmentedData)

        self.segmentedData = np.array(segmentedData)

        self.sizeOfSegment = self.segmentedData.shape[0]
        #print segmentedData[5][:, -1]



        #这个segmentedData的结构为[599][49,5889],第一个括号是把所有动作分段的个数， [49]指的是这一段数据有49帧，
        # [5889]其中前5888是像素点，前sizeOfFrontsideCleanData个是正面像素，后sizeOfBacksideCleanData是背面像素
        #最后一位数是类名
        #print segmentedData.shape
        #print segmentedData[5].shape
        #print segmentedData[5]


        classTrainset = []
        for i in range(0, self.segmentedData.shape[0]):
            # classTrainset.append(u''+unique_classes[int(randomClasses[i])-1])
            classTrainset.append(str(int(np.max(segmentedData[i][:, -1]))))
            #每一段数据最后的类标签都是一样的，这里用max取一个就行
        self.classTrainset = np.array(classTrainset)
        #print self.classTrainset.shape


    def getSegmentedFrame1(self):
        #self.classTrainset = self.label
        self.segmentedData = self.serial_data_stream
        segmentedData = []
        for i in range(0, self.label.shape[0]):
            print self.segmentedData[self.activity_start_frame[i]: self.activity_end_frame[i]].shape
            segmentedData.append(self.segmentedData[self.activity_start_frame[i]:self.activity_end_frame[i]])

        self.segmentedData = np.array(segmentedData)
        self.sizeOfSegment = self.segmentedData.shape[0]
        classTrainset = []
        for i in range(0, self.segmentedData.shape[0]):
            # classTrainset.append(u''+unique_classes[int(randomClasses[i])-1])
            classTrainset.append(str(int(np.max(segmentedData[i][:, -1]))))
            # 每一段数据最后的类标签都是一样的，这里用max取一个就行
        self.classTrainset = np.array(classTrainset)
        print self.classTrainset


    def calcAreaFeatures(self):
        self.frontAreaFeature = np.zeros((self.sizeOfSegment))
        self.frontBwMasks = np.zeros((self.sizeOfSegment,self.frontsideCleanData.shape[0], self.frontsideCleanData.shape[1]))
        self.backAreaFeature = np.zeros((self.sizeOfSegment))
        self.backBwMasks = np.zeros((self.sizeOfSegment,self.backsideCleanData.shape[0], self.backsideCleanData.shape[1]))
        for i in range(0,self.sizeOfSegment):
            segment = self.segmentedData[i][:,:-1]

            serialFrontsideCleanData = segment[:, 0:self.sizeOfFrontsideCleanData]
            serialBacksideCleanData = segment[:, self.sizeOfFrontsideCleanData:]

            #print serialFrontsideCleanData.shape
            #print serialBacksideCleanData.shape

            FrontsideCleanData = serialFrontsideCleanData.reshape(
                (serialFrontsideCleanData.shape[0], self.frontsideCleanData.shape[0], self.frontsideCleanData.shape[1]))
            BacksideCleanData = serialBacksideCleanData.reshape(
                (serialBacksideCleanData.shape[0], self.backsideCleanData.shape[0], self.backsideCleanData.shape[1]))

            frontMax = np.max(FrontsideCleanData, axis=0)
            backMax = np.max(BacksideCleanData, axis=0)
            frontMedian = np.median(FrontsideCleanData, axis=0)
            backMedian = np.median(BacksideCleanData, axis=0)
            self.frontBwMasks[i] = frontMax > frontMedian
            self.frontAreaFeature[i] = np.sum(self.frontBwMasks[i])
            self.backBwMasks[i] = backMax > backMedian
            self.backAreaFeature[i] = np.sum(self.backBwMasks[i])

    def calcWeightFeature(self):
        self.maxWeightFeature = np.zeros((self.sizeOfSegment))
        self.minWeightFeature = np.zeros((self.sizeOfSegment))
        self.meanWeightFeature = np.zeros((self.sizeOfSegment))
        self.medianWeightFeature = np.zeros((self.sizeOfSegment))
        for i in range(0,self.sizeOfSegment):
            segment = self.segmentedData[i][:,:-1]

            self.maxWeightFeature[i] = np.max(segment)
            self.minWeightFeature[i] = np.min(segment)
            self.meanWeightFeature[i] = np.mean(segment)
            self.medianWeightFeature[i] = np.median(segment)


    def centreOfWeightFeature(self):

        self.centreOfFrontXWeightFeature = np.zeros((self.sizeOfSegment))
        self.centreOfFrontYWeightFeature = np.zeros((self.sizeOfSegment))
        self.centreOfBackXWeightFeature = np.zeros((self.sizeOfSegment))
        self.centreOfBackYWeightFeature = np.zeros((self.sizeOfSegment))
        self.changeOfFrontXWeightFeature = np.zeros((self.sizeOfSegment))
        for i in range(0,self.sizeOfSegment):
            segment = self.segmentedData[i][:,:-1]
            #print segment.shape
            segment = np.max(segment,axis=0)
            #print segment.shape
            serialFrontsideCleanData = segment[0:self.sizeOfFrontsideCleanData]
            serialBacksideCleanData = segment[self.sizeOfFrontsideCleanData:]

            FrontsideCleanData = serialFrontsideCleanData.reshape(
                (self.frontsideCleanData.shape[0], self.frontsideCleanData.shape[1]))
            BacksideCleanData = serialBacksideCleanData.reshape(
                (self.backsideCleanData.shape[0], self.backsideCleanData.shape[1]))

            # 第一次听说这么算重心的方法，牛！
            frontX, frontY = np.meshgrid(range(0, FrontsideCleanData.shape[1]), range(0, FrontsideCleanData.shape[0]))
            self.centreOfFrontXWeightFeature[i] = np.sum(FrontsideCleanData * frontX) / float(np.sum(FrontsideCleanData))
            self.centreOfFrontYWeightFeature[i] = np.sum(FrontsideCleanData * frontY) / float(np.sum(FrontsideCleanData))

            # 第一次听说这么算重心的方法，牛！
            backX, backY = np.meshgrid(range(0, BacksideCleanData.shape[1]), range(0, BacksideCleanData.shape[0]))
            self.centreOfBackXWeightFeature[i] = np.sum(BacksideCleanData * backX) / float(np.sum(BacksideCleanData))
            self.centreOfBackYWeightFeature[i] = np.sum(BacksideCleanData * backY) / float(np.sum(BacksideCleanData))


    def changeOfWeightFeature(self):
            frontFrameFeatures = pd.read_pickle(self.settings['Name_of_run'] + '/' + 'frontFrameFeature' + '.pkl')
            backFrameFeatures = pd.read_pickle(self.settings['Name_of_run'] + '/' + 'backFrameFeature' + '.pkl')
            self.changeOfFrontXWeightFeature = np.zeros((self.sizeOfSegment))
            self.changeOfFrontYWeightFeature = np.zeros((self.sizeOfSegment))
            for i in range(0, self.sizeOfSegment):
                self.changeOfFrontXWeightFeature[i] = np.max(frontFrameFeatures['centreOfWeightXFeature'][
                                                   self.activity_start_frame[i]:self.activity_end_frame[i]])-np.min(frontFrameFeatures['centreOfWeightXFeature'][
                                                   self.activity_start_frame[i]:self.activity_end_frame[i]])
                self.changeOfFrontYWeightFeature[i] = np.max(frontFrameFeatures['centreOfWeightYFeature'][
                                                          self.activity_start_frame[i]:self.activity_end_frame[
                                                              i]]) - np.min(
                    frontFrameFeatures['centreOfWeightYFeature'][
                    self.activity_start_frame[i]:self.activity_end_frame[i]])


    def calcHuFeatures(self):
        self.frontHuFeatures = np.zeros((self.sizeOfSegment, 7))
        self.backHuFeatures = np.zeros((self.sizeOfSegment, 7))
        for i in range(0, self.sizeOfSegment):
            segment = self.segmentedData[i][:, :-1]
            serialFrontsideCleanData = segment[:, 0:self.sizeOfFrontsideCleanData]
            serialBacksideCleanData = segment[:, self.sizeOfFrontsideCleanData:]

            FrontsideCleanData = serialFrontsideCleanData.reshape(
                (serialFrontsideCleanData.shape[0], self.frontsideCleanData.shape[0], self.frontsideCleanData.shape[1]))
            BacksideCleanData = serialBacksideCleanData.reshape(
                (serialBacksideCleanData.shape[0], self.backsideCleanData.shape[0], self.backsideCleanData.shape[1]))

            FrontSegment = np.max(FrontsideCleanData, axis=0)
            BackSegment = np.max(BacksideCleanData, axis=0)

            print FrontSegment.shape
            print FrontSegment.dtype

            self.frontHuFeatures[i] = cv2.HuMoments(cv2.moments(FrontSegment)).flatten()  # flatten: from array[[]] to array[]
            self.backHuFeatures[i] = cv2.HuMoments(cv2.moments(BackSegment)).flatten()  # flatten: from array[[]] to array[]

    def calcHuFeaturesForMask(self):
        self.frontMaskHuFeatures = np.zeros((self.sizeOfSegment, 7))
        self.backMaskHuFeatures = np.zeros((self.sizeOfSegment, 7))
        for i in range(0, self.sizeOfSegment):
            self.frontMaskHuFeatures[i] = cv2.HuMoments(cv2.moments(self.frontBwMasks[i])).flatten()  # flatten: from array[[]] to array[]
            self.backMaskHuFeatures[i] = cv2.HuMoments(cv2.moments(self.backBwMasks[i])).flatten()  # flatten: from array[[]] to array[]


    def calcCrossingFeature(self):
        self.frontCrossingWeight = np.zeros((self.sizeOfSegment))
        #self.frontCrossingArea = np.zeros((self.sizeOfSegment))
        self.backCrossingWeight = np.zeros((self.sizeOfSegment))
        #self.backCrossingArea = np.zeros((self.sizeOfSegment))
        for i in range(0,self.sizeOfSegment):
            segment = self.segmentedData[i][:,:-1]
            segment = np.max(segment,axis=0)
            serialFrontsideCleanData = segment[0:self.sizeOfFrontsideCleanData]
            serialBacksideCleanData = segment[self.sizeOfFrontsideCleanData:]

            frontMeanWt = np.median(serialFrontsideCleanData)
            self.frontCrossingWeight[i] = np.size(toolbox.crossing(serialFrontsideCleanData,frontMeanWt))

            backMeanWt = np.median(serialBacksideCleanData)
            self.backCrossingWeight[i] = np.size(toolbox.crossing(serialBacksideCleanData, backMeanWt))


    def saveSegmentFeatures(self):
        print'* saving Statistical Features *'
        print self.frontAreaFeature.shape
        #print self.frontBwMasks.shape
        print self.backAreaFeature.shape
        #print self.backBwMasks.shape
        print self.centreOfFrontXWeightFeature.shape
        print self.centreOfFrontYWeightFeature.shape
        print self.centreOfBackXWeightFeature.shape
        print self.centreOfBackYWeightFeature.shape
        #print self.weightFeature.shape
        print self.frontHuFeatures[:,0].shape
        print self.backHuFeatures[:,0].shape
        print self.frontCrossingWeight.shape
        print self.backCrossingWeight.shape
        print self.classTrainset.shape





        featureTable = {"frontAreaFeature": self.frontAreaFeature,
                        #"frontBwMasks": self.frontBwMasks,
                        "backAreaFeature": self.backAreaFeature,
                        #"backBwMasks": self.backBwMasks,
                        "centreOfFrontXWeightFeature": self.centreOfFrontXWeightFeature,
                        "centreOfFrontYWeightFeature": self.centreOfFrontYWeightFeature,
                        "centreOfBackXWeightFeature": self.centreOfBackXWeightFeature,
                        "centreOfBackYWeightFeature": self.centreOfBackYWeightFeature,
                        "changeOfFrontXWeightFeature": self.changeOfFrontXWeightFeature,
                        "changeOfFrontYWeightFeature": self.changeOfFrontYWeightFeature,
                        "maxWeightFeature": self.maxWeightFeature,
                        "minnWeightFeature": self.minWeightFeature,
                        "meanWeightFeature": self.meanWeightFeature,
                        "medianWeightFeature": self.medianWeightFeature,
                        "frontHuFeatures0": self.frontHuFeatures[:,0],
                        "frontHuFeatures1": self.frontHuFeatures[:,1],
                        "frontHuFeatures2": self.frontHuFeatures[:,2],
                        "frontHuFeatures3": self.frontHuFeatures[:,3],
                        "frontHuFeatures4": self.frontHuFeatures[:,4],
                        "frontHuFeatures5": self.frontHuFeatures[:,5],
                        "frontHuFeatures6": self.frontHuFeatures[:,6],
                        "backHuFeatures0": self.backHuFeatures[:,0],
                        "backHuFeatures1": self.backHuFeatures[:,1],
                        "backHuFeatures2": self.backHuFeatures[:,2],
                        "backHuFeatures3": self.backHuFeatures[:,3],
                        "backHuFeatures4": self.backHuFeatures[:,4],
                        "backHuFeatures5": self.backHuFeatures[:,5],
                        "backHuFeatures6": self.backHuFeatures[:,6],
                        "frontCrossingWeight": self.frontCrossingWeight,
                        "backCrossingWeight": self.backCrossingWeight,
                        "wwclasses": self.classTrainset
                        }

        featureTable = pd.DataFrame(featureTable)
        '''
        featureTable.to_pickle(self.settings['Name_of_run'] + '/middle_segmentFeatures.pkl')
        print 'save in' + self.settings['Name_of_run'] + '/middle_segmentFeatures.pkl'
        '''
        '''
        featureTable.to_pickle(self.settings['Name_of_run'] + '/small_segmentFeatures.pkl')
        print 'save in' + self.settings['Name_of_run'] + '/small_segmentFeatures.pkl'

        '''
        featureTable.to_pickle(self.settings['Name_of_run'] + '/segmentFeatures.pkl')
        print 'save in' + self.settings['Name_of_run'] + '/segmentFeatures.pkl'

        '''
        featureTable.to_pickle(self.settings['Name_of_run'] + '/highFrequence_segmentFeatures.pkl')
        print 'save in' + self.settings['Name_of_run'] + '/highFrequence_segmentFeatures.pkl'
        '''







if __name__ == "__main__":
    segemntFeatre = segmentFeature('settings5','frontside_CleanData', 'backside_CleanData')
    #segemntFeatre.getSegmentedFrame(5,0.5)
    segemntFeatre.getSegmentedFrame1()
    segemntFeatre.changeOfWeightFeature()
    segemntFeatre.calcAreaFeatures()
    segemntFeatre.calcWeightFeature()
    segemntFeatre.centreOfWeightFeature()
    segemntFeatre.calcHuFeatures()
    segemntFeatre.calcCrossingFeature()
    segemntFeatre.saveSegmentFeatures()



