# coding:utf-8

# version 3.0
# Author: Zhixin Huang
# Date: 03.05.2017
# File Name: 	calcfeatures.py
#
# Class Name : CalcFeatures
#
# Description:  In this class you can calculate a lot of features for data mining.
# you need to extend this class to add new features when you create own applications

import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
import utilities
from peakdetect import *


class CalcFeatures():

    def __init__(self, setting_name, data_stream_name, save_name ='segment_features.pkl', start_frame_name='/activity_start_frame.npy',end_frame_name='/activity_end_frame.npy', label_name='/label.npy'):
        self.setting_name = setting_name 
        self.settings = pd.read_pickle(self.setting_name + '.pkl')
        self.save_name = save_name
        self.activity_start_frame = np.load(self.settings['Name_of_run'] + start_frame_name)
        self.activity_end_frame = np.load(self.settings['Name_of_run'] + end_frame_name)
        self.label = np.load(self.settings['Name_of_run'] + label_name)

        #remove syn
        self.activity_start_frame = self.activity_start_frame[1:-1]
        self.activity_end_frame = self.activity_end_frame[1:-1]
        self.label = self.label[1:-1]

        self.data_stream = np.load(self.settings['Name_of_run'] + '/' + data_stream_name + '.npy')
        #print self.data_stream.shape
        #print self.settings['scale_ratio']
        self.frame_features = pd.read_pickle(self.settings['Name_of_run'] + '/frame_features.pkl')
        self.segmented_data = None
        self.num_of_segment = None
        
        
    def get_segment_frame(self):
        
        self.segmented_data = []
        for i in range(0, self.label.shape[0]):
            #print self.data_stream[:,:,self.activity_start_frame[i]: self.activity_end_frame[i]].shape
            self.segmented_data.append(self.data_stream[:,:,self.activity_start_frame[i]: self.activity_end_frame[i]])

        self.num_of_segment = self.segmented_data.__len__()
        #print self.num_of_segment
        
    def calc_segment_weight_feature(self):
        self.segment_mean_weight_feature = np.zeros((self.num_of_segment))
        self.segment_change_weight_feature = np.zeros((self.num_of_segment))
        self.segment_max_weight_feature = np.zeros((self.num_of_segment))
        self.segment_min_weight_feature = np.zeros((self.num_of_segment))
        self.segment_median_weight_feature = np.zeros((self.num_of_segment))
        self.segment_mean_area_feature = np.zeros((self.num_of_segment))
        self.segment_change_area_feature = np.zeros((self.num_of_segment))
        self.segment_max_area_feature = np.zeros((self.num_of_segment))
        self.segment_min_area_feature = np.zeros((self.num_of_segment))
        self.segment_median_area_feature = np.zeros((self.num_of_segment))

        for i in range(0, self.num_of_segment):
            segment = self.frame_features["smooth_weight_feature"][self.activity_start_frame[i]:self.activity_end_frame[i]]
            self.segment_mean_weight_feature[i] = np.mean(segment)
            self.segment_change_weight_feature[i] = np.max(segment) - np.median(segment)
            self.segment_max_weight_feature[i] = np.max(segment)
            self.segment_min_weight_feature[i] = np.min(segment)
            self.segment_median_weight_feature[i] = np.median(segment)
            segment = self.frame_features["smooth_area_feature"][self.activity_start_frame[i]:self.activity_end_frame[i]]
            self.segment_mean_area_feature[i] = np.mean(segment)
            self.segment_change_area_feature[i] = np.max(segment) - np.median(segment)
            self.segment_max_area_feature[i] = np.max(segment)
            self.segment_min_area_feature[i] = np.min(segment)
            self.segment_median_area_feature[i] = np.median(segment)
     
    def calc_segment_hu_features(self):
        self.hu_features = np.zeros((self.num_of_segment, 7))
        for i in range(0, self.num_of_segment):
            segment = self.segmented_data[i]
            merge_segment = np.max(segment, axis=2)
            merge_segment = merge_segment.astype('float64')
            #print merge_segment.shape
            #print merge_segment.dtype
            self.hu_features[i] = cv2.HuMoments(cv2.moments(merge_segment)).flatten()  # flatten: from array[[]] to array[]
            

    def calc_segment_centre_weight_feature(self):

        self.x_centre_weight_feature = np.zeros((self.num_of_segment))
        self.y_centre_weight_feature = np.zeros((self.num_of_segment))
        
        for i in range(0,self.num_of_segment):
            segment = self.segmented_data[i]
            merge_segment = np.max(segment,axis=2)
            #print segment.shape

            # 第一次听说这么算重心的方法，牛！
            X, Y = np.meshgrid(range(0, merge_segment.shape[1]), range(0, merge_segment.shape[0]))
            self.x_centre_weight_feature[i] = np.sum(merge_segment * X) / float(np.sum(merge_segment))
            self.y_centre_weight_feature[i] = np.sum(merge_segment * Y) / float(np.sum(merge_segment))


    def calc_peak_feature(self):
        self.num_peak = np.zeros((self.num_of_segment))
        self.num_vally = np.zeros((self.num_of_segment))
        for i in range(0, self.num_of_segment):
            peaks = peakdetect(self.frame_features["smooth_weight_feature"][int(self.activity_start_frame[i]):int(self.activity_end_frame[i])],lookahead=10)
            self.num_peak[i] = peaks[0].__len__()
            self.num_vally[i] = peaks[1].__len__()


    def save_segment_feature(self):
        feature_table = {
            'segment_mean_weight_feature' : self.segment_mean_weight_feature,
            'segment_change_weight_feature':self.segment_change_weight_feature,
            'segment_max_weight_feature':self.segment_max_weight_feature,
            'segment_min_weight_feature':self.segment_min_weight_feature,
            'segment_median_weight_feature':self.segment_median_weight_feature,
            'segment_mean_area_feature':self.segment_mean_area_feature,
            'segment_change_area_feature':self.segment_change_area_feature,
            'segment_max_area_feature':self.segment_max_area_feature,
            'segment_min_area_feature':self.segment_min_area_feature,
            'segment_median_area_feature':self.segment_median_area_feature,
            'x_centre_weight_feature': self.x_centre_weight_feature,
            'y_centre_weight_feature': self.y_centre_weight_feature,
            'num_peak':self.num_peak,
            'num_vally': self.num_vally,
            'hu_features1':self.hu_features[:,0],
            'hu_features2':self.hu_features[:,1],
            'hu_features3':self.hu_features[:,2],
            'hu_features4':self.hu_features[:,3],
            'hu_features5':self.hu_features[:,4],
            'hu_features6':self.hu_features[:,5],
            'hu_features7':self.hu_features[:,6],
            "classes": self.label
             }

        self.feature_table = pd.DataFrame(feature_table)
        self.feature_table.to_pickle(self.settings['Name_of_run'] + '/'+ self.save_name + '.pkl')
        print 'save in ' + self.settings['Name_of_run'] + '/'+self.save_name

if __name__ == "__main__":
    front_segment_feature = CalcFeatures('setting5','frontside_clean_data','front_segment_feature')
    front_segment_feature.get_segment_frame()
    front_segment_feature.calc_segment_weight_feature()
    front_segment_feature.calc_segment_hu_features()
    front_segment_feature.calc_segment_centre_weight_feature()
    front_segment_feature.calc_peak_feature()
    front_segment_feature.save_segment_feature()


