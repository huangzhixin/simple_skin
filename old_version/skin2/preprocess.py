# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 02.03.2016
# Latest Date: 10.03.2017
# Method Name : preprocess
#
# Description:  This method  is used to interpolation and smoothing and Split data
#
# Argument : 	  none  (values are read from settings loaded in load settings )
#
# Output:  		  text files are read and saved in a Two dimesional matrix.

import pandas as pd

import numpy as np
import cv2


# reading text file

# Checking if file exists in given location
def preprocess(nameOfSettings, nameOfData='thresholdCleanData', saveName='preprocessed_thresholdCleanData'):

    settings = pd.read_pickle(nameOfSettings + '.pkl')

    data_stream = np.load(settings['Name_of_run'] + '/'+nameOfData+'.npy')

    scale_ratio = settings['scale_ratio']

    new_scale_x = data_stream.shape[0] * scale_ratio

    new_scale_y = data_stream.shape[1] * scale_ratio

    preprocessed_datastream = np.zeros((new_scale_x, new_scale_y, int(settings['minTime'])))
    print preprocessed_datastream.shape
    #print data_stream[:, :, 100].dtype
    #print type(data_stream[:, :, 100])
    #print np.zeros((16,16)).dtype
    #print type(np.zeros((16,16)))
    data_stream1=data_stream.astype(np.float64)         #opencv 只认可数据格式为float64， 原数据读出来int64，resize函数不认
    print data_stream1.dtype                            #一个数组的类型这样data_stream.astype(np.float64)没法自己改自己，return的才是
    #print data_stream1[:, :, 1000]
    for i in range(0, int(settings['minTime'])):
        img = cv2.resize(data_stream1[:, :, i], (new_scale_y, new_scale_x))
        #print img.shape
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        preprocessed_datastream[:, :, i] = blur


    print '* saving preprocessed data * '
    np.save(settings['Name_of_run'] + '/'+saveName, preprocessed_datastream)


def splitData(nameOfSettings, nameOfData, saveName1,saveName2):
    settings = pd.read_pickle(nameOfSettings + '.pkl')

    data_stream = np.load(settings['Name_of_run'] + '/'+nameOfData+'.npy')
    scale_ratio = settings['scale_ratio']
    #print data_stream.shape  #(23,16,n)
    frontside = data_stream[0* scale_ratio:13* scale_ratio, 0* scale_ratio:16* scale_ratio, :]
    backside = data_stream[13* scale_ratio:23* scale_ratio, 0* scale_ratio:16* scale_ratio, :]  # 这里只适用于我的海绵宝宝项目程序

    print backside[:, :, 1000]

    frontside = np.rot90(frontside, 2)
    backside = np.rot90(backside, 3)

    print backside[:, :, 1000]

    print '* saving front and backdata * '
    np.save(settings['Name_of_run'] + '/'+saveName1, frontside)
    np.save(settings['Name_of_run'] + '/'+saveName2, backside)