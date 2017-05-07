# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 31.10.2016
# Latest Date: 28.01.2017
# File Name: thresholdClean.py
# 
# Method Name : thresholdClean
# 
# Description:  This method is used to apply a threhsold to reduce noice
#               and replace values below threshold with 1 i.e(the lowest value in the mat)
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  n X n data read from  text file

# load config and data_stream

import pandas as pd
import pickle
import os
import numpy as np


def thresholdClean(settingName):
    settings = pd.read_pickle(settingName + '.pkl')

    data_stream = np.load(settings['Name_of_run'] + '/data_stream.npy')

    print'* applying threshold clean*'
    print'* threshold value *'
    print settings["clean_threshold"]
    print data_stream.shape
    meanValue = np.mean(data_stream, axis=2)
    print meanValue
    clean_threshold = 0
    data_stream1 = data_stream.copy()
    if settings['clean_threshold'] == 'mean':
        print'* clean the data with mean*'
        clean_threshold = data_stream[:, :, 100]
        for i in range(0, settings['minTime']):
            data_stream1[:, :, i] = data_stream[:, :, i] - clean_threshold
        data_stream1[data_stream1 < 1] = 1

    else:
        clean_threshold = settings['clean_threshold']
        # clean_threshold = 20
        data_stream[data_stream < clean_threshold] = 1

    print '* saving threshold clean data * '

    np.save(settings['Name_of_run'] + '/thresholdCleanData', data_stream1)


def diynamicThresholdClean(settingName):
    settings = pd.read_pickle(settingName + '.pkl')
    data_stream = np.load(settings['Name_of_run'] + '/data_stream.npy')
    print'* applying diynamic threshold clean*'
    print 'this way maybe will run longer time'
    data_stream1 = data_stream.copy()  # cleaned data will be saved in data_stream1
    DCs = data_stream.copy()
    init_DC = data_stream[:, :, 600]
    clean_threshold = init_DC
    step = 20
    scale = 1500
    #可以理解为窗口的大小，在窗口的前沿和后沿进行判断，若硬件的刷新率越小，窗口小一点，刷新率大就窗口大一点
    for i in range(step, settings['minTime']):
        print i
        if np.max(np.abs(data_stream[:, :, i - step] - data_stream[:, :, i])) <= 1:
            #前step贞和后step贞若值相同，证明这段时间数据稳定，很有可能是两个动作之间的空闲，但也不排除一个动作压在上面时间过长
           # print i
            if np.sum(np.abs(data_stream[:, :, i] - init_DC)) <= scale:
                #print i
                #为啥这里使用sum， 应为发现使用max<=60不靠谱，应为会出现所有点之差小于60，然而可能都是59，他们的之和就超级大
                #所以进入第二个判断，若这个动作和宝宝的初始状态压力值相似，便认定他是新的DC
                # print np.max(np.abs(data_stream[:, :, i] - init_DC))
                clean_threshold = data_stream[:, :, i]
                for j in range(i - step, i+1):                #除去step之间的DC，包括第i帧
                    data_stream1[:, :, j] = data_stream[:, :, j] - clean_threshold
                    DCs[:, :, j] =  clean_threshold
            else:
                data_stream1[:, :, i] = data_stream[:, :, i] - clean_threshold
                DCs[:, :, i] = clean_threshold
                #剩下两个else就是判断若目前没有新的DC，就用现有的老DC除去现有的帧，直达有新的DC（也就是clean_threshold）出现
        else:
            data_stream1[:, :, i] = data_stream[:, :, i] - clean_threshold
            DCs[:, :, i] = clean_threshold
            # print i

    data_stream1[data_stream1 < 1] = 1

    print '* saving threshold clean data * '

    np.save(settings['Name_of_run'] + '/thresholdCleanData', data_stream1)
    np.save(settings['Name_of_run'] + '/thresholdData', DCs)
