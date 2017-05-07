# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 31.10.2016
# Latest Date: 28.01.2017
# Method Name : readFile
# 
# Description:  This method  is used read data from the hardware generated text file
# 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  text files are read and saved in a Two dimesional matrix.

import pandas as pd
import pickle
import os
import numpy as np


# reading text file

# Checking if file exists in given location 
def readFile(nameOfSettings):
    settings = pd.read_pickle(nameOfSettings + '.pkl')

    #fileLoc = settings['File_location'] + settings['File_Name']

    #if os.path.isfile(fileLoc) == False:
    #    print 'fopen ' + fileLoc + ' failed!!'

    print '* reading file  * '
    rawData1 = pd.read_csv(settings['File_location']+'e_rec1.txt', header=None)
    print rawData1.values.shape
    rawData2 = pd.read_csv(settings['File_location'] + 'e_rec2.txt', header=None)
    print rawData2.values.shape











    merge=[rawData1,rawData2]
    rawData = pd.concat(merge)
    print rawData.values.shape


    # the type of rawData.values is array, the data type of rawData.values is int

    time = rawData.values.shape[0]

    data_stream = np.zeros((int(settings['mat_size_x']), int(settings['mat_size_y']), time))

    rawData = np.array(rawData)

    for i in range(0, time):
        data_stream[:, :, i] = rawData[i, :].reshape(int(settings['mat_size_x']), int(settings['mat_size_y']))

    # print data_stream[:,:,18450]
    data_stream = data_stream.astype('int')
    data_stream = data_stream >> 4
    """
    #data_stream = rawData.reshape(int(settings['mat_size_x']) , int(settings['mat_size_y']),time)
    #虽然这样代码比较简洁，但是这样转化的格式是错误的，具体咋改有时间再研究
    """
    # resave the settings
    settings['minTime'] = time
    print time

    settings.to_pickle(nameOfSettings + '.pkl')

    print '* saving raw data * '

    np.save(settings['Name_of_run'] + '/data_stream', data_stream)
