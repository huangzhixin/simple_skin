#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 02.11.2016
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
from scipy.signal import find_peaks_cwt
import scipy.signal as signal

settings = pd.read_pickle('settings.pkl')

data_stream = np.load(settings['Name_of_run']+'/thresholdCleanData.npy')
print sum(sum(data_stream,0),0)[700]
plt.plot(sum(sum(data_stream,0),0))

plt.show()

