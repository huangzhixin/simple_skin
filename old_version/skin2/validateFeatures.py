#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 09.11.2016
#
# File Name: validateFeatures.py
# 
# Method Name : validateFeatures
# 
# Description:  This function was used to plot the one of features, we can validate Features with eyes 
# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:         none

import numpy as np
import pandas as pd
import toolbox

settings = pd.read_pickle('settings.pkl')

activity_start_frame = range(0,100)
activity_end_frame = range(1,101)

featuresTable = pd.read_pickle(settings['Name_of_run']+'/featuresTable.pkl')
labelData = pd.read_pickle(settings['Name_of_run']+'/labelData.pkl')

xdata = featuresTable["crossingPressure"].values
ydata = featuresTable["areaFeature"].values
label = labelData["class"][1:101].values
#toolbox.plotSegmentedData(data,label,activity_start_frame,activity_end_frame,isLine=False)
toolbox.plotScatterDat(xdata,ydata,label)
