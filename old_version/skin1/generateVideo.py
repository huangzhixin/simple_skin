#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 01.11.2016
# File Name: generateVideo.py
# 
# Method Name : generateVideo
# 
# Description:  This method is used to generate  a video file from this
#               given data_stream  the video can be used to generate labels.
# 

# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:  		  run_name.mp4 where run_name is specified in
#                 loadConfig.csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
#import cv2
#import cv2.cv as cv
import pandas as pd

#load config and data_stream

settings = pd.read_pickle('settings.pkl')

data_stream = np.load(settings['Name_of_run']+'/thresholdCleanData.npy')

#create handel of videowriter
FFMpegWriter = manimation.writers['avconv']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')

writer = FFMpegWriter(fps=settings["frame_rate"], metadata=metadata)
fig = plt.figure()
handel=plt.matshow(data_stream[:,:,0], fignum=0,cmap=plt.cm.viridis)
#cmap=plt.cm.viridis这个很重要colormap！！，没有它图像颜色很混乱cmap=plt.cm.viridis
#具体你想用哪个，ipython下help(colormaps)便可

with writer.saving(fig, settings['Name_of_run']+"/"+settings['Name_of_run']+".mp4", 100):
    #for i in range(settings["minTime"]):
    for i in range(16000,19000):
        #hier can not use plt.matshow directly, the process will run very slowly
        #so we need use the handel of plt.matshow
        #then become the object of the figur, then with fig.set_figure to set new figure
        handel.set_data(data_stream[:,:,i])
        
        f = handel.get_figure()
        fig.set_figure(f)
        print i
        writer.grab_frame()









"""
#the first way!!!

img1=np.array([4, 1, 1, 14, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 1, 1, 34, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 18, 1, 1, 18, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 27, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 1, 37, 6, 1, 17, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 62, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 65, 13, 35, 13, 1, 9, 8, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 23, 27, 69, 91, 20, 40, 48, 42, 15, 1, 1, 1, 1, 1, 4, 1, 45, 63, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 39, 19, 32, 6, 22, 42, 59, 41, 10, 1, 1, 1, 1, 31, 11, 1, 126])
    
img1=img1.reshape(16,16)
img2=np.ones((16,16))

FFMpegWriter = manimation.writers['avconv']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
fig = plt.figure()
a=plt.matshow(img1, fignum=0)
with writer.saving(fig, "test.mp4", 100):
    for i in range(100):
        
        if i%2==0:
          a.set_data(img2)
        else:
          a.set_data(img1)

        f = a.get_figure()
        fig.set_figure(f)
        print i
        writer.grab_frame()

"""

"""
#the second way!!
    img = img/(float(img.max()-img.min()))
    img=img*256

    img.astype(int)

    img = cv2.resize(img,(160,160))
writer = cv2.VideoWriter('test1.avi', cv.CV_FOURCC(*'XVID'), 25, (160, 160))
for i in range(1000):
   # x = (np.random.randint(126,size=(160,160))+img).astype('uint8')
    x = img.astype('uint8')
    x=cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
    x[:,:,0]=x[:,:,0]*1           #GBR
    x[:,:,1]=x[:,:,1]*1
    x[:,:,2]=x[:,:,2]*0.1
    x.astype(int)
    x.astype('uint8')
    writer.write(x)
"""


