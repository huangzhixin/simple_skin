# coding:utf-8

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
from matplotlib import gridspec
# import cv2
# import cv2.cv as cv
import pandas as pd
import colormaps as cmaps


def generateVideo(settingName, nameOfDatastream, startFrame, endFrame):
    settings = pd.read_pickle(settingName + '.pkl')
    # load config and data_stream


    data_stream = np.load(settings['Name_of_run'] + '/' + nameOfDatastream + '.npy')

    # create handel of videowriter

    # FFMpegWriter = manimation.writers
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')

    writer = manimation.FFMpegWriter(fps=settings["frame_rate"], metadata=metadata)
    fig = plt.figure()
    example_image = np.random.uniform(1, 20, size=data_stream1[:, :, 0].shape)

    handel = plt.matshow(example_image, fignum=0, cmap=cmaps.viridis)
    # 这个地方要改，handel里面这个图片必须有色彩，若一开始颜色全一样则无法显示出颜色
    # cmap=plt.cm.viridis这个很重要colormap！！，没有它图像颜色很混乱cmap=plt.cm.viridis
    # 具体你想用哪个，ipython下help(colormaps)便可
    # print settings['Name_of_run'] + "/" + settings['Name_of_run'] + ".mp4"

    with writer.saving(fig, settings['Name_of_run'] + "/" + settings['Name_of_run'] + "1.mp4", 100):
        # for i in range(settings["minTime"]):
        for i in range(startFrame, endFrame):
            # hier can not use plt.matshow directly, the process will run very slowly
            # so we need use the handel of plt.matshow
            # then become the object of the figur, then with fig.set_figure to set new figure
            handel.set_data(data_stream[:, :, i])

            f = handel.get_figure()
            fig.set_figure(f)
            print i
            writer.grab_frame()


def generateSplitVideo(settingName, nameOfDatastream1, nameOfDatastream2, startFrame, endFrame):
    settings = pd.read_pickle(settingName + '.pkl')
    # load config and data_stream
    preprocessed_thresholdCleanData = np.load(settings['Name_of_run'] + '/preprocessed_thresholdCleanData.npy')

    data_stream1 = np.load(settings['Name_of_run'] + '/' + nameOfDatastream1 + '.npy')
    data_stream2 = np.load(settings['Name_of_run'] + '/' + nameOfDatastream2 + '.npy')
    # create handel of videowriter

    # FFMpegWriter = manimation.writers
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')

    writer = manimation.FFMpegWriter(fps=settings["frame_rate"], metadata=metadata)
    fig = plt.figure()
    example_image1 = np.random.uniform(1, 20, size=data_stream1[:, :, 0].shape)
    example_image2 = np.random.uniform(1, 20, size=data_stream2[:, :, 0].shape)

    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    handel1 = plt.matshow(example_image1, fignum=0, cmap=cmaps.viridis)
    plt.axis('off')

    plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=2)
    handel2 = plt.matshow(example_image2, fignum=0, cmap=cmaps.viridis)
    plt.axis('off')

    plt.subplot2grid((3, 3), (2, 0), colspan=3)

    plotData = sum(sum(preprocessed_thresholdCleanData, 0), 0)
    handel, = plt.plot(range(0, plotData.shape[0]), plotData)



    # 这个地方要改，handel里面这个图片必须有色彩，若一开始颜色全一样则无法显示出颜色
    # cmap=plt.cm.viridis这个很重要colormap！！，没有它图像颜色很混乱cmap=plt.cm.viridis
    # 具体你想用哪个，ipython下help(colormaps)便可
    # print settings['Name_of_run'] + "/" + settings['Name_of_run'] + ".mp4"

    with writer.saving(fig, settings['Name_of_run'] + "/" + settings['Name_of_run'] + "1.mp4", 100):
        # for i in range(settings["minTime"]):

        for i in range(startFrame, endFrame):
            # hier can not use plt.matshow directly, the process will run very slowly
            # so we need use the handel of plt.matshow
            # then become the object of the figur, then with fig.set_figure to set new figure
            plt.title('The current frames: '+str(i))
            pointer = np.zeros(plotData.shape[0])
            pointer[i] = max(plotData)

            handel1.set_data(data_stream1[:, :, i])
            handel2.set_data(data_stream2[:, :, i])
            handel.set_data(range(0, plotData.shape[0]), pointer+plotData)
            f1 = handel1.get_figure()
            f2 = handel2.get_figure()
            f = handel.get_figure()
            fig.set_figure(f1)
            fig.set_figure(f2)
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
