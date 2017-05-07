# coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 01.11.2016
# Latest Date: 29.01.2017
# File Name: 	validateSementation.py
# 
# Method Name : validateSementation
# 
# Description:  Generating images to validate segmentation against given
#               class
# 
# Argument :	none data is read from segmented_frame
#
# Output:  		  one image per activity

import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps
from matplotlib import gridspec

def validateSementation(nameOfSettings, nameOfData1, nameOfData2, nameOfLabel):
    settings = pd.read_pickle(nameOfSettings + '.pkl')


    segmented_frame1 = np.load(settings['Name_of_run'] + '/'+nameOfData1+'.npy')
    segmented_frame2 = np.load(settings['Name_of_run'] + '/' + nameOfData2 + '.npy')

    segmented_Mask1 = np.load(settings['Name_of_run'] + '/frontside_bwMasks.npy')
    segmented_Mask2 = np.load(settings['Name_of_run'] + '/backside_bwMasks.npy')

    labelData = pd.read_pickle(settings['Name_of_run'] + '/'+nameOfLabel+'.pkl')

    fig = plt.figure()
    example_image1 = np.random.uniform(1, np.max(segmented_frame1)/3, size=segmented_frame1[:, :, 0].shape)
    example_image2 = np.random.uniform(1, np.max(segmented_frame2)/3, size=segmented_frame2[:, :, 0].shape)


    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    plt.subplot(gs[0])
    handel1 = plt.matshow(example_image1, fignum=0, cmap=cmaps.viridis)

    plt.subplot(gs[1])
    handel2 = plt.matshow(example_image2, fignum=0, cmap=cmaps.viridis)

    # print segmented_frame[:,:,58]
    for i in range(1, settings["total_events"] - 1):
        # fig 0 and fig 101 is synchronizframe
        handel1.set_data(segmented_frame1[:, :, i]*segmented_Mask1[:, :, i-1])   #segmented_Mask1 不包括syn
        handel2.set_data(segmented_frame2[:, :, i]*segmented_Mask2[:, :, i-1])
        f1 = handel1.get_figure()
        f2 = handel2.get_figure()
        fig.set_figure(f1)
        fig.set_figure(f2)

        #fig = plt.matshow(segmented_frame[:, :, i], fignum=0, cmap=cmaps.viridis)
        filepath = settings['Name_of_run'] + "/" + str(i) + "_" + labelData["class"][i] + ".png"
        print filepath
        plt.savefig(filepath)
        #fig.write_png(filepath)
