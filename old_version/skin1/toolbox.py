#coding:utf-8

# version 1.0  
# Author: Zhixin Huang
# Date: 07.11.2016
# File Name: toolbox.py
# 
# Method Name : toolbox
# 
# Description:  This toolbox contain some useful function...... 

# Argument : 	  none  (values are read from settings loaded in load settings )
# 			
# Output:         none

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
from scipy.signal import find_peaks_cwt

import itertools
from sklearn.metrics import confusion_matrix

def dofilter(curve):
  window = signal.general_gaussian(51, p=1.5, sig=7)
  filteredCurve = signal.fftconvolve(window, curve)
  filteredCurve = (np.average(curve) / np.average(filteredCurve)) * filteredCurve
  filteredCurve = np.roll(filteredCurve, -25)
  #高斯拟合后居然比原信号长一截什么鬼。。。。
  return filteredCurve[:curve.shape[0]]

#ind = CROSSING(S,level) returns an index vector ind, the signal
#   S crosses zero at ind or at between ind and ind+1
#   given level instead of the zero crossings
def crossing(S,level):
  index = np.where(np.diff(np.sign(S-level)))[0]
  return index

def findPeaks(filteredCurve,expectedValue):
  #print "expectedValue is "+ str(expectedValue)
  expectedValues=np.arange(10,100)
  indexes = find_peaks_cwt(filteredCurve, expectedValues) #第二个参数是你对峰值的期望宽度
  #这里还可以改进可以只找前3000帧和后3000帧的峰值，中间的数据完全可以忽略这样可以提高速度
  #https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
  #this link talk about many ways to find the peak
  #the type of indexes is list, we must change the type to ndarray
  indexes=np.array(indexes)
  #plt.plot(indexes,curve[indexes],'r+')                    #draw all the peak points
  #plt.show()
  return indexes

def plotSegmentedData(data,label,activity_start_frame,activity_end_frame,nameOfTitle='title',nameOfxlabel='x',nameOfylabel='y',isLine=True):
  unique_classes = np.unique(label)
  num_of_classes  = unique_classes.shape[0]
  lines = []
  cmap = plt.get_cmap('jet')
  colors = cmap(np.linspace(0, 1.0, num_of_classes))

  fig = plt.figure()
  fig.hold()
  fig.suptitle(nameOfTitle, fontsize=12)


  for i in range(0,label.shape[0]):
    index = 1
    currentClass = label[i]
    #print i
    for j in range(0,num_of_classes):
      if unique_classes[j]==currentClass:
        index = j
        c = colors[index,:]
        xdata = range(int(activity_start_frame[i]),int(activity_end_frame[i]))
        #plt.scatter(xdata,data[range(activity_start_frame[i],activity_end_frame[i])])
        if isLine == False:
          plt.plot(xdata,data[range(activity_start_frame[i],activity_end_frame[i])],color=c,linewidth=1,marker="o")
        else:
          plt.plot(xdata,data[activity_start_frame[i]:activity_end_frame[i]],color=c,linewidth=1)
    #end for j
  #end for i
  
  plt.xlabel(nameOfxlabel)

  plt.ylabel(nameOfylabel)
  
  #only for test the color map
  for k in range(0,num_of_classes):
    c = colors[k,:]
    line, = plt.plot(range(1,2),color=c,linewidth=2)
    lines.append(line)

  fig.legend(lines,unique_classes,'upper right')

  plt.show()

def plotScatterDat(xData,yData,label,nameOfTitle='title',nameOfxlabel='x',nameOfylabel='y'):
  unique_classes = np.unique(label)
  num_of_classes  = unique_classes.shape[0]
  lines = []
  cmap = plt.get_cmap('jet')
  colors = cmap(np.linspace(0, 1.0, num_of_classes))

  fig = plt.figure()
  fig.hold()
  fig.suptitle(nameOfTitle, fontsize=12)


  for i in range(0,label.shape[0]):
    index = 1
    currentClass = label[i]
    print i
    for j in range(0,num_of_classes):
      if unique_classes[j]==currentClass:
        index = j
        c = colors[index,:]
        plt.scatter(xData[i],yData[i],color=c,marker="o")
    #end for j
  #end for i
  
  plt.xlabel(nameOfxlabel)

  plt.ylabel(nameOfylabel)
  
  #only for test the color map
  for k in range(0,num_of_classes):
    c = colors[k,:]
    line, = plt.plot(range(1,2),color=c,linewidth=2)
    lines.append(line)

  fig.legend(lines,unique_classes,'upper right')

  plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #np.set_printoptions(precision=2)  #控制精度保留小数点后2位
    cm = np.around(cm,2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.around(cm,2)     #控制精度保留小数点后2位

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
