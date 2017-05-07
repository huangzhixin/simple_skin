# coding:utf-8

# version 3.0
# Author: Zhixin Huang
# Date: 04.05.2017
# File Name: 	recordingtool.py
#
# Class Name : RecordingTool
#
# Description:  This class is an application for recording
#

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from matplotlib.figure import Figure
from Tkinter import *
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import random
import time
from pylab import imshow,imread
import matplotlib.animation as animation
import cv2
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



class RecordingTool():
    def __init__(self,settingName ):
        self.actionsList = []
        self.startTime = []
        self.endTime = []
        self.repeatTime = 10
        self.imgLocal = './image/'
        self.loadConfig()
        self.pointer = 0  # 指示动作做到哪一个了
        localTime = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))

        self.settings = pd.read_pickle(settingName + '.pkl')
        self.Logname = self.settings['Name_of_run'] + '/' + localTime + "_log.txt"
        log = codecs.open(self.Logname,'a', "utf-8")

        self._init_MainView()
        self.displayView()
        self.master.mainloop()

    def loadConfig(self):
        file = open('lab.txt')
        while 1:

            line = file.readline()
            if not line or line == '\n':
                break
            line = line[:-1]  # remove last /n

            for i in range(0, self.repeatTime):
                self.actionsList.append(line)

        random.shuffle(self.actionsList)
        self.actionsList.insert(0, 'synchronize')  # 第一个syn不参与random
        self.actionsList.append('synchronize')
        print self.actionsList.__len__()

    def _init_MainView(self):

        self.master = Tk()
        self.master.title("Hello World!")
        self.master.geometry('600x600')

        self.nowLabel = Label(self.master, text='当前动作')
        self.nowLabel.grid(row=1, column=1)

        self.nextLabel = Label(self.master, text='下一个动作')
        self.nextLabel.grid(row=2, column=1)

        self.nextLabel.setvar('text', '下一个动作')

        self.nowEntry = Entry(self.master)
        self.nowEntry.grid(row=1, column=2)
        self.nowEntry.insert(0, "a default value")

        self.nextEntry = Entry(self.master)
        self.nextEntry.grid(row=2, column=2)
        self.nextEntry.insert(0, "a default value")

        self.frameForButtons = Frame(self.master)
        self.frameForButtons.grid(row=3, column=1, )
        self.startButton = Button(self.frameForButtons, text='开始动作', command=self.startButtonUpdate)
        self.startButton.grid(row=1, column=2)
        self.sucessButton = Button(self.frameForButtons, text="动作成功", command=self.sucessButtonUpdate)
        self.sucessButton.grid(row=2, column=2, pady=30, padx=10)
        self.failButton = Button(self.frameForButtons, text="动作失败", command=self.failButtonUpdate)
        self.failButton.grid(row=2, column=3)
        self.saveButton = Button(self.frameForButtons, text="保存", command=self.saveButtonUpdate)
        self.saveButton.grid(row=3, column=3)


        self.actionListbox = Listbox(self.master, height=30)

        for i in range(0, self.actionsList.__len__()):
            self.actionListbox.insert(i, str(i) + '  ' + self.actionsList[i])
        self.actionListbox.grid(row=3, column=3)
        '''
        self.actionListbox.see(10)
        self.actionListbox.selection_set(10)
        '''

    def displayView(self):



        self.display = Toplevel(self.master)
        self.display.title("Display!")
        self.canvas = Canvas(self.display)
        self.canvas.pack(side=TOP, fill=BOTH, expand=1)
        im = plt.imread(r"./image/Data_recording_Tool.png")
        self.fig = plt.figure()
        self.a =self.fig.add_subplot(111)
        self.a.imshow(im)  # later use a.set_data(new_data)
        self.a.hold(False)

        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # a tk.DrawingArea
        self.dataPlot = FigureCanvasTkAgg(self.fig, master=self.canvas)

        self.dataPlot.show()
        self.dataPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


    def startButtonUpdate(self):
        if self.pointer < self.actionsList.__len__()-1:
            self.nowEntry.delete(0, END)
            self.nextEntry.delete(0, END)
            self.nowEntry.insert(0, self.actionsList[self.pointer])
            self.nextEntry.insert(0, self.actionsList[self.pointer + 1])
            self.actionListbox.see(self.pointer)
            self.actionListbox.selection_set(self.pointer)

        elif self.pointer == self.actionsList.__len__()-1:
            self.nowEntry.insert(0, self.actionsList[self.pointer])
            self.nextEntry.insert(0, '你已经完成数据记录')
            self.actionListbox.see(self.pointer)
            self.actionListbox.selection_set(self.pointer)
            self.startButton.configure(state=DISABLED)

        self.actionListbox.see(self.pointer)
        self.actionListbox.selection_set(self.pointer)

        self.imageShow()

        localTime = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        log = open(self.Logname, "a")
        log.write(localTime +'\t'+ self.actionListbox.get(self.pointer)+'开始记录'+'\n')
        log.close()
        print localTime +'\t'+ self.actionListbox.get(self.pointer)+'开始记录'
        self.startButton.configure(state=DISABLED)
        self.sucessButton.configure(state=NORMAL)
        self.failButton.configure(state=NORMAL)


    def sucessButtonUpdate(self):


        localTime = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        log = open(self.Logname, "a")
        log.write(localTime + '\t' + self.actionListbox.get(self.pointer) + '记录成功' + '\n')
        log.close()
        #print localTime + '\t' + self.actionListbox.get(self.pointer) + '记录成功'

        self.pointer = self.pointer + 1
        im = plt.imread(r'./image/'+self.actionsList[self.pointer]+ '.png')
        self.a.hold(False)
        self.a.imshow(im)  # later use a.set_data(new_data)
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])


        self.dataPlot.show()
        self.startButton.configure(state=NORMAL)
        self.sucessButton.configure(state=DISABLED)
        self.failButton.configure(state=DISABLED)
        self.startButtonUpdate()


    def failButtonUpdate(self):
        nameOfFailActionWithoutIndex = self.actionsList[self.pointer]
        self.actionsList[self.pointer] = 'fail'
        self.actionsList.insert(self.actionsList.__len__()-1,nameOfFailActionWithoutIndex)

        nameOfFailActionWithIndex = self.actionListbox.get(self.pointer)
        self.actionListbox.delete(self.pointer)
        self.actionListbox.insert(self.pointer,'fail')
        self.actionListbox.insert(self.actionListbox.size()-1, nameOfFailActionWithIndex)
        self.startButton.configure(state=NORMAL)

        localTime = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        log = open(self.Logname, "a")
        log.write(localTime + '\t' + self.actionListbox.get(self.pointer) + '记录失败，添加到末尾' + '\n')
        log.close()
        print localTime + '\t' + self.actionListbox.get(self.pointer) + '记录失败，添加到末尾'
        print self.actionsList

        self.pointer = self.pointer + 1
        self.nowEntry.delete(0, END)
        self.nextEntry.delete(0, END)
        self.nowEntry.insert(0, self.actionsList[self.pointer])
        self.nextEntry.insert(0, self.actionsList[self.pointer + 1])
        self.actionListbox.see(self.pointer)
        self.actionListbox.selection_set(self.pointer)

        self.startButton.configure(state=NORMAL)
        self.sucessButton.configure(state=DISABLED)
        self.failButton.configure(state=DISABLED)
        self.startButtonUpdate()


    def imageShow(self):
        ims = []
        #time.sleep(3)
        '''
        for i in range(1, 5):
            filename = r'./image/' + str(i) + '.png'
            I = cv2.imread(filename)
            I = cv2.resize(I, (300, 180))
            #I = imread(filename)
            #I.resize(I.shape)

            im = self.a.imshow(I, animated=True)
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ims.append([im])

        ani = animation.ArtistAnimation(self.fig, ims, interval=1000, blit=True,
                                       repeat_delay=10000,repeat=False)

        '''
        im = plt.imread(r'./image/'+self.actionsList[self.pointer]+'.png')
        self.a.hold(False)
        self.a.imshow(im)  # later use a.set_data(new_data)
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        self.dataPlot.show()

    def saveButtonUpdate(self):
        label = np.array(self.actionsList)
        np.save(self.settings['Name_of_run'] + '/label', label)




if __name__ == "__main__":
    RecordingTool('settings5')
