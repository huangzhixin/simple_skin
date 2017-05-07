# coding:utf-8

# version 3.0
# Author: Zhixin Huang
# Date: 04.05.2017
# File Name: 	labelingtool.py
#
# Class Name : LabelingTool
#
# Description:  This class is an application for labeling
#

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from matplotlib.figure import Figure
from Tkinter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#import matplotlib
#print matplotlib.get_configdir()
#from matplotlib.font_manager import findfont, FontProperties
#print matplotlib.font_manager.FontProperties().get_family()
#print findfont(FontProperties(family=FontProperties().get_family()))
plt.rcParams['font.sans-serif']=['simhei'] #用来正常显示中文标签



class LabelingTool():
    def __init__(self,settingName):

        self.settings = pd.read_pickle(settingName + '.pkl')
        print self.settings['Name_of_run']
        self.data_stream = np.load(self.settings['Name_of_run'] + '/preprocessed_thresholdCleanData.npy')
        self.weightdata = sum(sum(self.data_stream, 0), 0)

        self.activity_start_frame = [0]
        self.activity_end_frame = [self.data_stream.shape[2]]
        self.labelData = np.load(self.settings['Name_of_run'] + '/label.npy')
        self.unique_classes = np.unique(self.labelData)
        self.num_of_classes = self.unique_classes.shape[0]
        #self.addNewClass()
        self.label = [u'not_segmented']
        self.pointer = 0  # 管理列表的指针

        self._init_app()
        self._init_app1()

    def _init_app(self):
        self.master = Tk()
        self.master.title("Hello World!")
        self.master.geometry('1300x700')

        # -------------------------------------------------------------------------------


        self.scale = 0
        self.windowsSize = 5000
        self.canvas = Canvas(self.master)
        self.canvas.grid(row=3, column=1, columnspan=2)
        self.f = plt.Figure(figsize=(10, 5), dpi=100)

        self.a = self.f.add_subplot(111)

        # self.plotSegmentedDatas(self.a, self.weightdata, self.label, self.activity_start_frame, self.activity_end_frame)

        self.a.plot(sum(sum(self.data_stream, 0), 0))
        self.dataPlot = FigureCanvasTkAgg(self.f, master=self.canvas)
        self.dataPlot.show()
        self.dataPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.toolbar = NavigationToolbar2TkAgg(self.dataPlot, self.canvas)

        self.toolbar.update()
        # 这个地方要注意，如果放在plot之前会显示
        # AttributeError: 'NoneType' object has no attribute 'mpl_connect'
        # 所以要放到之后
        # http://stackoverflow.com/questions/31933393/event-handling-matplotlibs-figure-and-pyplots-figure
        self.f.canvas.mpl_connect('button_press_event', self.on_press)
        self.f.canvas.mpl_connect('button_release_event', self.on_release)
        self.f.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.startSlider = Scale(self.master, from_=0, to=self.data_stream.shape[2] - 1, length=800, orient=HORIZONTAL,
                                 command=self.startSliderUpdate)
        self.startSlider.grid(row=1, column=1)
        self.endSlider = Scale(self.master, from_=0, to=self.data_stream.shape[2] - 1, length=800, orient=HORIZONTAL,
                               command=self.endSliderUpdate)
        self.endSlider.grid(row=2, column=1)

        self.smartStartSlider = Scale(self.master, from_=-10, to=10, length=100, orient=HORIZONTAL,
                                      command=self.smartSliderUpdate)
        self.smartStartSlider.grid(row=1, column=2)

        self.smartEndSlider = Scale(self.master, from_=-10, to=10, length=100, orient=HORIZONTAL,
                                    command=self.smartSliderUpdate)
        self.smartEndSlider.grid(row=2, column=2)

        self.plotSlider = Scale(self.master, from_=0, to=self.data_stream.shape[2] - 1 - self.windowsSize, length=500,
                                orient=HORIZONTAL,
                                command=self.plotSliderUpdate)
        self.plotMaxValue = self.data_stream.shape[2] - 1 - self.windowsSize
        self.plotSlider.grid(row=4, column=1)

        self.windowSlider = Scale(self.master, from_=2000, to=self.data_stream.shape[2] - 1, length=50,
                                  command=self.windowSliderUpdate)
        self.windowSlider.set(5000)
        self.windowSlider.grid(row=1, column=3)

        self.saveButton = Button(self.master, text='save', command=self.saveAllButtonUpdate)
        self.saveButton.grid(row=4, column=2)
        self.resetButton = Button(self.master, text="撤销", command=self.removeButtonUpdate)
        self.resetButton.grid(row=4, column=3)
        '''
        self.frame1 = Frame(self.master)
        self.frame1.grid(row=3, column=4)
        self.actions = IntVar()
        self.actions.set(1)
        for i in range(self.num_of_classes):
            Radiobutton(self.frame1, variable=self.actions, text=self.unique_classes[i], value=i).pack()
        '''
        self.actionListbox = Listbox(self.master, height=30)

        for i in range(0, self.labelData.__len__()):
            self.actionListbox.insert(i, str(i) + '  ' + self.labelData[i])
        self.actionListbox.grid(row=3, column=4)

        self.nowEntry = Entry(self.master)
        self.nowEntry.grid(row=2, column=4)
        self.nowEntry.insert(0, self.labelData[0])

        # -------------------------------------------------------------------------------

    def _init_app1(self):
        self.display = Toplevel(self.master)
        self.display.title("Display!")
        self.canvas1 = Canvas(self.display)
        self.canvas1.grid(row=3, column=1, columnspan=2)
        self.f1 = plt.Figure(figsize=(11, 5), dpi=100)
        a1 = self.f1.add_subplot(111)
        self.plotSegmentedDatas(a1, self.weightdata, self.label, self.activity_start_frame, self.activity_end_frame)

        # self.a.plot(sum(sum(self.data_stream, 0), 0))
        self.dataPlot1 = FigureCanvasTkAgg(self.f1, master=self.canvas1)
        self.dataPlot1.show()
        self.dataPlot1.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.toolbar1 = NavigationToolbar2TkAgg(self.dataPlot1, self.canvas1)

        self.toolbar1.update()

        self.master.mainloop()

    def on_press(self, event):
        print 'press'
        x0 = event.xdata
        # y0 = event.ydata
        self.startSlider.set(x0)

    def on_release(self, event):
        print 'release'
        x1 = event.xdata
        y1 = event.ydata
        self.endSlider.set(x1)
        self.saveButtonUpdate()

        self.actionListbox.see(self.pointer)
        self.actionListbox.selection_set(self.pointer)

        self.pointer = self.pointer + 1
        self.nowEntry.delete(0, END)
        self.nowEntry.insert(0, self.labelData[self.pointer])

    def on_motion(self, event):
        print 'motion'
        x2 = event.xdata
        y2 = event.ydata
        self.endSlider.set(x2)

    def sliderUpdate(self, start, end):
        if end < start:
            self.endSlider.set(start)
        startPointer = np.zeros(self.weightdata.shape[0])
        endPointer = np.zeros(self.weightdata.shape[0])
        pointer = self.plotSlider.get()

        startPointer[start] = max(self.weightdata)
        endPointer[end] = max(self.weightdata)

        self.a.plot(range(pointer, pointer + self.windowsSize), self.weightdata[pointer:pointer + self.windowsSize])
        self.a.hold(True)
        self.a.plot(range(pointer, pointer + self.windowsSize), startPointer[pointer:pointer + self.windowsSize], 'r')
        self.a.plot(range(pointer, pointer + self.windowsSize), endPointer[pointer:pointer + self.windowsSize], 'g')
        self.a.hold(False)
        self.dataPlot.show()

        a1 = self.f1.add_subplot(111)
        a1.plot(startPointer, 'r')
        a1.hold(True)
        a1.plot(endPointer, 'g')
        self.plotSegmentedDatas(a1, self.weightdata, self.label, self.activity_start_frame, self.activity_end_frame)

        # self.a1.hold(False)
        self.dataPlot1.show()

    def startSliderUpdate(self, source):
        start = self.startSlider.get()
        end = self.scale + start
        self.endSlider.set(end)
        self.sliderUpdate(start, end)

        # print start

    def endSliderUpdate(self, source):
        start = self.startSlider.get()
        end = self.endSlider.get()
        self.scale = end - start
        self.sliderUpdate(start, end)

    def smartSliderUpdate(self, source):
        start = self.startSlider.get()
        end = self.endSlider.get()
        smartStart = self.smartStartSlider.get()
        smartEnd = self.smartEndSlider.get()
        newStart = start + smartStart
        newEnd = end + smartEnd
        self.startSlider.set(newStart)
        self.endSlider.set(newEnd)
        self.smartStartSlider.set(0)
        self.smartEndSlider.set(0)

    def plotSliderUpdate(self, source):
        start = self.startSlider.get()
        end = self.endSlider.get()
        startPointer = np.zeros(self.weightdata.shape[0])
        endPointer = np.zeros(self.weightdata.shape[0])
        pointer = self.plotSlider.get()
        if pointer > self.plotMaxValue - 1:
            self.plotSlider.set(self.plotMaxValue - 1)
        else:
            '''
            if start < pointer:
              self.startSlider.set(pointer)
            if end < pointer:
              self.endSlider.set(pointer)
            if start > (pointer + self.windowsSize):
              #print type(pointer),type(self.windowsSize)
              self.startSlider.set(pointer + self.windowsSize)
            if end > (pointer + self.windowsSize):
              self.endSlider.set(pointer + self.windowsSize)
           '''
            startPointer[start] = max(self.weightdata)
            endPointer[end] = max(self.weightdata)
            self.a.plot(range(pointer, pointer + self.windowsSize),
                        self.weightdata[pointer:(pointer + self.windowsSize)])
            self.a.hold(True)
            self.a.plot(range(pointer, pointer + self.windowsSize), startPointer[pointer:pointer + self.windowsSize],
                        'r')
            self.a.plot(range(pointer, pointer + self.windowsSize), endPointer[pointer:pointer + self.windowsSize], 'g')
            self.a.hold(False)
            self.dataPlot.show()

    def windowSliderUpdate(self, source):
        windows = self.windowSlider.get()
        pointer = self.plotSlider.get()
        if (pointer + windows < self.weightdata.shape[0]):
            self.windowsSize = windows
            self.plotSliderUpdate(source)
            self.plotMaxValue = self.weightdata.shape[0] - windows

        else:
            self.windowsSize = windows
            self.plotSlider.set(self.weightdata.shape[0] - self.windowsSize)

    def plotSegmentedDatas(self, handel, data, label, activity_start_frame, activity_end_frame, isLine=True):

        unique_classes = np.unique(label)
        num_of_classes = unique_classes.shape[0]
        lines = []
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1.0, num_of_classes))

        handel.hold(True)
        # self.a.suptitle(nameOfTitle, fontsize=12)

        for i in range(0, label.__len__()):
            index = 1
            currentClass = label[i]
            for j in range(0, num_of_classes):
                if unique_classes[j] == currentClass:
                    index = j
                    c = colors[index, :]
                    xdata = range(int(activity_start_frame[i]), int(activity_end_frame[i]))
                    # plt.scatter(xdata,data[range(activity_start_frame[i],activity_end_frame[i])])

                    if isLine == False:
                        handel.plot(xdata, data[range(activity_start_frame[i], activity_end_frame[i])], color=c,
                                    linewidth=1,
                                    marker="o")
                    else:
                        handel.plot(xdata, data[int(activity_start_frame[i]):int(activity_end_frame[i])], color=c,
                                    linewidth=1)


                        # end for j
        # end for i

        # self.a.xlabel(nameOfxlabel)

        # self.a.ylabel(nameOfylabel)

        # only for test the color map
        for k in range(0, num_of_classes):
            c = colors[k, :]
            line, = handel.plot(range(1, 2), color=c, linewidth=2)
            lines.append(line)

        handel.legend(lines, unique_classes, 'upper right')

        handel.hold(False)

    def saveButtonUpdate(self):
        print 'save'
        start = self.startSlider.get()
        end = self.endSlider.get()
        self.activity_start_frame.append(start)
        self.activity_start_frame[0] = end + 1
        self.activity_end_frame.append(end)
        a1 = self.f1.add_subplot(111)
        a1.plot(np.zeros(self.weightdata.shape[0]))
        a1.hold(False)
        self.label.append(self.labelData[self.pointer])
        # self.label.append(self.unique_classes[self.actions.get()])
        print self.activity_start_frame, self.activity_end_frame, self.label

        self.plotSegmentedDatas(a1, self.weightdata, self.label, self.activity_start_frame,
                                self.activity_end_frame)

        self.dataPlot1.show()

    def removeButtonUpdate(self):
        if (self.activity_start_frame.__len__() > 1):
            print 'remove'
            start = self.activity_start_frame.pop(-1)
            end = self.activity_end_frame.pop(-1)
            self.label.pop(-1)
            self.activity_start_frame[0] = self.activity_end_frame[-1] + 1
            a1 = self.f1.add_subplot(111)
            a1.plot(np.zeros(self.weightdata.shape[0]))
            a1.hold(False)
            # self.a1.plot(np.zeros(self.weightdata))
            self.plotSegmentedDatas(a1, self.weightdata, self.label, self.activity_start_frame,
                                    self.activity_end_frame)
            self.dataPlot1.show()
            print self.activity_start_frame, self.activity_end_frame, self.label

            self.pointer = self.pointer - 1
            self.actionListbox.selection_clear(self.pointer)
            self.nowEntry.delete(0, END)
            self.nowEntry.insert(0, self.labelData[self.pointer])

        if (self.activity_start_frame.__len__() == 1):
            self.activity_start_frame = [0]
            self.activity_end_frame = [self.data_stream.shape[2]]
            a1 = self.f1.add_subplot(111)
            a1.plot(np.zeros(self.weightdata.shape[0]))
            a1.hold(False)
            # self.a1.plot(np.zeros(self.weightdata))
            self.plotSegmentedDatas(a1, self.weightdata, self.label, self.activity_start_frame,
                                    self.activity_end_frame)
            self.dataPlot1.show()
            print self.activity_start_frame, self.activity_end_frame, self.label

            self.pointer = 0
            self.actionListbox.selection_clear(self.pointer)
            self.nowEntry.delete(0, END)
            self.nowEntry.insert(0, self.labelData[self.pointer])

    def saveAllButtonUpdate(self):
        activity_start_frames = np.array(self.activity_start_frame[1:])
        activity_end_frames = np.array(self.activity_end_frame[1:])
        np.save(self.settings['Name_of_run'] + '/activity_start_frame', activity_start_frames)
        np.save(self.settings['Name_of_run'] + '/activity_end_frame', activity_end_frames)

    def addNewClass(self):
        # New Class are sleeping, starting, ending
        newLabel= []
        newLabel.append(self.labelData[0])
        for i in range(1,self.labelData.shape[0]-1):
            newLabel.append('sleeping')
            newLabel.append('starting')
            newLabel.append(self.labelData[i])
            newLabel.append('ending')
        newLabel.append(self.labelData[-1])
        self.labelData = np.array(newLabel)
        np.save(self.settings['Name_of_run'] + '/newLabelData', newLabel)




if __name__ == "__main__":
    LabelingTool('settings5')
