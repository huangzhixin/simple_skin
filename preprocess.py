# coding:utf-8

# version 3.0
# Author: Zhixin Huang
# Date: 02.05.2017
# File Name: 	preprocess.py
#
# Class Name : Preprocess
#
# Description:  this Class is first step to data mining, that include read data, resize, smooth, remove DC
# and calculate basic frame features.
import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
import utilities
import matplotlib.animation as manimation
import colormaps as cmaps

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
plt.rcParams['font.sans-serif']=['simhei'] #用来正常显示中文标签

class Preprocess():

    def __init__(self, setting_name,config__file_name,resolution=12):
        self.setting_name = setting_name
        self.config__file_name = config__file_name
        try:
          self.settings = pd.read_pickle(self.setting_name + '.pkl')
        except:
          self.loadConfig(self.config__file_name, self.setting_name)
          self.settings = pd.read_pickle(self.setting_name + '.pkl')

        self.resolution = resolution
        self.data_stream = None
        self.clean_data_stream = None
        self.DCs = None
        self.preprocessed_data_stream = None
        self.frontside = None
        self.backside = None
        self.frame_features = None
        self.utilities = utilities.Utilities()

    def loadConfig(self,configFileName, settingName):
        # reading file name of the config file
        # configFileName = raw_input("Enter the name of config file,(config.csv) ")

        # default file name case
        # if configFileName == '':
        #   configFileName = 'config.csv'

        # importing pre structured config file
        print 'Reading config file '
        data_table = pd.read_csv(configFileName)

        settings = pd.Series(data_table.value.values, index=data_table.Name.values)

        # change the string to number in settings
        for i in range(0, settings.shape[0]):
            if str.isdigit(settings.values[i]) == True:
                settings.values[i] = int(settings.values[i])

        settings.to_pickle(settingName + '.pkl')

        print '* saving settings *'
        print settings
        # make direction
        # making a directory with name of run

        if os.path.exists(settings['Name_of_run']) == False:
            os.popen('mkdir ' + settings['Name_of_run']).readlines()

    def read_file(self, fileNames=[]):
        if fileNames.__len__() == 0:
            fileLoc = self.settings['File_location'] + self.settings['File_Name']

            if os.path.isfile(fileLoc) == False:
                print 'fopen ' + fileLoc + ' failed!!'
            print '* reading file  * '
            raw_data = pd.read_csv(fileLoc, header=None)

        else:
            merge = []
            for fileName in fileNames:
                raw_data = pd.read_csv(self.settings['File_location'] + fileName, header=None)
                merge.append(raw_data)
            raw_data = pd.concat(merge)
            print raw_data.values.shape

        time = raw_data.values.shape[0]

        self.data_stream = np.zeros((int(self.settings['mat_size_x']), int(self.settings['mat_size_y']), time))

        raw_data = np.array(raw_data)

        # channal last
        for i in range(0, time):
            self.data_stream[:, :, i] = raw_data[i, :].reshape(int(self.settings['mat_size_x']),
                                                               int(self.settings['mat_size_y']))

        # print data_stream[:,:,18450]
        self.data_stream = self.data_stream.astype('int')
        self.data_stream = self.data_stream >> (12 - self.resolution)
        """
        #data_stream = raw_data.reshape(int(settings['mat_size_x']) , int(settings['mat_size_y']),time)
        #虽然这样代码比较简洁，但是这样转化的格式是错误的，具体咋改有时间再研究
        """
        # resave the settings
        self.settings['minTime'] = time
        print time

        self.settings.to_pickle(self.setting_name + '.pkl')

        print '* saving raw data * '

        np.save(self.settings['Name_of_run'] + '/data_stream', self.data_stream)

    def diynamic_threshold_clean(self, step=20, scale=1500):

        if self.data_stream == None:
            self.data_stream = np.load(self.settings['Name_of_run'] + '/data_stream.npy')

        print'* applying diynamic threshold clean*'
        print 'this way maybe will run longer time'
        self.clean_data_stream = self.data_stream.copy()  # cleaned data will be saved in self.clean_data_stream
        self.DCs = self.data_stream.copy()
        #init_DC = self.data_stream[:, :, 600]
        init_DC = np.mean(self.data_stream[:, :, 0:600], axis=2)
        clean_threshold = init_DC
        #step = 20
        #scale = 1500
        # 可以理解为窗口的大小，在窗口的前沿和后沿进行判断，若硬件的刷新率越小，窗口小一点，刷新率大就窗口大一点
        for i in range(step, self.settings['minTime']):
            print i
            if np.max(np.abs(self.data_stream[:, :, i - step] - self.data_stream[:, :, i])) <= 1:
                # 前step贞和后step贞若值相同，证明这段时间数据稳定，很有可能是两个动作之间的空闲，但也不排除一个动作压在上面时间过长
                # print i
                if np.sum(np.abs(self.data_stream[:, :, i] - init_DC)) <= scale:
                    # print i
                    # 为啥这里使用sum， 应为发现使用max<=60不靠谱，应为会出现所有点之差小于60，然而可能都是59，他们的之和就超级大
                    # 所以进入第二个判断，若这个动作和宝宝的初始状态压力值相似，便认定他是新的DC
                    # print np.max(np.abs(data_stream[:, :, i] - init_DC))
                    clean_threshold = self.data_stream[:, :, i]
                    for j in range(i - step, i + 1):  # 除去step之间的DC，包括第i帧
                        self.clean_data_stream[:, :, j] = self.data_stream[:, :, j] - clean_threshold
                        self.DCs[:, :, j] = clean_threshold
                else:
                    self.clean_data_stream[:, :, i] = self.data_stream[:, :, i] - clean_threshold
                    self.DCs[:, :, i] = clean_threshold
                    # 剩下两个else就是判断若目前没有新的DC，就用现有的老DC除去现有的帧，直达有新的DC（也就是clean_threshold）出现
            else:
                self.clean_data_stream[:, :, i] = self.data_stream[:, :, i] - clean_threshold
                self.DCs[:, :, i] = clean_threshold
                # print i

        self.clean_data_stream[self.clean_data_stream < 1] = 1

        print '* saving threshold clean data * '

        np.save(self.settings['Name_of_run'] + '/threshold_clean_data', self.clean_data_stream)
        np.save(self.settings['Name_of_run'] + '/threshold_data', self.DCs)


    def resize_smooth(self, name_of_data, save_name):

        data_stream = np.load(self.settings['Name_of_run'] + '/' + name_of_data + '.npy')

        scale_ratio = self.settings['scale_ratio']

        new_scale_x = data_stream.shape[0] * scale_ratio

        new_scale_y = data_stream.shape[1] * scale_ratio

        self.preprocessed_data_stream = np.zeros((new_scale_x, new_scale_y, int(self.settings['minTime'])))
        print 'the size of preprocessed datastream: ', self.preprocessed_data_stream.shape
        # print data_stream[:, :, 100].dtype
        # print type(data_stream[:, :, 100])
        # print np.zeros((16,16)).dtype
        # print type(np.zeros((16,16)))
        data_stream_cv = data_stream.astype(np.float64)  # opencv 只认可数据格式为float64， 原数据读出来int64，resize函数不认
        # print data_stream_cv.dtype  # 一个数组的类型这样data_stream.astype(np.float64)没法自己改自己，return的才是
        # print self.clean_data_stream[:, :, 1000]
        for i in range(0, int(self.settings['minTime'])):
            img = cv2.resize(data_stream_cv[:, :, i], (new_scale_y, new_scale_x))
            # print img.shape
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            self.preprocessed_data_stream[:, :, i] = blur
        print 'the new shape is: ',self.preprocessed_data_stream.shape
        print '* saving preprocessed data * '
        np.save(self.settings['Name_of_run'] + '/' + save_name, self.preprocessed_data_stream)

    def plot_curve(self, name_of_datastream):
        data_stream = np.load(self.settings['Name_of_run'] + '/' + name_of_datastream + '.npy')
        fig = plt.figure()

        plt.plot(sum(sum(data_stream, 0), 0))
        # print data_stream[:, :, 1000]
        # print data_stream[:, :, 2000]
        plt.show()

    def generate_video(self, nameOfDatastream, startFrame, endFrame):

        # load config and data_stream
        preprocessed_thresholdCleanData = np.load(self.settings['Name_of_run'] + '/data_stream.npy')

        data_stream1 = np.load(self.settings['Name_of_run'] + '/' + nameOfDatastream + '.npy')
        # data_stream2 = np.load(settings['Name_of_run'] + '/' + nameOfDatastream2 + '.npy')
        # create handel of videowriter

        # FFMpegWriter = manimation.writers
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')

        writer = manimation.FFMpegWriter(fps=self.settings["frame_rate"], metadata=metadata)
        fig = plt.figure()
        example_image1 = np.random.uniform(1, 20, size=data_stream1[:, :, 0].shape)
        # example_image2 = np.random.uniform(1, 20, size=data_stream2[:, :, 0].shape)

        plt.subplot2grid((3, 4), (0, 1), colspan=2, rowspan=2)
        handel1 = plt.matshow(example_image1, fignum=0, cmap=cmaps.viridis)
        plt.axis('off')

        # plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=2)
        # handel2 = plt.matshow(example_image2, fignum=0, cmap=cmaps.viridis)
        # plt.axis('off')

        plt.subplot2grid((3, 3), (2, 0), colspan=3)

        plotData = sum(sum(data_stream1, 0), 0)
        handel, = plt.plot(range(0, plotData.shape[0]), plotData)

        # 这个地方要改，handel里面这个图片必须有色彩，若一开始颜色全一样则无法显示出颜色
        # cmap=plt.cm.viridis这个很重要colormap！！，没有它图像颜色很混乱cmap=plt.cm.viridis
        # 具体你想用哪个，ipython下help(colormaps)便可
        # print settings['Name_of_run'] + "/" + settings['Name_of_run'] + ".mp4"

        with writer.saving(fig, self.settings['Name_of_run'] + "/" + self.settings['Name_of_run'] + "1.mp4", 100):
            # for i in range(settings["minTime"]):

            for i in range(startFrame, endFrame):
                # hier can not use plt.matshow directly, the process will run very slowly
                # so we need use the handel of plt.matshow
                # then become the object of the figur, then with fig.set_figure to set new figure
                plt.title('The current frames: ' + str(i))
                pointer = np.zeros(plotData.shape[0])
                pointer[i] = max(plotData)

                handel1.set_data(data_stream1[:, :, i])
                # handel2.set_data(data_stream2[:, :, i])
                handel.set_data(range(0, plotData.shape[0]), pointer + plotData)
                f1 = handel1.get_figure()
                # f2 = handel2.get_figure()
                f = handel.get_figure()
                fig.set_figure(f1)
                # fig.set_figure(f2)
                fig.set_figure(f)

                print i
                writer.grab_frame()


    def calc_frame_feature(self, name_of_datastream, save_name='frame_features'):
        print '* calculating frame features * '

        data_stream = np.load(self.settings['Name_of_run'] + '/' + name_of_datastream + '.npy')
        # data_stream = np.load(settings['Name_of_run'] + '/combineData.npy')
        number_of_frames = data_stream.shape[2]

        rows = data_stream.shape[0]
        cols = data_stream.shape[1]

        # prelocating
        self.area_feature = np.zeros((number_of_frames))
        self.pressure_feature = np.zeros((number_of_frames))
        self.weight_feature = np.zeros((number_of_frames))
        self.centre_weight_X_feature = np.zeros((number_of_frames))
        self.centre_weight_Y_feature = np.zeros((number_of_frames))

        threshold = np.median(data_stream, axis=2)
        # 这里的threshold可能和matlab算得不太一样，导致后面的bwMask算得可能也不一样，导致这里的所有feature算得都和matlab不一样
        # octave 无法用mean2，后面在实验用matlab试试

        # calculating mask frames having more value than threshold
        print '* starting calculate frame features * may be cost lang time'
        for i in range(0, number_of_frames):
            #print i
            frame = data_stream[:, :, i]
            bwMask = np.zeros(frame.shape)
            bwMask = frame > threshold

            # 当a和b为array时， a * b 计算了a和b的数量积（对应Matlab的 a .* b ），
            # dot(a, b) 计算了a和b的矢量积（对应Matlab的 a * b ）
            # in python sum(mat) == in matlab sum(sum(mat))

            weight = np.sum(frame * bwMask)
            area = np.sum(bwMask)
            if area > 10:
                pressure = np.sum(bwMask * frame) / float(area)
            else:
                pressure = 0

            # 第一次听说这么算重心的方法，牛！
            X, Y = np.meshgrid(range(0, frame.shape[1]), range(0, frame.shape[0]))

            if area > 10:
                centerOfWeightX = np.sum(frame * X) / float(np.sum(frame))
                centerOfWeightY = np.sum(frame * Y) / float(np.sum(frame))
            else:
                centerOfWeightX = frame.shape[0] / 2
                centerOfWeightY = frame.shape[1] / 2

            self.area_feature[i] = area
            self.weight_feature[i] = weight
            self.pressure_feature[i] = pressure
            self.centre_weight_X_feature[i] = centerOfWeightX
            self.centre_weight_Y_feature[i] = centerOfWeightY
        # end for
        self.smooth_pressure_feature = self.utilities.dofilter(self.pressure_feature)
        self.smooth_area_feature = self.utilities.dofilter(self.area_feature)
        self.smooth_weight_feature = self.utilities.dofilter(self.weight_feature)

        # smooth_pressure_feature[1:60] = 0
        # smooth_area_feature[1:60] = 0
        # smooth_weight_feature[1:60] = 0

        """

        https://docs.scipy.org/doc/scipy-0.18.1/reference/signal.html
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html

        """

        print '* saving frame features * '
        # format : from array n*1 [[,,,]]to array n [,,,,]

        self.frame_features = {'area_feature': self.area_feature,
                               'weight_feature': self.weight_feature,
                               'pressure_feature': self.pressure_feature,
                               'centre_weight_X_feature': self.centre_weight_X_feature,
                               'centre_weight_Y_feature': self.centre_weight_Y_feature,
                               'smooth_pressure_feature': self.smooth_pressure_feature,
                               'smooth_area_feature': self.smooth_area_feature,
                               'smooth_weight_feature': self.smooth_weight_feature
                               }

        feature_table = pd.DataFrame(self.frame_features)
        # feature_table.to_pickle(settings['Name_of_run']+'/frameFeatures.pkl')
        feature_table.to_pickle(self.settings['Name_of_run'] + '/' + save_name + '.pkl')


    def plot_segmented_data(self, name_of_frame_feature, name_of_feature, name_of_start, name_of_end, name_of_label,isSort=False):

        # labelData = pd.read_pickle(settings['Name_of_run']+'/'+nameOfLabel+'.pkl')
        frameFeatures = pd.read_pickle(self.settings['Name_of_run'] + '/' + name_of_frame_feature + '.pkl')
        data = frameFeatures[name_of_feature]
        # data_stream = np.load(settings['Name_of_run'] + '/combineData.npy')
        activity_start_frame = np.load(self.settings['Name_of_run'] + '/' + name_of_start + '.npy')
        activity_end_frame = np.load(self.settings['Name_of_run'] + '/' + name_of_end + '.npy')

        # data = np.sum(data_stream,axis=(0,1))

        # label = labelData["class"]
        label = np.load(self.settings['Name_of_run'] + '/' + name_of_label + '.npy')
        self.utilities.plotSegmentedDatas(data, label, activity_start_frame, activity_end_frame, isSort=isSort)


if __name__ == "__main__":
    preprocess = Preprocess('setting5','config5.csv',8)
    #preprocess.read_file(['e_rec1.txt','e_rec2.txt'])
    #preprocess.diynamic_threshold_clean()
    #preprocess.plot_curve('threshold_clean_data')
    #preprocess.resize_smooth('threshold_clean_data','resize_smooth_data')
    #preprocess.split_data('threshold_clean_data', 'frontside_clean_data', 'backside_clean_data')
    # preprocess.plot_curve('frontside_clean_data')
    #preprocess.calc_frame_feature('resize_smooth_data')
    #preprocess.plot_segmented_data('frame_features', 'smooth_weight_feature', 'activity_start_frame','activity_end_frame', 'label', isSort=False)
