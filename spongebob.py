# coding:utf-8

# version 3.0
# Author: Zhixin Huang
# Date: 07.05.2017
# File Name: 	spongebob.py
#
# Class Name : SpongeBob
#
# Description:

import preprocess
import calcfeatures
import classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import colormaps as cmaps


class SpongeBob(preprocess.Preprocess):
    def __init__(self, setting_name, config__file_name, resolution, save_name='segment_features'):
        preprocess.Preprocess.__init__(self, setting_name, config__file_name, resolution)
        self.save_name = save_name
        self.setting_name = setting_name
        self.Classifier = classifier.Classifier(setting_name='trainset', config_name='clsConfig')

    def split_data(self, name_of_data, save_name1, save_name2):
        print 'this function is just used in smart toy project'
        data_stream = np.load(self.settings['Name_of_run'] + '/' + name_of_data + '.npy')
        scale_ratio = self.settings['scale_ratio']
        # print data_stream.shape  #(23,16,n)
        self.frontside = data_stream[0 * scale_ratio:13 * scale_ratio, 0 * scale_ratio:16 * scale_ratio, :]
        self.backside = data_stream[13 * scale_ratio:23 * scale_ratio, 0 * scale_ratio:16 * scale_ratio, :]
        # 这里只适用于我的海绵宝宝项目程序

        # print backside[:, :, 1000]

        self.frontside = np.rot90(self.frontside, 2)
        self.backside = np.rot90(self.backside, 3)
        print self.frontside.shape
        print self.backside.shape

        # print backside[:, :, 1000]

        print '* saving front and backdata * '
        np.save(self.settings['Name_of_run'] + '/' + save_name1, self.frontside)
        np.save(self.settings['Name_of_run'] + '/' + save_name2, self.backside)

    def generate_split_video(self, nameOfDatastream1, nameOfDatastream2, startFrame, endFrame):
        # load config and data_stream

        data_stream1 = np.load(self.settings['Name_of_run'] + '/' + nameOfDatastream1 + '.npy')
        data_stream2 = np.load(self.settings['Name_of_run'] + '/' + nameOfDatastream2 + '.npy')
        # create handel of videowriter

        # FFMpegWriter = manimation.writers
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')

        writer = manimation.FFMpegWriter(fps=self.settings["frame_rate"], metadata=metadata)
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

        plotData = sum(sum(data_stream1, 0), 0) + sum(sum(data_stream2, 0), 0)
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
                handel2.set_data(data_stream2[:, :, i])
                handel.set_data(range(0, plotData.shape[0]), pointer + plotData)
                f1 = handel1.get_figure()
                f2 = handel2.get_figure()
                f = handel.get_figure()
                fig.set_figure(f1)
                fig.set_figure(f2)
                fig.set_figure(f)

                print i
                writer.grab_frame()

    def calc_feature(self):
        print 'start calculating features'
        self.front_segment_feature = calcfeatures.CalcFeatures(self.setting_name, 'frontside_clean_data',
                                                               'front_segment_feature')
        self.back_segment_feature = calcfeatures.CalcFeatures(self.setting_name, 'backside_clean_data',
                                                              'back_segment_feature')

        self.front_segment_feature.get_segment_frame()
        self.front_segment_feature.calc_segment_weight_feature()
        self.front_segment_feature.calc_segment_hu_features()
        self.front_segment_feature.calc_segment_centre_weight_feature()
        self.front_segment_feature.calc_peak_feature()
        self.front_segment_feature.save_segment_feature()
        self.back_segment_feature.get_segment_frame()
        self.back_segment_feature.calc_segment_weight_feature()
        self.back_segment_feature.calc_segment_hu_features()
        self.back_segment_feature.calc_segment_centre_weight_feature()
        self.back_segment_feature.calc_peak_feature()
        self.back_segment_feature.save_segment_feature()
        # self.front_segment_feature.feature_table.pop('zzclasses')  #删除最后一列类
        print self.front_segment_feature.feature_table.shape
        print self.back_segment_feature.feature_table.shape
        self.feature_table = pd.merge(self.front_segment_feature.feature_table, self.back_segment_feature.feature_table,
                                      on='classes', how='inner', left_index=True, right_index=True,
                                      suffixes=('_front', '_back'))
        # 两个feature table 里都有class列，以class列作为键把这两个dataframe以内相关相连，其他列名相同的用front和back加上小尾巴作为区分
        # http://blog.csdn.net/zutsoft/article/details/51498026
        print self.feature_table.shape
        print self.feature_table
        self.feature_table.to_pickle(self.settings['Name_of_run'] + '/' + self.save_name + '.pkl')
        print 'save in ' + self.settings['Name_of_run'] + '/' + self.save_name
        return self.feature_table


if __name__ == "__main__":
    spongebob = SpongeBob('setting5', 'config5.csv', 8)
    # spongebob.split_data('resize_smooth_data','frontside_clean_data', 'backside_clean_data')
    spongebob.calc_feature()
    spongebob.Classifier.auto_validation()
