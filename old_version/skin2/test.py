#coding:utf-8

# version 2.0
# Author: Zhixin Huang
# Date: 28.01.2017

from toolbox import *
from loadConfig import *
from readFile import *
from readLabel import *
from thresholdClean import *
from synchronizCurve import *
from segmentData import *
from manualSegmentData import *
from plotSegmentedData import *
from calcFrameFeatures import *
from calcPositionFeatures import *
from calcFrameHuFeatures import *
from calcStatisticalFeatures import *
from calcBasicHuFeatures import *
from calcCrossingFeature import *
from calcPeakCount import *
from calcSegmentedFrameFeatures import *
from saveFeaturesTable import *
from loadClassificationConfig import *
from createTrainingSet import *
from generateVideo import *
from preprocess import *
from validateSementation import *
from calcAreaFeatures import *
from calcFrequenceFeature import *




settingsName = 'settings5'
#loadConfig('config5.csv', settingsName)


settings = pd.read_pickle(settingsName+'.pkl')
#print settings['minTime']
#readFile(settingsName)


#readLabel(settingsName)



#thresholdClean(settingsName)




diynamicThresholdClean(settingsName)

#preprocess(settingsName, 'thresholdCleanData', 'preprocessed_thresholdCleanData')
#preprocess(settingsName, 'thresholdData', 'preprocessed_thresholdData')

#splitData(settingsName, 'preprocessed_thresholdCleanData', 'frontside_CleanData', 'backside_CleanData')
#splitData(settingsName, 'preprocessed_thresholdData', 'frontside_thresholdData', 'backside_thresholdData')



#plotCurve(settingsName,'data_stream')
plotCurve(settingsName,'thresholdCleanData')
#plotCurve(settingsName,'preprocessed_thresholdCleanData')

#plotCurve(settingsName,'preprocessed_thresholdData')
#plotCurve(settingsName,'frontside_thresholdData')
#plotCurve(settingsName,'backside_thresholdData')


#generateVideo(settingsName,'thresholdCleanData',0,settings['minTime'])
#generateSplitVideo(settingsName, 'frontside_CleanData', 'backside_CleanData', 1000, 2500)
#generateSplitVideo(settingsName, 'frontside_thresholdData', 'backside_thresholdData', 0, settings['minTime'])
#generateVideo(settingsName,'preprocessed_datastream',0,settings['minTime'])
#generateVideo(settingsName,'thresholdData',0,settings['minTime'])

''''
frontTrainset = np.load(settings['Name_of_run'] + '/frontTrainset.npy')
classTrainset = np.load(settings['Name_of_run'] + '/classTrainset.npy')
test = frontTrainset[classTrainset=='0']
print test.shape
for image in test:
  plotOneFrame(image)

'''

#calcFrameFeature(settingsName,'frontside_CleanData','frontFrameFeature')
#calcFrameFeature(settingsName,'backside_CleanData','backFrameFeature')
#calcFrameFeature(settingsName,'preprocessed_thresholdCleanData','frameFeatures')
#calcDiffFrameFeature(settingsName)

#manualSegmentData('settings4')


#correctSegment(settingsName, 'preprocessed_thresholdCleanData')

#sortLabel(settingsName)
#get10class(settingsName)
#getMiddleClass(settingsName)
#getHighFrequenceClass(settingsName)


plotSegmentedData(settingsName, 'frameFeatures',"weightFeatureSmooth", 'sorted_activity_start_frame','sorted_activity_end_frame','sorted_Label',isSort = True)
#plotSegmentedData(settingsName, 'frameFeatures',"weightFeatureSmooth", 'small_activity_start_frame','small_activity_end_frame','small_label',isSort = True)
#plotSegmentedData(settingsName, 'frameFeatures',"weightFeatureSmooth", 'activity_start_frame','activity_end_frame','label',isSort = False)

#plotSegmentedData(settingsName, 'preprocessed_thresholdCleanData', 'activity_start_frame','activity_end_frame','label')
#plotSegmentedData(settingsName, 'frontside_thresholdData', 'labelData')
#lotSegmentedData(settingsName, 'backside_thresholdData', 'labelData')

#calcAreaFeature(settingsName, 'frontside_CleanData', 'frontside_bwMasks')
#calcAreaFeature(settingsName, 'backside_CleanData', 'backside_bwMasks')

#validateSementation(settingsName, 'front_Segmented_frame', 'back_Segmented_frame', 'labelData')

#primaryClassFeatures=pd.read_pickle(settings['Name_of_run']+'/primaryClassFeatures.pkl')
#segmentFeatures = pd.read_pickle(settings['Name_of_run']+'/segmentFeatures.pkl')
#label = np.load(settings['Name_of_run'] + '/sorted_primaryLabel.npy')
#print segmentFeatures
#plotScatterDat(range(1,label.shape[0]-1),primaryClassFeatures["meadian_weightFeature_segmented"],label[0:-2])
#plotScatterDat(segmentFeatures["meanWeightFeature"],segmentFeatures["maxWeightFeature"],segmentFeatures['wwclasses'])




#calcFrequenceFeature(settingsName, 'frameFeatures','weightFeatureSmooth', 'frontside_Frequence')





#calcFrameFeature('settings1')
#segmentData('settings1',False,500,63000)
#plotSegmentedData('settings1')

#calcFrameFeature('settings2')
#segmentData('settings2',False,900,63550)
#plotSegmentedData('settings2')



#calcPositionFeatures('settings1')
#calcFrameHuFeature('settings1')
#calcStatisticalFeatures('settings1')
#calcBasicHuFeatures('settings1')
#calcCrossingFeatures('settings1')
#calcPeakCount('settings1')
#calcSegmentedFrameFeatures('settings1')
#saveFeaturesTable('settings1')
#loadClassificationConfig('clsConfig.csv')
#createTrainingSet('clsconfig')