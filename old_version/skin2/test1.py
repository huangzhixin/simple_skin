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
from sklearn import preprocessing

def normalization(trainData):
    trainData_normalized = preprocessing.normalize(trainData, norm='l2',axis = 0)
    #axis used to normalize the data along. If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
    return trainData_normalized

settingsName = 'settings1'
#loadConfig('config1.csv', settingsName)


settings = pd.read_pickle(settingsName+'.pkl')

frontFrameFeatures = pd.read_pickle(settings['Name_of_run'] + '/' + 'frontFrameFeature' + '.pkl')
backFrameFeatures = pd.read_pickle(settings['Name_of_run'] + '/' + 'backFrameFeature' + '.pkl')
activity_start_frame = np.load(settings['Name_of_run'] + '/sorted_activity_start_frame.npy')
activity_end_frame = np.load(settings['Name_of_run'] + '/sorted_activity_end_frame.npy')
label = np.load( settings['Name_of_run'] + '/sorted_label.npy')
#print np.max(activity_end_frame-activity_start_frame)
#print np.min(activity_end_frame-activity_start_frame)
#print np.mean(activity_end_frame-activity_start_frame)
#print np.median(activity_end_frame-activity_start_frame)
#print label.shape


#data=frontFrameFeatures['weightFeature']
data=frontFrameFeatures.values
#data=frontFrameFeatures.values
#print data.shape
#print frontFrameFeatures.head
data=normalization(data)
#print data
#imageSize = int(np.mean(activity_end_frame-activity_start_frame))*2
imageSize = 300
print imageSize

fearureImage = np.zeros((label.shape[0],data.shape[1],imageSize))
newfearureImage=[]

#print data.shape,fearureImage.shape
for i in range(1,label.shape[0]-1):    #去除前两个syn
    actionSize = activity_end_frame[i]-activity_start_frame[i]
    centre = actionSize/2

    if actionSize<=imageSize:
       for j in range(0,fearureImage.shape[1]):
         #print data[activity_start_frame[i]:activity_end_frame[i]].shape
         #print fearureImage[i, 0, imageSize / 2 - centre:imageSize / 2 + centre].shape
         fearureImage[i,j,imageSize/2-centre:imageSize/2-centre+actionSize]= data[activity_start_frame[i]:activity_end_frame[i],j]
    if actionSize>imageSize:
       for j in range(0,fearureImage.shape[1]):
         #print data[activity_start_frame[i]:activity_end_frame[i]].shape
         #print fearureImage[i, 0, imageSize / 2 - centre:imageSize / 2 + centre].shape
         fearureImage[i,j,:]= data[activity_start_frame[i]:activity_end_frame[i],j][actionSize/2-imageSize/2:actionSize/2-imageSize/2+imageSize]
    img = cv2.resize(fearureImage[i,:,:],(imageSize/5, imageSize/5))
    newfearureImage.append(img)
    #print img.shape
    print u''+str(i)+label[i]
    plotOneFrame(img)

newfearureImage= np.array(newfearureImage)
newfearureImage=newfearureImage.reshape(newfearureImage.shape[0],(imageSize/5)*(imageSize/5))
print newfearureImage.shape
print settings['Name_of_run']
np.save(settings['Name_of_run'] + '/frontNewfearureImage', newfearureImage)