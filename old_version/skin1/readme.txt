------------------------------Required libraries------------------------------------------
python 2.7
ipython

numpy
scipy
matplotlib

pandas

openCV

scikit learn

------------------------------------------------------------------------------------------

--------------------------------------Not-------------------------------------------------
1. there are differences between Windows and Linux when you write a file path, this code have been tested in Linux secussfully, If you want to test in Windows you need change all the "/" to "\"

2. On "readlabel.py" I have removed all the synchronisation between two activities.

3. I have tested some machine learning algorithms, some algorithms will have better      result, if you normalize data before training, such as KNN and SVM.

------------------------------------------------------------------------------------------   

-------------------------------------Code------------------------------------------------
You can execute the code in the following order step by step, or you can write a .sh(linux) or .bat(windows) script.

python loadConfig.py

python readFile.py
python readlabel.py
python thresholdClean.py
python calcFrameFeatures.py

python synchronizCurve.py
python segmentData.py
python plotSegmentedData.py
python validateSementation.py

python calcPositionFeatures.py 
python calcFrameHuFeatures.py
python calcStatisticalFeatures.py
python calcBasicHuFeatures.py
python calcCrossingFeature.py
python calcPeakCount.py
python calcSegmentedFrameFeatures.py
python saveFeaturesTable.py


python loadClassificationConfig.py
python createTrainingSet.py

python classifier.py
