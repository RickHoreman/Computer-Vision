import scipy.io as sio
import numpy as np
import skimage
from skimage.viewer import ImageViewer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump

print("Opening training data", end='\r')
with open('Data/trainImages.npy', 'rb') as f:
    trainImages = np.load(f)

with open('Data/trainLabels.npy', 'rb') as f:
    trainLabels = np.load(f)

flatTrainImages = np.reshape(trainImages, (len(trainImages), (len(trainImages[0])*len(trainImages[0][0]))))

desiredClasses = []
for i in range(1, 63): #Put 63 for training all classes, 11 for numbers only, etc.
    desiredClasses.append(i)
newLen=0
for classID in desiredClasses:
    newLen += np.count_nonzero(trainLabels == classID)
filteredTrainImages = np.empty((newLen, len(flatTrainImages[0])))
filteredTrainLabels = np.empty((newLen))
i=0
for j in range(len(trainLabels)):
    if trainLabels[j] in desiredClasses:
        filteredTrainImages[i] = flatTrainImages[j]
        filteredTrainLabels[i] = trainLabels[j]
        i+=1

print("Finished opening training data.")
print("Training SVC with all training data.", end='\r')

clf = SVC(gamma=0.001, C=100)
clf.fit(filteredTrainImages, filteredTrainLabels)

dump(clf, 'Data/clf.joblib') 

print("Finished training SVC with all training data.")