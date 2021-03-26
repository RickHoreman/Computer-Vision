import scipy.io as sio
import numpy as np
import skimage
from skimage.viewer import ImageViewer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import load

clf = load('Data/clf.joblib')

with open('Data/testImages.npy', 'rb') as f:
    testImages = np.load(f)

with open('Data/testLabels.npy', 'rb') as f:
    testLabels = np.load(f)

flatTestImages = np.reshape(testImages, (len(testImages), (len(testImages[0])*len(testImages[0][0]))))

desiredClasses = []
for i in range(1, 63): #Put 63 for training all classes, 11 for numbers only, etc.
    desiredClasses.append(i)
newLen=0
for classID in desiredClasses:
    newLen += np.count_nonzero(testLabels == classID)
filteredTestImages = np.empty((newLen, len(flatTestImages[0])))
filteredTestLabels = np.empty(newLen)
i=0
for j in range(len(testLabels)):
    if(testLabels[j] in desiredClasses):
        filteredTestImages[i] = flatTestImages[j]
        filteredTestLabels[i] = testLabels[j]
        i+=1

print(np.unique(filteredTestLabels))

correct = 0
wrongIndexes = []
for i in range(len(filteredTestImages)):
    prediction = clf.predict(filteredTestImages[i:i+1])
    if prediction[0] == filteredTestLabels[i]:
        correct += 1
    else:
        wrongIndexes.append(i)
    percent = round(i/(len(filteredTestImages)-1)*100, 2)
    print("Testing accuracy: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')

print("\n{} out of {} characters correctly recognised".format(correct, len(filteredTestImages))) 
print("Accuracy: {}%\n".format(correct/len(filteredTestImages)*100))
print("The images at the following indexes (in filteredTestImages) were not recognised correctly:")
print(wrongIndexes)