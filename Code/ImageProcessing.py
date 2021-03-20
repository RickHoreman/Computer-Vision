import scipy.io as sio
import numpy as np
import os
import skimage
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
from skimage.transform import resize

mat = sio.loadmat('Data/Lists/English/Img/lists.mat')
allNames = mat['list'][0][0][0]
trainIndexes = mat['list'][0][0][8]
testIndexes = mat['list'][0][0][6]
allLabels = mat['list'][0][0][2]

allImages = np.empty((len(allNames), 50, 50, 3))
trainImages = np.empty((len(trainIndexes)*len(trainIndexes[0]), 50, 50, 3))
trainLabels = np.empty((len(trainIndexes)*len(trainIndexes[0])))
testImages = np.empty((len(testIndexes)*len(testIndexes[0]), 50, 50, 3))
testLabels = np.empty((len(testIndexes)*len(testIndexes[0])))

for i in range(len(allNames)):  #Opening and resizing all images
    filename = os.path.join('Data/English/Img/' + allNames[i] + '.png')
    allImages[i] = resize(skimage.io.imread(filename),(50,50, 3)) #Not sure if this method of resizing is optimal, but I'm using it for now
    percent = round(i/(len(allNames)-1)*100, 2)
    print("Opening & resizing all images: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')
print()

for i in range(len(trainIndexes)):
    for j in range(len(trainIndexes[i])):
        if trainIndexes[i][j] > 0:
            trainIndex = trainIndexes[i][j]-1 #Indexes in the list start on 1 :/ but to make things even worse there are also 0's in the list
            trainImages[(i*len(trainIndexes[i]))+j] = allImages[trainIndex] 
            trainLabels[(i*len(trainIndexes[i]))+j] = allLabels[trainIndex]
            percent = round(((i*len(trainIndexes[i]))+j)/((len(trainIndexes)*len(trainIndexes[i]))-1)*100, 2)
            print("Selecting and labeling training images: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')
print()

for i in range(len(testIndexes)):
    for j in range(len(testIndexes[i])):
        if testIndexes[i][j] > 0:
            testIndex = testIndexes[i][j]-1
            testImages[(i*len(testIndexes[i]))+j] = allImages[testIndex]
            testLabels[(i*len(testIndexes[i]))+j] = allLabels[testIndex]
            percent = round(((i*len(testIndexes[i]))+j)/((len(testIndexes)*len(testIndexes[i]))-1)*100, 2)
            print("Selecting and labeling testing images: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')
print()

with open('Data/npy/trainImages.npy', 'wb') as f:
    np.save(f, trainImages)
with open('Data/npy/trainLabels.npy', 'wb') as f:
    np.save(f, trainLabels)
with open('Data/npy/testImages.npy', 'wb') as f:
    np.save(f, testImages)
with open('Data/npy/testLabels.npy', 'wb') as f:
    np.save(f, testLabels)

print("Done!")

# with open('allImages.npy', 'rb') as f:
#     allImages = np.load(f)

# viewer = ImageViewer(allImages[700])
# viewer.show()
#print(allImages)