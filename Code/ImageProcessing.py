import scipy.io as sio
import numpy as np
import os
import skimage
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
from skimage.transform import resize

mat = sio.loadmat('Data/Lists/English/Fnt/lists.20.mat')
allNames = mat['list'][0][0][1]
trainIndexes = mat['list'][0][0][5]
testIndexes = mat['list'][0][0][6]
allLabels = mat['list'][0][0][0]

allImages = np.empty((len(allNames), 32,32))
trainImages = np.empty((len(trainIndexes)*len(trainIndexes[0]), 32,32))
trainLabels = np.empty((len(trainIndexes)*len(trainIndexes[0])))
testImages = np.empty((len(testIndexes)*len(testIndexes[0]), 32,32))
testLabels = np.empty((len(testIndexes)*len(testIndexes[0])))

for i in range(len(allNames)):  #Opening and resizing all images
    filename = os.path.join('Data/English/Fnt/' + allNames[i] + '.png')
    allImages[i] = resize(skimage.io.imread(filename),(32,32)) #Not sure if this method of resizing is optimal, but I'm using it for now
    percent = round(i/(len(allNames)-1)*100, 2)
    print("Opening & resizing all images: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')
print()

for i in range(len(trainIndexes)):
    for j in range(len(trainIndexes[i])):
        trainIndex = trainIndexes[i][j]-1 #Indexes in the list start on 1 :/
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

print("Saving to files", end='\r')
with open('Data/trainImages.npy', 'wb') as f:
    np.save(f, trainImages)
print("Saving to files.", end='\r')
with open('Data/trainLabels.npy', 'wb') as f:
    np.save(f, trainLabels)
print("Saving to files..", end='\r')
with open('Data/testImages.npy', 'wb') as f:
    np.save(f, testImages)
print("Saving to files...")
with open('Data/testLabels.npy', 'wb') as f:
    np.save(f, testLabels)

print("Done!")

# with open('allImages.npy', 'rb') as f:
#     allImages = np.load(f)

# viewer = ImageViewer(allImages[700])
# viewer.show()
#print(allImages)