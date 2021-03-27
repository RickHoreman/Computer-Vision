import scipy.io as sio
import numpy as np
import os
import skimage
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
from skimage.transform import resize
import random

newSize = 32

mat = sio.loadmat('Data/Lists/English/Fnt/lists.20.mat')
allNames = mat['list'][0][0][1]
allLabels = mat['list'][0][0][0]

testIndexes = []
trainIndexes = []
for i in range(len(allLabels)):
    if random.randrange(3) == 0:
        testIndexes.append(i)
    else:
        trainIndexes.append(i)

allImages = np.empty((len(allNames), newSize,newSize))
trainImages = np.empty((len(trainIndexes), newSize,newSize))
trainLabels = np.empty((len(trainIndexes)))
testImages = np.empty((len(testIndexes), newSize,newSize))
testLabels = np.empty((len(testIndexes)))

for i in range(len(allNames)):  #Opening and resizing all images
    filename = os.path.join('Data/English/Fnt/' + allNames[i] + '.png')
    allImages[i] = resize(skimage.io.imread(filename),(newSize,newSize)) #Not sure if this method of resizing is optimal, but I'm using it for now
    percent = round(i/(len(allNames)-1)*100, 2)
    print("Opening & resizing all images: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')
print()

for i in range(len(trainIndexes)):
    trainImages[i] = allImages[trainIndexes[i]]
    trainLabels[i] = allLabels[trainIndexes[i]]
    percent = round(i/(len(trainIndexes)-1)*100, 2)
    print("Selecting and labeling training images: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')
print()

for i in range(len(testIndexes)):
    testImages[i] = allImages[testIndexes[i]]
    testLabels[i] = allLabels[testIndexes[i]]
    percent = round(i/(len(testIndexes)-1)*100, 2)
    print("Selecting and labeling testing images: [{:20s}] {}%".format('='*int(percent//5), percent), end='\r')
print()

print("Saving to files              {} training images".format(len(trainImages)))
with open('Data/trainImages.npy', 'wb') as f:
    np.save(f, trainImages)
print("Saving to files.             and their {} labels.".format(len(trainLabels)))
with open('Data/trainLabels.npy', 'wb') as f:
    np.save(f, trainLabels)
print("Saving to files..            Plus {} testing images".format(len(testImages)))
with open('Data/testImages.npy', 'wb') as f:
    np.save(f, testImages)
print("Saving to files...           and their {} labels.".format(len(testLabels)))
with open('Data/testLabels.npy', 'wb') as f:
    np.save(f, testLabels)
print("Adding up to a total of {} images and their labels".format(len(trainImages) + len(testImages)))

print("Done!")