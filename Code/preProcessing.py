import scipy.io as sio
import numpy as np
import skimage
import skimage.filters as skfilt
import matplotlib.pyplot as plt
import random

#Meant to be expandable with more pre-processing types
def applyPreProcessing(images, ppType, show=False):
    resultImages = np.empty((len(images), len(images[0]), len(images[0][0])))
    for i in range(len(images)):
        if(ppType == "sobel" or ppType == "s"):
            resultImages[i] = skfilt.sobel(images[i])
        else:
            print("preProcessing.applyPreProcessing: Incorrect type! Returning unprocessed images")
            return images
    if show:
        rIndex = random.randrange(len(images))

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharex=True, sharey=True)

        ax0.imshow(images[rIndex])
        ax0.axis('off')
        ax0.set_title('Original', fontsize=20)

        ax1.imshow(resultImages[rIndex])
        ax1.axis('off')
        ax1.set_title(ppType, fontsize=20)

        fig.tight_layout()

        plt.show()
    return resultImages