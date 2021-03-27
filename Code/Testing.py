import scipy.io as sio
import numpy as np
import os
import skimage
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
from skimage.transform import resize

mat = sio.loadmat('Data/Lists/English/Fnt/lists.20.mat')
test = mat['list'][0][0][8] #<-- changed this up repeatedly to discover the values listed below
print("Content:")
print(test)
print("Type:")
print(type(test))
print("Len:")
print(len(test))
print("shape:")
print(test.shape)

# 0 == ALLlabels
# 1 == Allnames
# 2 == classlabels
# 3 == classnames
# 4 == NUMclasses
# 5 == TRNind
# 6 == TSTind
# 7 == VALind
# 8 == TXNind