import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2 

from sklearn.model_selection import train_test_split

from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


####
basicpath = 'data/'
path_train = basicpath + 'train/'
path_train_masks = path_train + 'masks/'

print(">>> creating dataset ... ")
contain_salt = [] 
for _id in ids:
    image_matplotlib = imread(path_train_masks+_id+".png")
    enc = rle_encode(image_matplotlib)
    contain_salt.append(int(enc!=''))

df = pd.DataFrame({'id': ids, 'contain_salt': contain_salt})

print(df.head())
print("...")
print(df.tail())

df.to_csv('binary_dataset.csv')






