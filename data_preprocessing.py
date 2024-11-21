import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os

absPath = os.path.dirname(__file__)
imgIn = os.path.join(absPath, "data/images/")
dataIn = os.path.join(absPath, "data/")
abortionList = os.listdir(os.path.join(imgIn, "abortion/"))
gunControlList = os.listdir(os.path.join(imgIn, "gun_control/"))

#abortion training data
filePath = os.path.join(dataIn, "abortion_train.csv")
abortion_train_data = pd.read_csv(filePath)

#reading training images in abortion folder
abortion_train_dict = {}
for ind in range(abortion_train_data.shape[0]):
    imInd = abortion_train_data.iat[ind, 0]
    imName = str(imInd) + ".jpg"
    imPath = os.path.join(imgIn, "abortion/", imName)
    try:
        with Image.open(imPath) as img:
            img.verify()
    except:
        print(imName, "invalid image")
    img = np.asarray(img)
    abortion_train_dict.update({imInd: [img,
                                abortion_train_data.iat[ind, 3],
                                abortion_train_data.iat[ind, 4]]})

#abortion dev data
filePath = os.path.join(dataIn, "abortion_dev.csv")
abortion_dev_data = pd.read_csv(filePath)

#reading training images in abortion folder
abortion_dev_dict = {}
for ind in range(abortion_dev_data.shape[0]):
    imInd = abortion_dev_data.iat[ind, 0]
    imName = str(imInd) + ".jpg"
    imPath = os.path.join(imgIn, "abortion/", imName)
    try:
        with Image.open(imPath) as img:
            img.verify()
    except:
        print(imName, "invalid image")
    img = np.asarray(img)
    abortion_dev_dict.update({imInd: [img,
                                abortion_dev_data.iat[ind, 3],
                                abortion_dev_data.iat[ind, 4]]})

#gun control train data
filePath = os.path.join(dataIn, "gun_control_train.csv")
gc_train_data = pd.read_csv(filePath)

#reading training images in gun control folder
gc_train_dict = {}
for ind in range(gc_train_data.shape[0]):
    imInd = gc_train_data.iat[ind, 0]
    imName = str(imInd) + ".jpg"
    imPath = os.path.join(imgIn, "gun_control/", imName)
    try:
        with Image.open(imPath) as img:
            img.verify()
    except:
        print(imName, "invalid image")
    img = np.asarray(img)
    gc_train_dict.update({imInd: [img,
                                gc_train_data.iat[ind, 3],
                                gc_train_data.iat[ind, 4]]})

#gun control dev data
filePath = os.path.join(dataIn, "gun_control_dev.csv")
gc_dev_data = pd.read_csv(filePath)

#reading training images in gun control folder
gc_dev_dict = {}
for ind in range(gc_dev_data.shape[0]):
    imInd = gc_dev_data.iat[ind, 0]
    imName = str(imInd) + ".jpg"
    imPath = os.path.join(imgIn, "gun_control/", imName)
    try:
        with Image.open(imPath) as img:
            img.verify()
    except:
        print(imName, "invalid image")
    img = np.asarray(img)
    gc_dev_dict.update({imInd: [img,
                                gc_dev_data.iat[ind, 3],
                                gc_dev_data.iat[ind, 4]]})