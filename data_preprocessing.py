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

def data_preproc():
    #abortion training data
    filePath = os.path.join(dataIn, "abortion_train.csv")
    abortion_train_data = pd.read_csv(filePath)
    abortion_train_data = abortion_train_data.mask(abortion_train_data == "oppose", False)
    abortion_train_data = abortion_train_data.mask(abortion_train_data == "support", True)
    abortion_train_data = abortion_train_data.mask(abortion_train_data == "no", False)
    abortion_train_data = abortion_train_data.mask(abortion_train_data == "yes", True)
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
            img = np.zeros((100, 100))
        abortion_train_dict.update({imInd: img})

    #abortion dev data
    filePath = os.path.join(dataIn, "abortion_dev.csv")
    abortion_dev_data = pd.read_csv(filePath)
    abortion_dev_data = abortion_dev_data.mask(abortion_dev_data == "oppose", False)
    abortion_dev_data = abortion_dev_data.mask(abortion_dev_data == "support", True)
    abortion_dev_data = abortion_dev_data.mask(abortion_dev_data == "no", False)
    abortion_dev_data = abortion_dev_data.mask(abortion_dev_data == "yes", True)

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
            img = np.zeros((100, 100))
        abortion_dev_dict.update({imInd: img})

    #gun control train data
    filePath = os.path.join(dataIn, "gun_control_train.csv")
    gc_train_data = pd.read_csv(filePath)
    gc_train_data = gc_train_data.mask(gc_train_data == "oppose", False)
    gc_train_data = gc_train_data.mask(gc_train_data == "support", True)
    gc_train_data = gc_train_data.mask(gc_train_data == "no", False)
    gc_train_data = gc_train_data.mask(gc_train_data == "yes", True)

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
            img = np.zeros((100, 100))
        gc_train_dict.update({imInd: img})

    #gun control dev data
    filePath = os.path.join(dataIn, "gun_control_dev.csv")
    gc_dev_data = pd.read_csv(filePath)
    gc_dev_data = gc_dev_data.mask(gc_dev_data == "oppose", False)
    gc_dev_data = gc_dev_data.mask(gc_dev_data == "support", True)
    gc_dev_data = gc_dev_data.mask(gc_dev_data == "no", False)
    gc_dev_data = gc_dev_data.mask(gc_dev_data == "yes", True)

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
            img = np.zeros((100, 100))
        gc_dev_dict.update({imInd: img})
    
    return [abortion_dev_data, abortion_dev_dict, abortion_train_data, abortion_train_dict,
            gc_dev_data, gc_dev_dict, gc_train_data, gc_train_dict]