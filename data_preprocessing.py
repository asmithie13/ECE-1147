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

from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

imInd = gc_dev_data.iat[10, 0]
imName = str(imInd) + ".jpg"
imPath = os.path.join(imgIn, "gun_control/", imName)
img = Image.open(imPath).convert("RGB")
inputs = processor(img, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

'''
BLIP can kind of describe an image. very basic, kind of wrong

'''

from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

imInd = gc_dev_data.iat[5, 0]
imName = str(imInd) + ".jpg"
imPath = os.path.join(imgIn, "gun_control/", imName)
img = Image.open(imPath).convert("RGB")

inputs = processor(
    text=["a photo against gun control", "a photo supporting gun control"], images=img, return_tensors="pt", padding=True
)
''' 
IDEA:
use CLIP to ask if tweet caption matches tweet image
use CLIP to ask if image seems to support or oppose gun control

'''

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) 
print(probs)

'''
easyOCR extracts text from images. use with CLIP?
'''

import easyocr

imInd = gc_dev_data.iat[15, 0]
imName = str(imInd) + ".jpg"
imPath = os.path.join(imgIn, "gun_control/", imName)
reader = easyocr.Reader(['en'])
result = reader.readtext(imPath)