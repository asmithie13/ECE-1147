from PIL import Image
import pandas as pd
import numpy as np
import data_preprocessing
import os

abortion_dev_data: pd.DataFrame
abortion_train_data: pd.DataFrame
gc_dev_data: pd.DataFrame
gc_train_data: pd.DataFrame

abortion_dev_dict: dict
abortion_train_dict: dict
gc_dev_dict: dict
gc_train_dict: dict

from transformers import BlipProcessor, BlipForConditionalGeneration

[abortion_dev_data, abortion_dev_dict, abortion_train_data, abortion_train_dict,
            gc_dev_data, gc_dev_dict, gc_train_data, gc_train_dict] = data_preprocessing.data_preproc()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

absPath = os.path.dirname(__file__)
imgIn = os.path.join(absPath, "data/images/")
dataIn = os.path.join(absPath, "data/")
abortionList = os.listdir(os.path.join(imgIn, "abortion/"))
gunControlList = os.listdir(os.path.join(imgIn, "gun_control/"))

imInd = gc_dev_data.iat[5, 0]
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
print(result)