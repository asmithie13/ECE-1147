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

[abortion_dev_data, abortion_dev_dict, abortion_train_data, abortion_train_dict,
            gc_dev_data, gc_dev_dict, gc_train_data, gc_train_dict] = data_preprocessing.data_preproc()

import datasets

dataset = datasets.load_dataset("glue", "mnli")