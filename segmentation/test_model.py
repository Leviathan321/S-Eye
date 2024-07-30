import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from config import Config
from self_driving_car_batch_generator import Generator
from utils import get_driving_styles, mapping, mappingrgb
from utils_models import *

np.random.seed(0)

# load image and test model on image    

Seg_model = U_Net(3, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/mnt/c/Unet/SegmentationModel.pth'
Seg_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
Seg_model.eval()