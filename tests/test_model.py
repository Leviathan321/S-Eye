import datetime
import os
import time

from segmentation.U_Net import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from matplotlib import image as mpimg

from config import Config
from PIL import Image
from xai import evaluation_segmentation

np.random.seed(0)

# load image and test model on image    

model = U_Net(3, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "/home/lev/Projects/testing/Multi-Simulation/S-Eye/models/SegmentationModel_CrossEntropyLoss49.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

###########

# data path
img_folder_path_data = "/home/lev/Projects/testing/Multi-Simulation/segment/images_matteo/donkey"

# labels path
img_folder_path_label = "/home/lev/Projects/testing/Multi-Simulation/segment/output_matteo/donkey"

# path writing predictions
output_path = "/home/lev/Projects/testing/Multi-Simulation/S-Eye/evaluation"

evaluation_segmentation.evaluation_sep_folders(model, 
                                image_folder_path = img_folder_path_data,
                               label_folder_path = img_folder_path_label,
                               num = 10,
                               path_output_segmentation = output_path)

# path = '/home/lev/Projects/testing/Multi-Simulation/S-Eye/data/images/train/image_0.jpg'
# image = mpimg.imread(path)

# img = image



# image = preprocess(image)

# with torch.no_grad():
#     prediction = model(image)
# prediction.save("./eval/prediction.png")
# predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
# maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
# predicted_rgb = class_to_rgb(maxindex).to('cpu')
# predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

# fig, axs = plt.subplots(1, 2)
# plt.tight_layout()
# axs[0].imshow(img)
# axs[0].axis('off')

# axs[1].imshow(predicted_rgb)
# axs[1].axis('off')

# plt.savefig('./eval/fail side normal.svg', bbox_inches='tight')

# plt.show()