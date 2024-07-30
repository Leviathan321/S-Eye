import sys 
sys.path.insert(0,r"C:\Users\sorokin\Documents\testing\segment")

import torch
import torch.nn as nn
from segmentation.data_loader_matteo import DatasetMatteo
from torch.utils.data import DataLoader
from segmentation.U_Net import U_Net

# Provided Trainer class
from segmentation.trainer import Trainer  # Ensure Trainer is imported from the correct file

# Set the training hyperparameters
# datadir = r'C:\Users\sorokin\Documents\testing\S-Eye\data\\'

batch_size = 4
lr = 0.001
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' TODO readin matteos images, apply mask based labeling, pass to model '''

datadir = r"C:\Users\sorokin\Documents\testing\segment\images_matteo"
images_dir = r"C:\Users\sorokin\Documents\testing\segment\images_matteo\beamng"
labels_dir = r"C:\Users\sorokin\Documents\testing\segment\output_matteo\beamng"

# Initialize the dataset and dataloader
train_dataset = DatasetMatteo(datadir, 
                            images_dir = images_dir,
                            targets_dir = labels_dir,
                            split='train', 
                            augment=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = DatasetMatteo(datadir, 
                            images_dir = images_dir,
                            targets_dir = labels_dir,
                            split='val', 
                            augment=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize the model output_ch is the num of classe
model = U_Net(3, 2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()


# Create an instance of Trainer
trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  optimizer=optimizer,
                  loss_func=loss_func,
                  device=device)

# Start training

trainer.run(epochs=epochs)
