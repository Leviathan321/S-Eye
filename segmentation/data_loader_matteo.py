import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from collections import namedtuple
from tqdm import tqdm
import cv2 as cv2
import re

class DatasetMatteo(Dataset):
    def __init__(self, 
                root, 
                images_dir = None , 
                targets_dir = None,
                split='train', augment=False):
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(datadir, 'images') if images_dir is None else images_dir
        self.targets_dir = os.path.join(datadir, 'targets') if targets_dir is None else targets_dir
        self.split = split
        self.augment = augment
        self.images = []
        self.targets = []
        #150, 76
        self.mapping = {
            149: 0,
            29: 1
        }
        self.mappingrgb = {
            149: (0, 255, 0),
            29: (255, 0, 0),

        }
        self.num_classes = 2

        num_all = len(os.listdir(self.images_dir))
       
        def extract_number_from_filename(filename):
            # Define a regular expression pattern to find numbers
            pattern = r'\d+'
            # Find all numbers in the filename
            numbers = re.findall(pattern, filename)
            # Join numbers into a single string or return a list of numbers
            return numbers[0]
        # =============================================
        # Read in the paths to all images
        # =============================================
        #for case of original jpg and segmentation png use 3 and 5 else use 2
        for i, file_name in enumerate(os.listdir(self.images_dir)):
            self.images.append(os.path.join(self.images_dir, file_name))
            #target_name = 'segmentation_{}'.format('_'.join(file_name.split('_')[1:]))
            file_name_without_extension = file_name.split('.')[0]
            # target_name = 'segmentation_{}.png'.format('_'.join(file_name_without_extension.split('_')[1:]))
            target_name = f'segmentation_image_{extract_number_from_filename(file_name)}.jpg'
            # target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'gtFine_color.png')
            self.targets.append(os.path.join(self.targets_dir, target_name))
        
        print(f"num_all: {num_all}")
        if self.split == "train":
            self.images = self.images[10: int(num_all * 0.9)]
            self.targets = self.targets[10: int(num_all * 0.9)]

        elif self.split == "val":
            self.images = self.images[int(num_all * 0.9):int(num_all * 0.95)]
            self.targets = self.targets[int(num_all * 0.9):int(num_all * 0.95)]
        else:
            self.images = self.images[:int(num_all * 0.95)]
            self.targets = self.targets[:int(num_all * 0.95)]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Augment: {}\n'.format(self.augment)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def __len__(self):
        return len(self.images)

    def mask_to_class(self, mask):
        '''
        Given the cityscapes dataset, this maps to a 0..classes numbers.
        This is because we are using a subset of all masks, so we have this "mapping" function.
        This mapping function is used to map all the standard ids into the smaller subset.
        '''
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        '''
        Given the Cityscapes mask file, this converts the ids into rgb colors.
        This is needed as we are interested in a sub-set of labels, thus can't just use the
        standard color output provided by the dataset.
        '''
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        '''
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        '''
        mask2class = dict((v, k) for k, v in self.mapping.items())
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
        return rgbimg

    def __getitem__(self, index):
        # Load the RGB image and target
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index]).convert('L')

        # print(target.shape)
        # print(target[0:10,0:100])
        # input()

        # Apply random transformations if augmenting,
        # otherwise, just resize the image to the correct size
        if self.split == 'train':
            if self.augment:
                # Resize
                image = TF.resize(image, size=(256 + 20, 512 + 20), interpolation=Image.BILINEAR)
                target = TF.resize(target, size=(256 + 20, 512 + 20), interpolation=Image.NEAREST)
                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 512))
                image = TF.crop(image, i, j, h, w)
                target = TF.crop(target, i, j, h, w)
                # Random horizontal flipping
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    target = TF.hflip(target)

            else:
                # Resize
                image = TF.resize(image, size=(160, 320), interpolation=Image.BILINEAR)
                target = TF.resize(target, size=(160, 320), interpolation=Image.NEAREST)
        if self.split == 'val':
            # Also resize for the validation/testing set
            image = TF.resize(image, size=(160, 320), interpolation=Image.BILINEAR)
            target = TF.resize(target, size=(160, 320), interpolation=Image.NEAREST)

        # Convert the image and target into PyTorch tensors
        image = TF.to_tensor(image)
        target = torch.from_numpy(np.array(target, dtype=np.uint8))

        # Now normalize
        image = self.normalize(image)

        # Convert the target into the desired format
        targetmask = self.mask_to_class(target)
        targetmask = targetmask.long()
        targetrgb = self.mask_to_rgb(target)
        targetrgb = targetrgb.long()

        # Finally, return the image pair
        return image, targetmask, targetrgb


if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    def denormalize(tensor):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        dtype = tensor.dtype
        mean = torch.tensor(mean).to(dtype).reshape(1, 3, 1, 1)
        std = torch.tensor(std).to(dtype).reshape(1, 3, 1, 1)
        return tensor * std + mean


    def tensor_to_pil(tensor):
        """
        Convert a PyTorch tensor to a PIL Image.
        """
        return Image.fromarray(tensor.byte().cpu().numpy().astype(np.uint8).transpose(1, 2, 0))


    datadir = r"C:\Users\sorokin\Documents\testing\segment\images_matteo"
    images_dir = r"C:\Users\sorokin\Documents\testing\segment\images_matteo\beamng"
    labels_dir = r"C:\Users\sorokin\Documents\testing\segment\output_matteo\beamng"
    train_dataset = DatasetMatteo(datadir,
                                  images_dir,
                                  labels_dir, 
                                  split='train', 
                                  augment=True)
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=4, 
                                    shuffle=True, 
                                    num_workers=0)
    val_iter = iter(train_dataloader)
    images, masks, _ = next(val_iter)  # Get a batch
    images, masks = images.to('cpu'), masks.to('cpu')
    images2, masks2 = images.to('cpu'), masks.to('cpu')

    idx = random.randint(0, images.size(0) - 1)
    image, mask = images[idx:idx + 1], masks[idx:idx + 1]  # Select a single image and mask
    image2, mask2 = images2[idx:idx + 1], masks2[idx:idx + 1]

    image = image2[0]  # Extract a single image from the batch

    image = denormalize(image).cpu()
    print(image.shape)
    image = image.squeeze().permute(1, 2, 0).numpy()
    plt.imshow(mask2)
    plt.show()
