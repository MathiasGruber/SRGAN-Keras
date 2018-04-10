#! /usr/bin/python
import os
import imageio
import numpy as np
from scipy.misc import imresize


class DataLoader():
    def __init__(self, datapath, height_hr, width_hr, height_lr, width_lr, scale):

        # Store the datapath
        self.datapath = datapath
        self.height_hr = height_hr
        self.height_lr = height_lr
        self.width_hr = width_hr
        self.width_lr = width_lr
        self.scale = scale

        # Get the paths for all the images
        self.img_paths = []
        for dirpath, _, filenames in os.walk(self.datapath):
            for filename in [f for f in filenames if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]:
                self.img_paths.append(os.path.join(dirpath, filename))
        print(f">> Found {len(self.img_paths)} images in dataset")

    def get_random_images(self, n_imgs=1):
        """Get n_imgs random images from the dataset"""
        return np.random.choice(self.img_paths, size=n_imgs)

    def scale_imgs(self, imgs):
        """Scale images prior to passing to SRGAN"""
        return imgs / 127.5 - 1

    def load_batch(self, batch_size=1, img_paths=None, training=True):
        """Loads a batch of images from datapath folder""" 

        # Pick a random set of images from the datapath if not already set
        if not img_paths:
            img_paths = self.get_random_images(batch_size)

        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        for img_path in img_paths:            

            # Load image
            img = imageio.imread(img_path).astype(np.float)            

            # If gray-scale, convert to RGB
            if len(img.shape) == 2:
                img = np.stack((img,)*3, -1)

            # Resize images appropriately
            if training:
                img_hr = imresize(img, (self.height_hr, self.width_hr))
                img_lr = imresize(img, (self.height_lr, self.width_lr))
            else:
                lr_shape = (int(img.shape[0]/self.scale), int(img.shape[1]/self.scale))
                img_hr = np.array(img)
                img_lr = imresize(img, lr_shape)

            # print(f">> Reading image: {img_path}")
            # print(f">> Image shapes: {img.shape} {img_hr.shape}, {img_lr.shape}")

            # Store images
            imgs_hr.append(self.scale_imgs(img_hr))
            imgs_lr.append(self.scale_imgs(img_lr))

        # Scale images
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_hr, imgs_lr