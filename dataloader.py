#! /usr/bin/python
import os
import imageio
import numpy as np
from scipy.misc import imresize


class DataLoader():
    def __init__(self, datapath, height_hr, width_hr, height_lr, width_lr):

        # Store the datapath
        self.datapath = datapath
        self.height_hr = height_hr
        self.height_lr = height_lr
        self.width_hr = width_hr
        self.width_lr = width_lr

        # Get the paths for all the images
        self.img_paths = []
        for dirpath, _, filenames in os.walk(self.datapath):
            for filename in [f for f in filenames if f.endswith(".JPEG")]:
                self.img_paths.append(os.path.join(dirpath, filename))

    def load_batch(self, batch_size=1, img_paths=None):
        """Loads a batch of images from datapath folder""" 
        # TODO: In case of only one channel / images not being loaded, deal with that in a smarter way       

        # Pick a random set of images from the datapath if not already set
        if not img_paths:
            img_paths = np.random.choice(self.img_paths, size=batch_size)

        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        for img_path in img_paths:            

            # Load image and resize appropriately
            img = imageio.imread(img_path).astype(np.float)            
            img_hr = imresize(img, (self.height_hr, self.width_hr))
            img_lr = imresize(img, (self.height_lr, self.width_lr))

            print(f">> Reading image: {img_path}")
            print(f">> Image shapes: {img.shape} {img_hr.shape}, {img_lr.shape}")

            # Assert 3 channels
            if len(img.shape) == 3:

                # Keep
                imgs_hr.append(img_hr)
                imgs_lr.append(img_lr)

        # Scale images
        imgs_hr = np.array(imgs_hr) / 127.5 - 1
        imgs_lr = np.array(imgs_lr) / 127.5 - 1

        # Check that we found pictures, otherwise search again
        if len(imgs_hr) == 0:
            imgs_hr, imgs_lr = self.load_batch(batch_size=batch_size)

        # Return image batch
        return imgs_hr, imgs_lr