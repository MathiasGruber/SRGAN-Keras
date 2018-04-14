import os
import imageio
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt


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
            # print(f">> Image shapes: {img.shape} {img_hr.shape}, {img_lr.shape} - {img_path}")

            # Store images
            imgs_hr.append(self.scale_imgs(img_hr))
            imgs_lr.append(self.scale_imgs(img_lr))

        # Scale images
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_hr, imgs_lr


def plot_test_images(model, loader, test_images, test_output, epoch):

    # Load the images to perform test on images
    imgs_hr, imgs_lr = loader.load_batch(batch_size=1, img_paths=test_images, training=False)

    # Create super resolution images
    imgs_sr = []
    for img in imgs_lr:
        imgs_sr.append(
            np.squeeze(
                model.generator.predict(
                    np.expand_dims(img, 0),
                    batch_size=1
                ),
                axis=0
            )
        )

    # Loop through images
    for img_hr, img_lr, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_sr, test_images):

        # Get the filename
        filename = os.path.basename(img_path).split(".")[0]

        # Images and titles
        images = {
            'Low Resolution': img_lr, 'SRGAN': img_sr, 'Original': img_hr
        }

        # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1                    
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (title, img) in enumerate(images.items()):
            axes[i].imshow(0.5 * img + 0.5)
            axes[i].set_title(title)
            axes[i].axis('off')
        plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

        # Save directory                    
        savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
        fig.savefig(savefile)
        plt.close()
