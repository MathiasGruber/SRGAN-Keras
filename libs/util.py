import os
import numpy as np
from PIL import Image
from scipy.misc import imresize
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, datapath, height_hr, width_hr, scale):
        """        
        :param string datapath: filepath to training images
        :param int height_hr: Height of high-resolution images
        :param int width_hr: Width of high-resolution images
        :param int height_hr: Height of low-resolution images
        :param int width_hr: Width of low-resolution images
        :param int scale: Upscaling factor
        """

        # Store the datapath
        self.datapath = datapath
        self.height_hr = height_hr
        self.height_lr = int(height_hr / scale)
        self.width_hr = width_hr
        self.width_lr = int(width_hr / scale)
        self.scale = scale
        self.total_imgs = None

        # Get the paths for all the images
        self.img_paths = []
        for dirpath, _, filenames in os.walk(self.datapath):
            for filename in [f for f in filenames if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]:
                self.img_paths.append(os.path.join(dirpath, filename))
        self.total_imgs = len(self.img_paths)
        print(f">> Found {self.total_imgs} images in dataset")
    
    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        print(height, width)
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    def scale_lr_imgs(self, imgs):
        """Scale low-res images prior to passing to SRGAN"""
        return imgs / 255
    
    def unscale_lr_imgs(self, imgs):
        """Un-Scale low-res images"""
        return imgs * 255
    
    def scale_hr_imgs(self, imgs):
        """Scale high-res images prior to passing to SRGAN"""
        return imgs / 127.5 - 1
    
    def unscale_hr_imgs(self, imgs):
        """Un-Scale high-res images"""
        return (imgs + 1) * 127.5

    def load_batch(self, batch_size=1, img_paths=None, training=True):
        """Loads a batch of images from datapath folder""" 

        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        while len(imgs_hr) < batch_size:
            try: 
                # Load image   
                rand = np.random.randint(0, self.total_imgs)
                img = np.array(Image.open(self.img_paths[rand])).astype(np.float)

                # If gray-scale, convert to RGB
                if len(img.shape) == 2:
                    img = np.stack((img,)*3, -1)

                # For HR, do a random crop as in paper if training
                img_hr = np.array(img)
                if training:
                    img_hr = self.random_crop(img_hr, (self.height_hr, self.width_hr))
                    #img_lr = self.random_crop(img_lr, (self.height_lr, self.width_lr))

                # For LR, do bicubic downsampling
                lr_shape = (int(img_hr.shape[0]/self.scale), int(img_hr.shape[1]/self.scale))            
                img_lr = imresize(img_hr, lr_shape)

                # Scale color values
                img_hr = self.scale_lr_imgs(img_hr)
                img_lr = self.scale_lr_imgs(img_lr)

                # Store images
                imgs_hr.append(img_hr)
                imgs_lr.append(img_lr)
                
            except:
                pass

        # Convert to numpy arrays when we are training
        if training:
            try:
                imgs_hr = np.array(imgs_hr)
                imgs_lr = np.array(imgs_lr)
            except:
                raise Exception("Something went wrong: LR: {}, HR: {}".format(
                    [im.shape for im in imgs_lr], [im.shape for im in imgs_hr]
                ))

        # Return image batch
        return imgs_hr, imgs_lr


def plot_test_images(model, loader, test_images, test_output, epoch):
    """        
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param list test_images: List of filepaths for testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    # Load the images to perform test on images
    imgs_hr, imgs_lr = loader.load_batch(batch_size=1, img_paths=test_images, training=False)

    # Create super resolution and bicubic interpolation images
    imgs_sr = []
    imgs_bc = []
    for i in range(len(test_images)):
        
        # Bicubic interpolation
        pil_img = loader.unscale_imgs(imgs_lr[i]).astype('uint8')
        pil_img = Image.fromarray(pil_img)
        hr_shape = (imgs_hr[i].shape[1], imgs_hr[i].shape[0])
        
        imgs_bc.append(
            loader.scale_imgs(
                np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
            )
        )
        
        # SRGAN prediction
        imgs_sr.append(
            np.squeeze(
                model.generator.predict(
                    np.expand_dims(imgs_lr[i], 0),
                    batch_size=1
                ),
                axis=0
            )
        )

    # Loop through images
    for img_hr, img_lr, img_bc, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_bc, imgs_sr, test_images):

        # Get the filename
        filename = os.path.basename(img_path).split(".")[0]

        # Images and titles
        images = {
            'Low Resolution': img_lr, 
            'Bicubic Interpolation': img_bc, 
            'SRGAN': img_sr, 
            'Original': img_hr
        }

        # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1                    
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, (title, img) in enumerate(images.items()):
            axes[i].imshow(0.5 * img + 0.5)
            axes[i].set_title("{} - {}".format(title, img.shape))
            axes[i].axis('off')
        plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

        # Save directory                    
        savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
        fig.savefig(savefile)
        plt.close()
